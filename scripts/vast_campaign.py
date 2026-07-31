#!/usr/bin/env python3
"""One-shot, manually authorized Vast API lifecycle for the cudarc LBM campaign.

This deliberately provisions exactly one offer and never retries automatically.
The instance is deleted and provider inventory is verified in a finally block.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://console.vast.ai/api/v0"
IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
GPU_NAMES = {
    "rtx4090": ["RTX 4090", "GeForce RTX 4090"],
    "a100": [
        "A100 SXM4",
        "A100-SXM4-80GB",
        "A100-SXM4-40GB",
        "A100 SXM4 80GB",
        "A100 SXM4 40GB",
    ],
}
MIN_MEMORY_MB = {"rtx4090": 23_000, "a100": 39_000}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Vast:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def request(self, method: str, path: str, body: Any | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            BASE_URL + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "neurali-tenstorrent-energy-campaign/1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Vast {method} {path} failed: HTTP {exc.code}: {detail}") from exc
        return json.loads(payload) if payload else {}

    def search(self, gpu: str, max_price: float) -> list[dict[str, Any]]:
        query = {
            "q": {
                "rentable": {"eq": True},
                "rented": {"eq": False},
                "reliability2": {"gte": 0.98},
                "dph_total": {"lte": max_price},
                "cuda_max_good": {"gte": 12.8},
                "num_gpus": {"eq": 1},
                "gpu_ram": {"gte": MIN_MEMORY_MB[gpu]},
                "disk_space": {"gte": 40},
                "direct_port_count": {"gte": 2},
                "gpu_name": {"in": GPU_NAMES[gpu]},
                "order": [["dph_total", "asc"]],
                "type": "on-demand",
                "limit": 256,
            }
        }
        offers = self.request("PUT", "/search/asks/", query).get("offers", [])
        admitted = []
        for offer in offers:
            name = str(offer.get("gpu_name", ""))
            if not any(
                expected.lower() in name.lower() or name.lower() in expected.lower()
                for expected in GPU_NAMES[gpu]
            ):
                continue
            if offer.get("rentable") is False or offer.get("rented") is True:
                continue
            if offer.get("verified") is False:
                continue
            if int(offer.get("num_gpus") or 0) != 1:
                continue
            memory = int(offer.get("gpu_total_ram") or offer.get("gpu_ram") or 0)
            if memory < MIN_MEMORY_MB[gpu]:
                continue
            if float(offer.get("dph_total") or 1e9) > max_price:
                continue
            admitted.append(offer)
        admitted.sort(
            key=lambda o: (
                o.get("verified") is not True,
                -float(o.get("reliability2") or 0),
                -float(o.get("dlperf_per_dphtotal") or 0),
                float(o.get("dph_total") or 1e9),
            )
        )
        return admitted

    def launch(self, offer_id: int, label: str) -> int:
        body = {
            "image": IMAGE,
            "disk": 40,
            "env": {},
            "target_state": "running",
            "label": label,
            "onstart": "#!/bin/bash\nmkdir -p /workspace /workspace/results\n",
            "runtype": "ssh",
        }
        result = self.request("PUT", f"/asks/{offer_id}/", body)
        instance_id = result.get("new_contract")
        if not instance_id:
            raise RuntimeError(f"Vast launch response has no new_contract: {result}")
        return int(instance_id)

    def instance(self, instance_id: int) -> dict[str, Any] | None:
        try:
            payload = self.request("GET", f"/instances/{instance_id}/")
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        value = payload.get("instances")
        if isinstance(value, list):
            return value[0] if value else None
        return value if isinstance(value, dict) else None

    def list_instances(self) -> list[dict[str, Any]]:
        value = self.request("GET", "/instances/").get("instances", [])
        return value if isinstance(value, list) else []

    def delete_verified(self, instance_id: int) -> dict[str, Any]:
        accepted_at = utc_now()
        try:
            self.request("DELETE", f"/instances/{instance_id}/")
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
        for attempt in range(1, 21):
            inventory = self.list_instances()
            if all(int(item.get("id", -1)) != instance_id for item in inventory):
                return {
                    "delete_accepted_at": accepted_at,
                    "verified_at": utc_now(),
                    "verification_attempts": attempt,
                    "instance_absent": True,
                }
            time.sleep(1.5)
        raise RuntimeError(f"instance {instance_id} remains visible after delete")


def wait_ssh(
    vast: Vast, instance_id: int, timeout_s: int = 900
) -> tuple[str, int, dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    last_state = None
    while time.monotonic() < deadline:
        instance = vast.instance(instance_id)
        if instance:
            status = instance.get("actual_status")
            message = instance.get("status_msg")
            state = (status, message)
            if state != last_state:
                print(
                    f"Vast instance {instance_id}: status={status!r} message={message!r}",
                    flush=True,
                )
                last_state = state
            if message and "Error" in str(message):
                raise RuntimeError(f"Vast provisioning error: {message}")
            host = instance.get("ssh_host")
            port = instance.get("ssh_port")
            if status == "running" and host and port:
                try:
                    with socket.create_connection((str(host), int(port)), timeout=5):
                        auth = subprocess.run(
                            ["ssh", *ssh_args(str(host), int(port)), "true"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=20,
                            check=False,
                        )
                        if auth.returncode == 0:
                            return str(host), int(port), instance
                except (OSError, subprocess.TimeoutExpired):
                    pass
        time.sleep(5)
    raise TimeoutError(
        f"instance {instance_id} was not SSH-auth-ready within {timeout_s}s"
    )


def ssh_args(host: str, port: int) -> list[str]:
    return [
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"root@{host}",
    ]


def scp_args(host: str, port: int) -> list[str]:
    return [
        "-P",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=sorted(GPU_NAMES), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-price", type=float, default=2.0)
    parser.add_argument("--max-lease-seconds", type=int, default=2700)
    parser.add_argument("--remaining-budget", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        raise RuntimeError("VAST_API_KEY is absent")
    vast = Vast(api_key)
    offers = vast.search(args.gpu, args.max_price)
    if not offers:
        raise RuntimeError(f"no admitted Vast offers for {args.gpu}")
    offer = offers[0]
    price = float(offer["dph_total"])
    worst_case_cost = price * args.max_lease_seconds / 3600.0
    public_offer = {
        key: offer.get(key)
        for key in (
            "id",
            "machine_id",
            "gpu_name",
            "num_gpus",
            "gpu_total_ram",
            "cuda_max_good",
            "dph_total",
            "reliability2",
            "inet_down",
            "inet_up",
            "geolocation",
            "verified",
            "datacenter",
            "dlperf_per_dphtotal",
            "disk_space",
            "disk_bw",
        )
    }
    print(
        json.dumps(
            {"selected_offer": public_offer, "worst_case_cost_usd": worst_case_cost},
            indent=2,
        )
    )
    if worst_case_cost > args.remaining_budget:
        raise RuntimeError(
            f"worst-case lease cost USD {worst_case_cost:.2f} exceeds remaining "
            f"budget USD {args.remaining_budget:.2f}"
        )
    if not args.execute:
        return 0

    if not args.binary.is_file() or not args.runner.is_file():
        raise RuntimeError("binary or runner does not exist")

    output_dir = args.output_root / args.gpu
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "offer.json", public_offer)

    label = f"neurali-d3q19-{args.gpu}-{int(time.time())}"
    instance_id: int | None = None
    lease_started_mono = time.monotonic()
    lease_started_at = utc_now()
    campaign_error: str | None = None
    remote_instance: dict[str, Any] | None = None
    teardown: dict[str, Any] | None = None
    try:
        instance_id = vast.launch(int(offer["id"]), label)
        write_json(
            output_dir / "launch.json",
            {
                "instance_id": instance_id,
                "label": label,
                "image": IMAGE,
                "lease_started_at": lease_started_at,
            },
        )
        host, port, remote_instance = wait_ssh(vast, instance_id)
        write_json(output_dir / "running-instance.json", remote_instance)

        subprocess.run(
            [
                "scp",
                *scp_args(host, port),
                str(args.binary),
                str(args.runner),
                f"root@{host}:/workspace/",
            ],
            check=True,
            timeout=180,
        )
        remote_cmd = (
            "chmod +x /workspace/cuda-lbm /workspace/run_gpu_campaign.sh "
            f"&& timeout {args.max_lease_seconds - 120} "
            f"/workspace/run_gpu_campaign.sh {args.gpu} /workspace/cuda-lbm"
        )
        subprocess.run(
            ["ssh", *ssh_args(host, port), remote_cmd],
            check=True,
            timeout=args.max_lease_seconds,
        )
        local_results = output_dir / "artifacts"
        if local_results.exists():
            shutil.rmtree(local_results)
        local_results.mkdir(parents=True)
        subprocess.run(
            [
                "scp",
                "-r",
                *scp_args(host, port),
                f"root@{host}:/workspace/results/{args.gpu}/.",
                str(local_results),
            ],
            check=True,
            timeout=300,
        )
        if not (local_results / "CAMPAIGN_COMPLETE").exists():
            raise RuntimeError("remote completion marker missing after artifact retrieval")
    except Exception as exc:
        campaign_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if instance_id is not None:
            try:
                teardown = vast.delete_verified(instance_id)
            except Exception as exc:
                teardown = {
                    "instance_absent": False,
                    "teardown_error": f"{type(exc).__name__}: {exc}",
                    "verified_at": utc_now(),
                }
        elapsed = time.monotonic() - lease_started_mono
        estimated_cost = price * elapsed / 3600.0
        lease = {
            "provider": "vast",
            "gpu_requested": args.gpu,
            "gpu_observed_api": (
                None if remote_instance is None else remote_instance.get("gpu_name")
            ),
            "offer_id": int(offer["id"]),
            "instance_id": instance_id,
            "image": IMAGE,
            "dollars_per_hour": price,
            "lease_started_at": lease_started_at,
            "lease_closed_at": utc_now(),
            "lease_elapsed_seconds": elapsed,
            "estimated_cost_usd": estimated_cost,
            "remaining_budget_at_start_usd": args.remaining_budget,
            "campaign_error": campaign_error,
            "teardown": teardown,
        }
        write_json(output_dir / "lease.json", lease)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

#!/usr/bin/env bash
set -euo pipefail

GPU_SLUG="${1:?usage: run_gpu_campaign.sh GPU_SLUG}"
BIN="${2:-/workspace/cuda-lbm}"
OUT="/workspace/results/${GPU_SLUG}"
mkdir -p "${OUT}/parity" "${OUT}/tuning" "${OUT}/trials" "${OUT}/profile"

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

exec > >(tee "${OUT}/campaign.stdout.log") 2> >(tee "${OUT}/campaign.stderr.log" >&2)

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/campaign.started_at"
nvidia-smi -L
nvidia-smi -q -x > "${OUT}/nvidia-smi-q.xml"
nvidia-smi   --query-gpu=timestamp,index,name,uuid,pci.bus_id,driver_version,memory.total,power.limit,pstate,clocks.sm,clocks.mem,temperature.gpu   --format=csv,noheader,nounits > "${OUT}/hardware.csv"
nvcc --version > "${OUT}/nvcc.txt" 2>&1 || true
ldd "${BIN}" > "${OUT}/binary.ldd.txt"
sha256sum "${BIN}" > "${OUT}/binary.sha256"

telemetry_pid=""
cleanup_telemetry() {
  if [[ -n "${telemetry_pid}" ]]; then
    kill "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" 2>/dev/null || true
  fi
}
trap cleanup_telemetry EXIT

nvidia-smi   --query-gpu=timestamp,index,name,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used   --format=csv,noheader,nounits -lms 100   > "${OUT}/telemetry.csv" 2> "${OUT}/telemetry.stderr.log" &
telemetry_pid=$!

run_case() {
  local prefix="$1"
  shift
  "${BIN}" "$@"     --result "${prefix}.json"     --markers "${prefix}.markers.json"     --power "${prefix}.power.csv"
}

# Strict field parity is independent of the long performance trajectory.
run_case "${OUT}/parity/parity-n16-s1"   --n 16 --warmup 0 --steps 1 --block 256 --cpu-parity
run_case "${OUT}/parity/parity-n16-s10"   --n 16 --warmup 0 --steps 10 --block 256 --cpu-parity

# A bounded, symmetric block-size search is outside every admitted window.
best_block=0
best_mlups=0
for block in 128 256 512; do
  prefix="${OUT}/tuning/block-${block}"
  run_case "${prefix}" --n 128 --warmup 2000 --steps 5000 --block "${block}"
  mlups="$(awk -F: '/"mlups"/ {gsub(/[ ,]/, "", $2); print $2; exit}' "${prefix}.json")"
  if awk -v candidate="${mlups}" -v current="${best_mlups}" 'BEGIN { exit !(candidate > current) }'; then
    best_block="${block}"
    best_mlups="${mlups}"
  fi
done
test "${best_block}" -ne 0
printf '{"best_block":%s,"pilot_mlups":%s}\n' "${best_block}" "${best_mlups}" > "${OUT}/tuning/selection.json"

# Calibrate one 70-second steady window, then freeze the step count for all trials.
cal_prefix="${OUT}/tuning/calibration"
run_case "${cal_prefix}" --n 128 --warmup 2000 --steps 10000 --block "${best_block}"
cal_elapsed="$(awk -F: '/"elapsed_s"/ {gsub(/[ ,]/, "", $2); print $2; exit}' "${cal_prefix}.json")"
steps="$(awk -v elapsed="${cal_elapsed}" 'BEGIN {
  target = int(10000.0 * 70.0 / elapsed);
  if (target < 10000) target = 10000;
  print target
}')"
printf '{"target_seconds":70,"calibration_steps":10000,"calibration_elapsed_s":%s,"measured_steps":%s}\n'   "${cal_elapsed}" "${steps}" > "${OUT}/tuning/calibrated-steps.json"

# Cool down, then record a full five-minute idle baseline without a CUDA context.
sleep 60
run_case "${OUT}/idle-300s"   --n 128 --warmup 0 --steps 1 --block "${best_block}" --idle-seconds 300

for trial in 1 2 3 4 5; do
  run_case "${OUT}/trials/trial-${trial}"     --n 128 --warmup 5000 --steps "${steps}" --block "${best_block}"
done

# Profiling is bounded and explicitly excluded from energy statistics.
if command -v ncu >/dev/null 2>&1; then
  set +e
  timeout 120 ncu     --set basic     --target-processes all     --launch-skip 5     --launch-count 1     --export "${OUT}/profile/lbm-basic"     --force-overwrite     "${BIN}"       --n 128 --warmup 0 --steps 10 --block "${best_block}"       --result "${OUT}/profile/profile-run.json"       --markers "${OUT}/profile/profile-run.markers.json"       --power "${OUT}/profile/profile-run.power.csv"     > "${OUT}/profile/ncu.stdout.log" 2> "${OUT}/profile/ncu.stderr.log"
  ncu_status=$?
  set -e
  printf '%s\n' "${ncu_status}" > "${OUT}/profile/ncu.exit-code"
else
  printf '%s\n' "ncu-unavailable" > "${OUT}/profile/ncu.exit-code"
fi

cleanup_telemetry
telemetry_pid=""
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT}/campaign.completed_at"
touch "${OUT}/CAMPAIGN_COMPLETE"

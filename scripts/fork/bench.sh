#!/bin/bash
# Repeatable before/after numbers for the fork's sprints.
#
#   scripts/fork/bench.sh PHOTO [RUNS]
#
# Renders PHOTO full size with the release CLI (cargo build --release -p lightcraft-cli first) in
# four configurations: plain, cropped, Convert Negative, cropped + Convert Negative, each on the
# GPU and on the CPU (LIGHTCRAFT_GPU=0), RUNS times (default 3). Prints the median wall time of the
# render and of the main LIGHTCRAFT_PROFILE stages. The load average is printed first: numbers
# taken while the machine is busy (load well above the core count) are not comparable.
set -euo pipefail

photo=${1:?usage: scripts/fork/bench.sh PHOTO [RUNS]}
runs=${2:-3}
root=$(cd "$(dirname "$0")/../.." && pwd)
cli="$root/target/release/lightcraft-cli"
[ -x "$cli" ] || { echo "build it first: cargo build --release -p lightcraft-cli" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

crop='"crop":{"geometry":{"rect":{"x0":0.1,"y0":0.05,"x1":0.9,"y1":0.85},"angle":0.0}}'
neg='"negative":{"enabled":true}'
off='"negative":{"enabled":false}'
echo "{$off}" > "$tmp/plain.json"
echo "{$off,$crop}" > "$tmp/crop.json"
echo "{$neg}" > "$tmp/neg.json"
echo "{$neg,$crop}" > "$tmp/neg_crop.json"
# a copy without its XMP sidecar, whose edits the CLI would apply under every configuration
cp "$photo" "$tmp/"
photo="$tmp/$(basename "$photo")"

median() { sort -n | awk '{a[NR]=$1} END {if (NR) print a[int((NR+1)/2)]; else print "-"}'; }

echo "photo: $photo"
echo "$(uptime | sed 's/.*load/load/'), $(sysctl -n hw.ncpu 2>/dev/null || nproc) cores"
printf '%-10s %-4s %9s %9s %9s %9s\n' config dev "total ms" "noise ms" "neg ms" "sample ms"
for cfg in plain crop neg neg_crop; do
  for dev in gpu cpu; do
    : > "$tmp/total"; : > "$tmp/noise"; : > "$tmp/neg"; : > "$tmp/sample"
    for _ in $(seq "$runs"); do
      # (one variable, not an array: macOS's bash 3.2 calls an empty array unbound under set -u)
      dev_env=LIGHTCRAFT_BENCH_DEVICE=gpu
      [ "$dev" = cpu ] && dev_env=LIGHTCRAFT_GPU=0
      # the total without LIGHTCRAFT_PROFILE (its raw-decode diagnostics add seconds), the stages with it
      start=$(python3 -c 'import time; print(time.time())')
      env "$dev_env" "$cli" render "$photo" -o "$tmp/out.jpg" --settings "$tmp/$cfg.json" > /dev/null 2>&1 || true
      end=$(python3 -c 'import time; print(time.time())')
      python3 -c "print(round(($end - $start) * 1000))" >> "$tmp/total"
      env "$dev_env" LIGHTCRAFT_PROFILE=1 "$cli" render "$photo" -o "$tmp/out.jpg" --settings "$tmp/$cfg.json" > "$tmp/log" 2>&1 || true
      grep -oE 'colour noise model \([0-9]+ tiles\): [0-9.]+' "$tmp/log" | awk '{print $NF}' | head -1 >> "$tmp/noise" || true
      grep -oE '^ *negative: [0-9.]+' "$tmp/log" | awk '{print $NF}' | sort -n | tail -1 >> "$tmp/neg" || true
      grep -oE 'gpu sample: [0-9.]+' "$tmp/log" | awk '{print $NF}' | head -1 >> "$tmp/sample" || true
    done
    printf '%-10s %-4s %9s %9s %9s %9s\n' "$cfg" "$dev" "$(median < "$tmp/total")" "$(median < "$tmp/noise")" "$(median < "$tmp/neg")" "$(median < "$tmp/sample")"
  done
done

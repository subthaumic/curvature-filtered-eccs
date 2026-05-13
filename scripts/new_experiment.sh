#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name_slug>"
  echo "Example: $0 compare_regularizers"
  exit 1
fi

slug="$1"

if [[ ! "$slug" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
  echo "Error: slug must match ^[a-z0-9][a-z0-9_-]*$"
  exit 1
fi

mkdir -p experiments

next_num=1
shopt -s nullglob
for path in experiments/e[0-9][0-9]_*; do
  base="$(basename "$path")"
  num="${base:1:2}"
  if [[ "$num" =~ ^[0-9]{2}$ ]]; then
    value=$((10#$num))
    if (( value >= next_num )); then
      next_num=$((value + 1))
    fi
  fi
done

if (( next_num > 99 )); then
  echo "Error: reached e99. Expand naming scheme before creating new experiments."
  exit 1
fi

exp_id="$(printf 'e%02d' "$next_num")"
exp_dir="experiments/${exp_id}_${slug}"

if [[ -e "$exp_dir" ]]; then
  echo "Error: target already exists: $exp_dir"
  exit 1
fi

mkdir -p "$exp_dir"/{analysis,config,data,results,runs}
touch "$exp_dir"/data/.gitkeep "$exp_dir"/results/.gitkeep "$exp_dir"/runs/.gitkeep

cat > "$exp_dir/README.md" <<EOF
# ${exp_id}_${slug}

## Goal
Describe the hypothesis or objective.

## Status
planned

## Analysis
- Add notebooks/scripts under \`analysis/\`.

## Data
Document data sources and local paths in \`data_manifest.yaml\`.

## Notes
- Created via \`scripts/new_experiment.sh\`.
EOF

cat > "$exp_dir/data_manifest.yaml" <<EOF
experiment_id: ${exp_id}
experiment_name: ${slug}

inputs:
  - name: local_input_data
    description: Local data used by this experiment (not committed)
    location: ${exp_dir}/data/
    checksum: null

outputs:
  - name: generated_outputs
    description: Generated outputs (not committed)
    location: ${exp_dir}/results/
EOF

cat > "$exp_dir/config/params.yaml" <<EOF
seed: 42
notes: "Experiment-local parameter file."
EOF

echo "Created experiment scaffold: $exp_dir"


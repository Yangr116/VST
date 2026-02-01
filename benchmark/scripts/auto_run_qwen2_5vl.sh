#!/bin/bash
set -x

NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_PORT=8888

model_path=$1
datetime=$(date +%Y-%m-%d_%H-%M-%S)

longest=""
IFS='/' read -ra arr <<< "$model_path"
for part in "${arr[@]}"; do
    if [[ ${#part} -gt ${#longest} ]]; then
        longest="$part"
    fi
done
basename=$longest
modified_string=$longest
work_dir="evaluation-$basename-$datetime"
echo "$work_dir"


# vlmevalkit
work_dir="work_dirs/$work_dir"
mkdir -p $work_dir

torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-port=$MASTER_PORT benchmark/run.py --config config_qwen2_5.json --model_path "$model_path" --work-dir "$work_dir" 2>&1 | tee "work_dirs/$modified_string-log.txt"

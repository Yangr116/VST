#!/bin/bash


NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_PORT=8888


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(($NNODES * $NPROC_PER_NODE))

model_path=$1
fps=${3:-4}
output_dir="$2/vsibench_fps$fps"


PIDS=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "$IDX"
    CUDA_VISIBLE_DEVICES=$IDX python benchmark/eval_vsibench.py \
        --model_name_or_path $model_path \
        --total-processor $CHUNKS \
        --processor-id $IDX \
        --batch-size 4 \
        --fps $fps \
        --output-dir $output_dir &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait $pid
done

python benchmark/utils/merge_vsibench.py --result-dir $output_dir

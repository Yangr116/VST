model=rayruiyang/VST-3B-SFT
outdir="work_dirs"
data_dir="data/vst_3d_grounding_benchmark"
dataset="sunrgbd/annotations/sunrgbd_val_total3d-prompt_v4_norm.json"
gt="sunrgbd/annotations/sunrgbd_total3d_val_metaqa_20250321.json"
image_dir="sunrgbd"
python3 benchmark/det3d/dist_inference_vllm_json.py \
    --batch_size 64 \
    --num_gpus 4 \
    --model_name_or_path=$model \
    --outdir=$outdir \
    --dataset $dataset \
    --data_dir=$data_dir \
    --image_dir=$image_dir \
    --gt-file $gt \
    --degree-range='1' \
    --enable_fov

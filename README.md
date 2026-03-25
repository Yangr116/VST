# Explore Spatial Intelligence in VLM



# Getting Start
Docker: `mariana_cv/v66`
Merlin trial:

* Train: https://ml.bytedance.net/development/instance/jobs/0c1fa73736e51a13?tabState=run_info&trialId=41174026

* Evaluation: https://ml.bytedance.net/development/instance/jobs/70db0208ef55149e?tabState=run_info


## Train (Veomni)
If you encounter the issue of "lib.so", try the following solution:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
```
Install flash attention
```
pip install flash-attn --no-build-isolation
```

the exported huggingface config has some issues:
* `_attn_implementation_autoset` should be false, or the flash_attn won't be used.
* `image_token_id` = -200, this can't be used by the hf transfomers.

```shell
bash scripts/train.sh vlm3d/train.py config/veomni/qwen2_5vl/qwen2_5_vl_fspd1_fov_uvd_packing.yaml
```
If you doesn't save huggingface weights, run the below scripts to export them:


NOTE:
* set pin_memory=False when your data has some bad points
* when token is consumed out, training will get stuck.

## Train (transformers trainer)
使用 trainer 的方式进行训练
```shell
torchrun --master_port=8889 --nproc_per_node=4 train.py --config ./config/trainconfig/qwen2vl_sft_config.yaml --per_device_train_batch_size 1
```

##  Train (old)
```shell
torchrun --nproc_per_node=8 --master_port 8888 train.py config=config/qwen2vl_7b_sunrgbd.yaml
```

### Traing with Merlin (old)
```shell
bash launch.sh train.py config=config/qwen2vl_7b_sunrgbd.yaml
```

Revise the param in the config
```shell
bash launch.sh train.py config=config/qwen2vl_7b_sunrgbd.yaml \ 
    training.output_dir="hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/work_dirs/20250204_qwen2vl_7b_sunrgbd_4gpu_bs4
```

##  Evaluation
Distributed inference using the dataset saved in HDFS.
```shell
bash launch.sh src/dist_inference.py \ 
    --model_name_or_path="hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/work_dirs/20250203_qwen2vl_7b_sunrgbd/checkpoint-10000" \ 
    --pt_weights="hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/work_dirs/20250203_qwen2vl_7b_sunrgbd/checkpoint-10000" \ 
    --outdir="hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/work_dirs/evaluate/20250203_qwen2vl_7b_sunrgbd-checkpoint-10000" \ 
    --dataset "annotations_yr/yr_sunrgbd_val_20250121.json" \ 
    --data_dir="hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/spatial_data" \ 
    --image_dir="images"
```

The results will be saved into ```hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/work_dirs/evaluate/20250203_qwen2vl_7b_sunrgbd-checkpoint-10000```


### vllm (recommended)
The attention head must be splited by  `$num_gpus`
```bash
model="$hdf_model"
dataset="annotations_yr/yr_sunrgbd_val_100_percate_qa_v1_norm_20250211.json"
gt_file="annotations/sunrgbd_val_100_20250211.json"

python3 src/inference/dist_inference_vllm.py \
    -m $model \
    -bs 64 --num_gpus 4 \
    --outdir="/opt/tiger/rayyang/spatial/work_dirs/evaluate" \
    --dataset $dataset \
    --data_dir="/opt/tiger/rayyang/spatial/data/inhouse" \
    --image_dir="images" \
    --gt-file $gt_file \
    --degree-range '1'
```

this is the reference trial: https://ml.bytedance.net/development/instance/jobs/f7d6ba8496f62791

```
source setup.sh
model="hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/work_dirs/spatial/20250428_qwen2vl_sft_3d_0_5Btokens_fov_v3/checkpoints/global_step_430/hf_ckpt"
output_dir="hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/work_dirs/spatial/20250428_qwen2vl_sft_3d_0_5Btokens_fov_v3/checkpoints/global_step_430/evaluation"
dataset="sunrgbd/annotations/sunrgbd_val_total3d-prompt_v4_norm.json"
gt="sunrgbd/annotations/sunrgbd_total3d_val_metaqa_20250321.json"
data_dir="hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/spatial_data/benchmark"
image_dir="sunrgbd"
python3 src/inference/dist_inference_vllm_json.py -bs 64 --num_gpus 4 --model_name_or_path=$model --outdir=$output_dir --dataset $dataset --data_dir=$data_dir --image_dir=$image_dir --gt-file $gt --degree-range='1' --enable_fov
```

# 压缩解压命令
多进程解压多个 zip
```
find $1 -name "*.zip" | parallel -j 16 'unzip -o -d {//} {}'
```

多线程压缩
```
tar --use-compress-program="pigz -p 64" -cvf $tgt_file  $src_dir
```


## Post-train (VeRL)


### evaluation
1. merger

2. change the preprocessor_config 

"image_processor_type": "Qwen2VLImageProcessor",


# Evaluation on multi-modal datasets
Install the necessary dependencies:
```
source benchmark/setup.sh
```

You can runing this one to eval:
eval trial: https://ml.bytedance.net/development/instance/jobs/46245e112c3f2d95?tabState=run_info

```bash
source setup.sh
source tools/env.sh
source benchmark/setup.sh
cd benchmark
bash auto_run_qwen2_5vl.sh hdfs://haruna/home/byte_data_seed/hl_lq/iccv/intern/data/rayyang/work_dirs/spatial/20250427_qwen2_5vl_sft_llavaov800k_4e-5/checkpoints/global_step_716/hf_ckpt
```

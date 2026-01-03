
# Content
- [Prepare data](#-Prepare-data)
- [Train](#-Train)
- [Adapt to VLA Model](#-Adapt-to-VLA-Model)


# Prepare data

We prepare the data into the parquet format and calculate the total token nums (used for data packing and iterable dataloder).

Each item must follow this format strictly:
```python
{
    'conversations': [{'from': 'human', 'value': xxx}, {'from': 'gpt', 'value': xxx}, ...], # list
    'id': item_id, # string
    'data_source': item.get('data_source', data_source), # string
    'images': [{'bytes': b"xxx", 'path': xxx}, ...],  # list 
    'type': data_type, # string
    'meta_info': json.dumps(meta_info_list),  # string
}
```


Here, we give image and video examples.

## Image and Multi-image data

### Taking "lmms-lab/LLaVA-NeXT-Data" as an example.

* Step1: Download data:
```shell
# modify cache_dir and local_dir in this script and run it, the data will be saved into local_dir
# export HF_ENDPOINT='https://hf-mirror.com'  # if you don't have a VPN at mainland.
python tools/download_hf_data.py --repo_id 'lmms-lab/LLaVA-NeXT-Data' --cache_dir $your_cache_dir --local_dir $your_local_path 
```

* Step2: Convert the data into required parquet format
```shell
python prepare_data/sft/convert_llavanext_parquet.py --data_dir "your_data/LLaVA-NeXT-Data/data" -o "your_save_path" --tag "llava_next_vst"
```
Parquet files are under `your_save_path/llava_next_vst`

Then, to create a yaml file to record the data path (like [config/data/llavanext.yaml](config/data/llavanext.yaml)):
```yaml
- ann_path: llava_next_vst
  data_dir: your_save_path # revise to your data directory
```

* Step3: calculate the token num
```shell
python tools/compute_num_token.py config/data/llavanext.yaml -p your_model_dir/Qwen2.5-VL-3B-Instruct -w 8
```
The token num will be recorded in the yaml file:
```yaml
- ann_path: llava_next_vst
  data_dir: your_save_path
  token_num: 20531761
```


### Taking JSON-based data as an example.

You can convert the JSON-based data into required parquet files following this script:
```shell
python prepare_data/sft/convert_json_parquet.py -j llavaov_jsonfile -i yourdata/images -o "work_dirs/data" --tag "json_data" -w 8
```

NOTE:
each json item should follow the llava format:
```python
{
    'id': xxx,
    'conversations': xxx, # list
    'data_source': data_source, # string
    'images': images,  # list , "image" key is ok
}
```

After that, you need to calculate the token num follow the above step-3.

## Video

We provide the script to convert the llava video data into parquet format.

Each json item should follow the llava format:
```python
{
    'id': xxx,
    'conversations': xxx, # list
    'data_source': data_source, # string
    'video': video_path,  #  string
}
```

Convert the video data into the parquet:

```bash
jsonfile="LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_mc_v0_1_qa_processed.json"
python prepare_data/sft/convert_json_parquet_video.py \
  --json_file $jsonfile \
  --output_dir data/video/debug \
  --video_dir "" \
  --workers 16 \
  --batch_size 100 \
  --save_batch_size 10 \
  --tag "debug"
```

After that, you need to prepare a yaml file following step-2 and calculate the token num following step-3.


**NOTE**:
* We use the `<|video_pad|>` as the video special token. `<image>` special token in the json file will be replaced by `<|video_pad|>` token. Please check [here](https://github.com/Yangr116/VST/blob/b32988e85078e2ccac10f662100270fa8550b0d6/prepare_data/sft/convert_json_parquet_video.py#L230-L250) for details.

* In the training code, we limit the max frames into `max_frames = total_pixels * FRAME_FACTOR // int(min_pixels * 1.05)`, details can be found in line 63 of `vst/utils/vision_process.py`.


Now, you can use **the prepared data** and **data config** to train your model!

# Train

The meaning of the config can be found in [veomni](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/config/config.md).

```shell
export WANDB_API_KEY="your_wandb_key"
```
## Stage 1: SFT
```bash
bash scripts/train.sh vst/train.py config/veomni/qwen2_5_vl_fspd1_fov_packing_example.yaml \
    --model.model_path 'Qwen/Qwen2.5-VL-3B-Instruct' \
    --data.train_path 'config/data/llavanext.yaml' \
    --data.train_size 20_531_761 \
    --train.output_dir 'work_dirs/qwen2_5vl_sft_llavanext_example' \
    --train.wandb_name 'qwen2_5vl_sft_llavanext_example'
```
You can change `'Qwen/Qwen2.5-VL-3B-Instruct'` to your local path.

#### Merge model

If the model is saved as dcp original format, you need to merge them into huggingface format for evaluation or other purpose:
```shell
python tools/veomni_to_hf.py work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx
```
The huggingface model will be saved to `work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx/hf_ckpt_global_step_xxx`.


## Stage 2: CoT Cold Start

As the stage-1, the only thing is to use the data with CoT trace.

**NOTE**: 
* The `type` in the CoT sample should include `thought` tag because we set the thinking system prompt according to `type`.

```python
{
    'conversations': [{'from': 'human', 'value': xxx}, {'from': 'gpt', 'value': '<think>xxx</think> answer'}, ...], # list
    'id': item_id, # string
    'data_source': item.get('data_source', data_source), # string
    'images': [{'bytes': b"xxx", 'path': xxx}, ...],  # list 
    'type': 'thought_xxx', # string
    'meta_info': json.dumps(meta_info_list),  # string
}
```
* You can revise the custom thinking prompt in `vst/prompt.py`

#### Merge model

If the model is saved as dcp original format, you need to merge them into huggingface format for evaluation or other purpose:
```shell
python tools/veomni_to_hf.py work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx
```
The huggingface model will be saved to `work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx/hf_ckpt_global_step_xxx`.


## Stage 3: RL
TODO

# Adapt to VLA Model

## Prepare LIBERO data

### Step1: Download the LIBERO dataset
You should download the LIBERO dataset following [instructions](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#Dataset).

### Step2: Preprocess
We follow [OpenVLA](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py) to filter data:
```
python prepare_data/vla/libero/regenerate_libero_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
    --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops
# you should replace the path
```
Then, we got processed dataset:
```shell
├── libero_10_no_noops
├── libero_goal_no_noops
├── libero_object_no_noops
└── libero_spatial_no_noops
```

### Step3: Convert to parquet
```shell
python prepare_data/vla/libero/preprocess_libero.py \ 
    --save_dir "./dataset/parquet/vla/libero" \ 
    --libero_dir "./LIBERO/libero/datasets"
```

We have prepared the data config files at `config/data/vla/*.yaml`, you just need to revise the `data_dir`.
For example:
```yaml
# 28_378_046
- ann_path: libero_10_no_noops
  data_dir: dataset/parquet/vla/libero # revise this data_dir into your data dir.
  token_num: 10124377
```


## Train VLA model

To train the action model on the spatial subset:
```shell
bash scripts/train.sh vst/train_vla.py config/veomni/qwen2_5vla/vla_qwen2_5_vl_fspd1_new_token.yaml \
    --model.model_path '/mnt/bn/ic-vlm/rayyang/cache/cache_model/Qwen2.5-VL-3B-Instruct' \
    --data.train_path 'config/data/vla/libero_norm_spatial.yaml' \
    --data.train_size 5_800_000 \
    --train.output_dir 'work_dirs/20250824_vla_qwen2_5vl_3b_spatial_sft_libero_spatial' \
    --train.wandb_name '20250824_vla_qwen2_5vl_3b_spatial_sft_libero_spatial' \
    --data.num_workers 2 \
    --data.buffer_size 6000 \
    --train.lr_warmup_ratio 0.03 \
    --train.num_train_epochs 50 \
    --train.lr 0.00008 \
    --train.vit_lr 0.000008 \
    --data.max_seq_len 2048 \
    --train.micro_batch_size 16
```

#### Merge model

If the model is saved as dcp original format, you need to merge them into huggingface format for evaluation or other purpose:
```shell
python tools/veomni_to_hf.py work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx
```
The huggingface model will be saved to `work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx/hf_ckpt_global_step_xxx`.


## Evaluation on LIBERO

Prepare the LIBERO evaluation env:
```shell
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
pip install -r benchmark/libero/libero_requirements.txt
echo N | python -c "from libero.libero import benchmark"
sudo apt-get install libegl-dev -y
```

Then:

```shell
bash benchmark/libero/auto_run_vst_vla_libero_norm.sh \
    $your_model_path \
    "libero_spatial" \
    $your_work_dirs
```
`libero_spatial` can be "libero_spatial" "libero_object" "libero_goal" "libero_10"

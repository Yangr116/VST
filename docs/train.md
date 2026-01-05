# Table of Contents
- [Prepare Environment](#prepare-environment)
- [Prepare VST Data](#prepare-vst-data)
- [Prepare Custom Data](#prepare-custom-data)
- [Train](#train)
- [Adapt to VLA Model](#adapt-to-vla-model)

# Prepare Environment

```bash
git clone https://github.com/Yangr116/VST
cd VST

# Install VeOmni
git clone -b v0.1.3 https://github.com/ByteDance-Seed/VeOmni.git third_party/VeOmni
cd third_party/VeOmni
pip install -e .

# Install requirements
cd ../..
pip install -r requirements.txt

# Install flash-attn (recommended)
pip install flash-attn --no-build-isolation
```

> [!NOTE]
> We use `torch2.6.0+cu126`, but other PyTorch versions are also compatible.

# Prepare VST Data

### Step 1: Download Data
```bash
python tools/download_hf_data.py \
  --repo_id="rayruiyang/vst_500k" \
  --local_dir "$YOUR_LOCAL_DIR/vst_500k"
```

### Step 2: Download and Preprocess Video Data
For video data, please follow the [VLM-3R instructions](https://github.com/VITA-Group/VLM-3R/tree/main/vlm_3r_data_process) to prepare the video files for Scannet, Scannetpp, and ARKitScenes. The directory structure should look like this:

```text
processed_data/
├── arkitscenes
│   └── videos
│       └── train
├── scannet
│   └── videos
│       ├── train
│       └── val
└── scannetpp
    └── videos
```

Then, convert the video files into Parquet format:
```bash
jsonfile="$YOUR_LOCAL_DIR/vst_500k/video/video.json"

python prepare_data/sft/convert_json_parquet_video.py \
  --json_file "$jsonfile" \
  --output_dir "$YOUR_LOCAL_DIR/vst_500k/video" \
  --video_dir "$YOUR_LOCAL_DIR/processed_data" \
  --workers 16 \
  --batch_size 100 \
  --save_batch_size 10 \
  --tag "vst_video"
```

### Step 3: Prepare Data Config
You have two options:
1.  **Manual:** Replace `$YOUR_LOCAL_DIR` in the generated file `config/data/vst_500k.yaml` with your actual save directory.
2.  **Automatic:** Generate the data config using the script below:

```bash
python tools/generate_data_config.py "$YOUR_LOCAL_DIR/vst_500k" config/data/vst_500k.yaml
```

# Prepare Custom Data

> [!IMPORTANT]
> **Special Tokens:**
> *   We use `<|image_pad|>` and `<|video_pad|>` as special tokens.
> *   Rule: One `<|image_pad|>` = one image; One `<|video_pad|>` = one video.
> *   Although [the code](https://github.com/Yangr116/VST/blob/c93eae1fde3304cd2b8a02633dce2f542cec5bae/vst/preprocess.py#L68-L75) can automatically replace `<image>` tokens with `<|image_pad|>`, we strongly recommend using `<|image_pad|>` explicitly in your custom data.

We require data in **Parquet format** with pre-calculated token numbers (used for data packing and iterable dataloaders).

Each item must strictly follow this format:
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

Below are examples for Image and Video data.

## Image and Multi-image Data

### Example 1: "lmms-lab/LLaVA-NeXT-Data"

**Step 1: Download Data**
```bash
# Modify cache_dir and local_dir in the script before running.
# export HF_ENDPOINT='https://hf-mirror.com'  # Uncomment if you need a mirror in Mainland China.

python tools/download_hf_data.py \
  --repo_id 'lmms-lab/LLaVA-NeXT-Data' \
  --cache_dir "$$YOUR_CACHE_DIR" \
  --local_dir "$YOUR_LOCAL_DIR"
```

**Step 2: Convert to Parquet**
```bash
python prepare_data/sft/convert_llavanext_parquet.py \
  --data_dir "your_data/LLaVA-NeXT-Data/data" \
  -o "$YOUR_LOCAL_DIR" \
  --tag "llava_next_vst"
```
*Output files will be located at `$YOUR_LOCAL_DIR/llava_next_vst`.*

Next, create a YAML file to record the data path (e.g., `config/data/llavanext.yaml`):
```yaml
- ann_path: llava_next_vst
  data_dir: $YOUR_LOCAL_DIR # Update this to your data directory
```

**Step 3: Calculate Token Count**
```bash
python tools/compute_num_token.py config/data/llavanext.yaml \
  -p $YOUR_MODEL_PATH/Qwen2.5-VL-3B-Instruct \
  -w 8
```
The script will automatically update the YAML file with the token count:
```yaml
- ann_path: llava_next_vst
  data_dir: $YOUR_LOCAL_DIR
  token_num: 20531761
```

### Example 2: JSON-based Data

To convert generic JSON data into the required Parquet format:

```bash
python prepare_data/sft/convert_json_parquet.py \
  -j llavaov_jsonfile \
  -i yourdata/images \
  -o "$YOUR_LOCAL_DIR" \
  --tag "json_data" \
  -w 8
```

**Note:** Each JSON item must follow the LLaVA format:
```python
{
    'id': xxx,
    'conversations': xxx, # list
    'data_source': data_source, # string
    'images': images,  # list (key "image" is also acceptable)
}
```
*After conversion, calculate the token count as shown in Step 3 above.*

## Video Data

### Prepare Custom Video Data
We provide a script to convert LLaVA-style video data into Parquet format.

**Input JSON Format:**
```python
{
    'id': xxx,
    'conversations': xxx, # list
    'data_source': data_source, # string
    'video': video_path,  # string
}
```

**Conversion Command:**
```bash
jsonfile="LLaVA-Video-178K/0_30_s_academic_v0_1/0_30_s_academic_mc_v0_1_qa_processed.json"

python prepare_data/sft/convert_json_parquet_video.py \
  --json_file "$jsonfile" \
  --output_dir $YOUR_LOCAL_DIR \
  --video_dir "your_video_save_path" \
  --workers 16 \
  --batch_size 100 \
  --save_batch_size 10 \
  --tag "video_debug"
```

After conversion, prepare a YAML file (Step 2) and calculate the token count (Step 3).

> [!NOTE]
> *   **Special Tokens:** We use `<|video_pad|>` as the video special token. The `<image>` token in llava-video JSON files will be automatically replaced by `<|video_pad|>`. See [details here](https://github.com/Yangr116/VST/blob/b32988e85078e2ccac10f662100270fa8550b0d6/prepare_data/sft/convert_json_parquet_video.py#L230-L250). We recommend to directly use `<|video_pad|>` in your file.
> *   **Frame Limit:** In the training code, max frames are limited by: `max_frames = total_pixels * FRAME_FACTOR // int(min_pixels * 1.05)`. See `vst/utils/vision_process.py` (Line 63).

**You are now ready to train your model using the prepared data and config!**

# Train

For configuration details, refer to the [VeOmni documentation](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/config/config.md).

```bash
export WANDB_API_KEY="your_wandb_key"
```

## Stage 1: SFT (Supervised Fine-Tuning)

```bash
bash scripts/train.sh vst/train.py config/veomni/qwen2_5_vl_fspd1_fov_packing_example.yaml \
    --model.model_path 'Qwen/Qwen2.5-VL-3B-Instruct' \
    --data.train_path 'config/data/llavanext.yaml' \
    --data.train_size 20_531_761 \
    --train.output_dir 'work_dirs/qwen2_5vl_sft_llavanext_example' \
    --train.wandb_name 'qwen2_5vl_sft_llavanext_example'
```

> [!TIP]
> *   **Model Path:** You can change `'Qwen/Qwen2.5-VL-3B-Instruct'` to your local model path.
> *   **Train Size:** Ensure `data.train_size` matches the total number of tokens in your dataset.
> *   **Video Training:** Reduce `data.buffer_size` to `2000` if you are training on video data only to manage memory usage.

### Merge Model
If the model is saved in the original DCP format, merge it into Hugging Face format for evaluation:

```bash
python tools/veomni_to_hf.py work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx
```
*Output:* `work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx/hf_ckpt_global_step_xxx`

## Stage 2: CoT (Chain of Thought) Cold Start

This stage is similar to Stage 1, but uses data containing CoT traces.

> [!IMPORTANT]
> *   **Data Format:** The `type` field in the sample must include the `thought` tag. The system prompt for thinking is set based on this `type`.
> *   **Prompt Customization:** You can revise the custom thinking prompt in `vst/prompt.py`.

**Example Data Format:**
```python
{
    'conversations': [{'from': 'human', 'value': xxx}, {'from': 'gpt', 'value': '<think>xxx</think> answer'}, ...],
    'id': item_id,
    'data_source': item.get('data_source', data_source),
    'images': [{'bytes': b"xxx", 'path': xxx}, ...],
    'type': 'thought_xxx', # Must include 'thought'
    'meta_info': json.dumps(meta_info_list),
}
```

### Merge Model
Merge the DCP checkpoints to Hugging Face format:
```bash
python tools/veomni_to_hf.py work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx
```

## Stage 3: RL (Reinforcement Learning)
Please refer to [projects/spatial_rl/README.md](projects/spatial_rl/README.md).

# Adapt to VLA Model

## Prepare LIBERO Data

### Step 1: Download Dataset
Download the LIBERO dataset by following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#Dataset).

### Step 2: Preprocess
We follow [OpenVLA](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py) to filter the data.

```bash
# Replace paths with your actual directories
python prepare_data/vla/libero/regenerate_libero_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
    --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops
```

**Resulting Directory Structure:**
```text
├── libero_10_no_noops
├── libero_goal_no_noops
├── libero_object_no_noops
└── libero_spatial_no_noops
```

### Step 3: Convert to Parquet
```bash
python prepare_data/vla/libero/preprocess_libero.py \
    --save_dir "./dataset/parquet/vla/libero" \
    --libero_dir "./LIBERO/libero/datasets"
```

We have prepared config files at `config/data/vla/*.yaml`. You simply need to update the `data_dir`.

**Example (`config/data/vla/libero_spatial.yaml`):**
```yaml
# 28_378_046
- ann_path: libero_10_no_noops
  data_dir: dataset/parquet/vla/libero # Update this to your data directory
  token_num: 10124377
```

## Train VLA Model

Two key parameters are added to the config for VLA training:

| Key | Description |
| :--- | :--- |
| `enable_vla` | Set to `true` to use the VLA transform. |
| `add_tokens` | Set to `['action_token']` to add new action tokens. |

**Training Command (Spatial Subset Example):**
```bash
bash scripts/train.sh vst/train_vla.py config/veomni/qwen2_5vla/vla_qwen2_5_vl_fspd1_new_token.yaml \
    --model.model_path 'rayruiyang/VST-3B-SFT' \
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

### Merge Model
Merge the DCP checkpoints to Hugging Face format:
```bash
python tools/veomni_to_hf.py work_dirs/qwen2_5vl_sft_llavanext_example/checkpoints/global_steps_xxx
```

## Evaluation on LIBERO

**1. Prepare Evaluation Environment:**
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
pip install -r benchmark/libero/libero_requirements.txt
echo N | python -c "from libero.libero import benchmark"
sudo apt-get install libegl-dev -y
```

**2. Run Evaluation:**
```bash
bash benchmark/libero/auto_run_vst_vla_libero_norm.sh \
    "$your_model_path" \
    "libero_spatial" \
    "$your_work_dirs"
```

> [!NOTE]
> The task argument (`libero_spatial` above) can be replaced with:
> *   `libero_spatial`
> *   `libero_object`
> *   `libero_goal`
> *   `libero_10`
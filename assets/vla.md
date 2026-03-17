# Adapt to VLA Model

## Prepare Environment

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
| `enable_augmentation` | Set to `true` to enable data augmentation. |

**Training Command (Spatial Subset Example):**
```bash
bash scripts/train.sh vst/train_vla.py config/veomni/qwen2_5vla/vla_qwen2_5_vl_fspd1_new_token_aug.yaml \
    --model.model_path 'rayruiyang/VST-3B-SFT' \
    --data.train_path 'config/data/vla/libero_norm_spatial.yaml' \
    --data.train_size 5_800_000 \
    --train.output_dir 'work_dirs/20250824_vla_qwen2_5vl_3b_spatial_sft_libero_spatial' \
    --train.wandb_name '20250824_vla_qwen2_5vl_3b_spatial_sft_libero_spatial' \
    --data.num_workers 2 \
    --data.buffer_size 6000 \
    --train.lr_warmup_ratio 0.1 \
    --train.num_train_epochs 200 \
    --train.lr 0.00005 \
    --train.vit_lr 0.000005 \
    --data.max_seq_len 1024 \
    --train.micro_batch_size 16 \
    --train.global_batch_size 128
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
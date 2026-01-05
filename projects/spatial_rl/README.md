# Table of Contents
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)

# Environment Setup

```bash
git clone https://github.com/Yangr116/VST
cd VST
cd projects/spatial_rl

# NOTE: We use torch 2.6.0+cu126. Other torch versions may also work but are untested.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_rl.txt
```

**Optional: 3D Object Detection (RLVR)**
If you intend to train 3D object detection tasks using RLVR, you must install `pytorch3d`.

> **Note:** The wheel below is specific to **Python 3.10** and **Linux**. For other configurations, please refer to the [PyTorch3D discussions](https://github.com/facebookresearch/pytorch3d/discussions/1752).

```bash
wget https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9%2Bpt2.6.0cu126-cp310-cp310-linux_x86_64.whl
pip install pytorch3d-0.7.9+pt2.6.0cu126-cp310-cp310-linux_x86_64.whl
```

# Data Preparation

## Prepare Custom Data
Your source data should be a JSON file containing a list of samples. Each sample must adhere to the following format:

```json
{
  "id": "638bafb8-2ac9-4912-b9fd-e49906c1e743",
  "images": ["openimages/0029d8bae76d9a79.jpg"],
  "question": "Consider the real-world 3D locations and orientations of the objects. Which side of the bird is facing the camera?\nOptions:\nA: front\nB: back\nC: left\nD: right\nPlease select the correct answer from the options above.",
  "thought": "",
  "answer_gt": "A",
  "answer": "A",
  "data_source": "spatial_reasoner_1k.json",
  "meta_info": "{\"mode\":[\"multi_choices_score\"],\"question_type\":\"multi_choices\"}"
}
```

### Configuration Details

1.  **Reward Modes (`mode`):**
    The reward mode in `meta_info` must be one of the following:

    | Mode | Use Case |
    | :--- | :--- |
    | `vqa_score` | VQA samples (e.g., GQA, VQAv2) |
    | `relaxed_score` | ChartQA and SROIE |
    | `anls_score` | DocVQA and Infographics |
    | `math_score` | Geometry3K and Mavis |
    | `multi_choices_score` | Multiple-choice QA data |

2.  **Meta Info Structure:**
    Ensure `meta_info` is a JSON string:
    `meta_info=json.dumps({'mode': ['multi_choices_score', ...], 'question_type': 'multi_choices'})`

3.  **Image Tokens:**
    If the `question` field does not include the image placeholder `<|image_pad|>`, it will be automatically added during the Parquet conversion.
    *   **Rule:** One `<|image_pad|>` corresponds to one image. If using two images, you must include two `<|image_pad|>` tokens.

### Convert to Parquet
Run the following command to convert your prepared JSON file into the Parquet format required for training:

```bash
cd ../..
python prepare_data/rl/convert_rl_parquet.py \
    -j dataset/rl_example.json \
    -i dataset/images \
    --parquet dataset/parquet/rl \
    -t rl_example
```

## Download Example Data
Alternatively, you can download example data from Hugging Face:

```bash
python ../../tools/download_hf_data.py \
    --repo_id 'xxx' \
    --cache_dir $YOUR_CACHE_DIR \
    --local_dir $YOUR_LOCAL_DIR
```

# Training

First, export your Weights & Biases API key:

```bash
export WANDB_API_KEY="your_wandb_key"
```

### Launch Training
Run the training script using the command below. Ensure you update `train_files` to point to your data directory.

```bash
nnodes=1
n_gpus_per_node=4
exp_name="0913_qwen2_5_vl_7b_grpo_dapo_mix"
save_checkpoint_path="../../work_dirs/vst_rl/$exp_name"
mkdir -p $save_checkpoint_path

project_name="vst_rl"
model_path="rayruiyang/VST-7B-SFT"
train_files="../../dataset/parquet/rl/spatial_reasoner_aug" # TODO: Update this path to your data directory

python3 -m verl.trainer.main \
    config=examples/config_vlm3d_reasoner.yaml \
    data.train_files=$train_files \
    data.val_files=$train_files \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=$model_path \
    worker.actor.ulysses_size=1 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.model.freeze_vision_tower=false \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=$exp_name \
    trainer.project_name=$project_name \
    trainer.save_checkpoint_path=$save_checkpoint_path \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True
```

### Implementation Notes
*   **Dataset Class:** We added `RLHFDatasetVLM3D` in `verl/utils/dataset.py` to support this implementation.
*   **Reward Function:** The custom reward function is located in `examples/reward_function/vlm3d.py`.
*   **Advanced Settings:** For more custom configurations, please refer to [EasyR1](https://github.com/hiyouga/EasyR1).

### Merge Weights
After training, merge the weights into the Hugging Face format:

```bash
python scripts/model_merger.py --local_dir "$save_checkpoint_path/global_step_xxx/actor"
```
The converted weights will be saved to: `$save_checkpoint_path/global_step_xxx/actor/huggingface`


# Content
- [Prepare Env](#-Prepare-ENV)
- [Prepare data](#-Prepare-data)
- [Train](#-Train)

# Prepare ENV

```shell
git clone https://github.com/Yangr116/VST
cd VST
cd projects/spatial_rl
# NOTE: we use torch2.6.0+cu126, other torch version is also fine.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_rl.txt
```

If you want to train the 3D object detection task using RLVR, you need to install the pytorch3d:
```
wget https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9%2Bpt2.6.0cu126-cp310-cp310-linux_x86_64.whl
pip install pytorch3d-0.7.9+pt2.6.0cu126-cp310-cp310-linux_x86_64.whl
```
NOTE: more details and other pytorch3d versions can be found [https://github.com/facebookresearch/pytorch3d/discussions/1752](https://github.com/facebookresearch/pytorch3d/discussions/1752)


# Prepare data

## Prepare your own data
The sample saved in json file should have the following format:
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
  },
```
* The reward mode must be one of: 'vqa_score', 'relaxed_score', 'anls_score', 'math_score', 'multi_choices_score'
  * `vqa_score`: used for vqa samples from GQA/VQAv2
  * `relaxed_score`: used for chartqa and SROIE
  * `anls_score`: used for docvqa and infographics
  * `math_score`: used for geometry3k and mavis
  * `multi_choices_score`: used for multi-choices qa data

* The `meta_info=json.dumps({'mode': ['multi_choices_score', xxx], 'question_type': 'multi_choices'})`
* If the question doesn't inclue the image_flag `<|image_pad|>`, it will be added during converting data into parquet format. One `<|image_pad|>` equals to one image. Two image should place two `<|image_pad|>`.

Convert the prepared json file into the parquet:
```
cd ../..
python prepare_data/rl/convert_rl_parquet.py \
    -j dataset/rl_example.json \
    -i dataset/images \
    --parquet dataset/parquet/rl \
    -t rl_example
```

## Download example data
```shell
python ../../tools/download_hf_data.py --repo_id 'xxx' --cache_dir $your_cache_dir --local_dir $your_local_path 
```

# Train

Set your wandb key:
```shell
export WANDB_API_KEY="your_wandb_key"
```

Start to train:
```shell
nnodes=1
n_gpus_per_node=4
exp_name="0913_qwen2_5_vl_7b_grpo_dapo_mix"
save_checkpoint_path="../../work_dirs/vst_rl/$exp_name"
mkdir -p $save_checkpoint_path
project_name="vst_rl"
model_path="rayruiyang/VST-7B-SFT"
train_files="../../dataset/parquet/rl/spatial_reasoner_aug" # revise to your data dir

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

**NOTE:**

* We add the new dataset class `RLHFDatasetVLM3D` in `verl/utils/dataset.py` to support our implementation.
* Our reward function is in `examples/reward_function/vlm3d.py`.
* More custom settings please follow [EasyR1](https://github.com/hiyouga/EasyR1)


After training, you need to merge the weights into huggingface format:
```shell
python scripts/model_merger.py --local_dir "$save_checkpoint_path/global_step_xxx/actor"
```
The converted weights will be saved into `$save_checkpoint_path/global_step_xxx/actor/huggingface`

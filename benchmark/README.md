# Benchmark Evaluation

This document describes how to evaluate VST on multi-modal understanding benchmarks and 3D object detection benchmarks.

## Table of Contents

- [Multi-Modal Understanding Benchmarks](#multi-modal-understanding-benchmarks)
- [3D Object Detection Benchmarks](#3d-object-detection-benchmarks)

---

## Multi-Modal Understanding Benchmarks

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate on multi-modal benchmarks.

### Environment Setup

```bash
git clone https://github.com/open-compass/VLMEvalKit.git benchmark/VLMEvalKit
cd benchmark/VLMEvalKit
pip install -e .
cd ../..
```

### Evaluation

**Config-based evaluation** (MMStar, 3DSRBench, BLINK, RealWorldQA, MMBench_DEV_EN, MMMU_DEV_VAL, OCRBench, AI2D_TEST, CV-Bench, MMSI_Bench):

```bash
model_path="your_model_path"
bash benchmark/scripts/auto_run_qwen2_5vl.sh $model_path
```

**Chain-of-Thought (CoT):**

```bash
model_path="your_model_path"
bash benchmark/scripts/auto_run_qwen2_5vl_cot.sh $model_path
```

**VSIBench:**

```bash
# Usage: bash benchmark/scripts/eval_vsibench.sh <model_path> <output_dir> [fps]
# fps defaults to 4 if not specified
model_path="your_model_path"
output_dir="work_dirs/vsi"
bash benchmark/scripts/eval_vsibench.sh $model_path $output_dir 4
```

---

## 3D Object Detection Benchmarks

Evaluation uses **vLLM** for inference. Supported benchmarks: **SUN RGB-D** and **ARKit**.

### 1. Environment Setup

Install PyTorch3D:

> **Note:** The wheel below is for **Python 3.10**, **PyTorch 2.6**, **CUDA 12.6**, and **Linux**. For other setups, see [PyTorch3D discussions](https://github.com/facebookresearch/pytorch3d/discussions/1752).

```bash
wget https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9%2Bpt2.6.0cu126-cp310-cp310-linux_x86_64.whl
pip install pytorch3d-0.7.9+pt2.6.0cu126-cp310-cp310-linux_x86_64.whl
```

### 2. Prepare the Benchmark Data

[![Benchmark](https://img.shields.io/badge/Benchmark-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/rayruiyang/vst_3d_grounding_benchmark)


Download and extract the 3D grounding benchmark:

```bash
python tools/download_hf_data.py --repo_id="rayruiyang/vst_3d_grounding_benchmark" --local_dir $YOUR_LOCAL_PATH
cd $YOUR_LOCAL_PATH/vst_3d_grounding_benchmark
tar -zxvf arkit/arkit_omni3d_test_640x480.tar.gz -C arkit
tar -zxvf sunrgbd/sunrgbd_val.tar.gz -C sunrgbd
```
You will get:
```shell
$YOUR_LOCAL_PATH/vst_3d_grounding_benchmark
├── arkit
│   ├── annotations
│   └── arkit_wo_rotation
│       └── arkit_omni3d_test_640x480
└── sunrgbd
    ├── annotations
    └── sunrgbd
        └── val
```


### 3. Run Evaluation

**Option A: Use the helper script (SUN RGB-D)**

From the repo root, with `$YOUR_LOCAL_PATH` pointing to the folder that contains `vst_3d_grounding_benchmark` (e.g. `data/`):

```bash
# Edit benchmark/scripts/auto_run_3dod.sh to set model, data_dir, etc., then:
bash benchmark/scripts/auto_run_3dod.sh
```

**Option B: Run inference manually**

Set shared variables (paths relative to your `data_dir`):

**SUN RGB-D:**

```bash
model="your_model_path"
outdir="work_dirs"
data_dir="data/vst_3d_grounding_benchmark"   
dataset="sunrgbd/annotations/sunrgbd_val_total3d-prompt_v4_norm.json"
gt="sunrgbd/annotations/sunrgbd_total3d_val_metaqa_20250321.json"
image_dir="sunrgbd"

python benchmark/det3d/dist_inference_vllm_json.py \
  -bs 64 --num_gpus 4 \
  --model_name_or_path $model \
  --outdir $outdir \
  --dataset $dataset \
  --data_dir $data_dir \
  --image_dir $image_dir \
  --gt-file $gt \
  --degree-range 1 \
  --enable_fov
```

**ARKit:**

```bash
model="your_model_path"
outdir="work_dirs"
data_dir="data/vst_3d_grounding_benchmark"   
dataset="arkit/annotations/arkit_omni3d_test_640x480_prompt_v4_norm.json"
gt="arkit/annotations/arkit_omni3d_test_640x480_metaqa_20250317.json"
image_dir="arkit"

python benchmark/det3d/dist_inference_vllm_json.py \
  -bs 64 --num_gpus 4 \
  --model_name_or_path $model \
  --outdir $outdir \
  --dataset $dataset \
  --data_dir $data_dir \
  --image_dir $image_dir \
  --gt-file $gt \
  --degree-range 1 \
  --enable_fov
```

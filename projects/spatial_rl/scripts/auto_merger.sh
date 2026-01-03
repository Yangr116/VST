#!/bin/bash

# 定义根目录
BASE_DIR=$1

# 确保目标目录存在
mkdir -p "$BASE_DIR/hf_weights"

echo "开始查找并处理目录..."

# 遍历所有匹配的目录
for dir in "$BASE_DIR"/*global_step*/; do
  # 检查目录是否真实存在
  if [ -d "$dir" ]; then
    # 去掉结尾的斜杠
    clean_dir="${dir%/}"
    
    # 使用 basename 提取目录名，例如 "global_step_1000"
    global_step_name=$(basename "$clean_dir")

    save_dir="$BASE_DIR/hf_weights/$global_step_name"
    
    if [ -d "$save_dir" ]; then
      echo "目标目录已存在，跳过处理: $save_dir"
      echo "--------------------------------"
      continue # 跳过本次循环的剩余部分
    fi

    echo "正在处理目录: $clean_dir"
    echo "提取到的步骤名是: $global_step_name"
    
    # 1. 执行你的 Python 脚本
    python scripts/model_merger.py --local_dir "$clean_dir/actor"
    
    # 检查上一步是否成功
    if [ $? -ne 0 ]; then
        echo "模型合并失败: $clean_dir，跳过复制步骤。"
        echo "--------------------------------"
        continue # 跳过本次循环的剩余部分
    fi

    # 2. 复制处理好的权重文件
    echo "正在复制权重到: $BASE_DIR/hf_weights/$global_step_name"
    cp -r "$clean_dir/actor/huggingface" "$BASE_DIR/hf_weights/$global_step_name"

    # 检查复制是否成功
    if [ $? -eq 0 ]; then
      echo "处理和复制成功: $clean_dir"
    else
      echo "复制失败: $clean_dir，请检查错误信息。"
    fi
    
    echo "--------------------------------"
  fi
done

echo "所有目录处理完毕！"

# Copyright 2025 [Visual Spatial Tuning] Authors

import os
from typing import Callable, Dict, List, Literal, Optional, Union

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, IterableDataset
from loguru import logger
import random
import torch.distributed as dist

from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.dist_utils import main_process_first

from vst.prompt import DUMMY_CONV


class DummyDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]


class MappingDataset(Dataset):
    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if self._transform is not None:
            return self._transform(self._data[index])
        else:
            return self._data[index]


class IterativeDataset(IterableDataset):
    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        while True: # continuting to read data
            for sample in self._data:
                if self._transform is not None:                    
                    try:
                        transformed_samples = self._transform(sample)
                        yield transformed_samples
                    except Exception as e:
                        logger.error(e)
                        continue
                else:
                    yield sample

    def load_state_dict(self, state_dict):
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self._data.state_dict()}

    def set_epoch(self, epoch: int):
        self._data.set_epoch(epoch)


def build_dummy_dataset(size: int, max_seq_len: int) -> "Dataset":
    return DummyDataset(size=size, seq_length=max_seq_len)


def build_mapping_dataset(
    data_path: Union[str, List],
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
) -> "Dataset":
    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if data_path.startswith("hdfs://"):
            raise ValueError
        elif os.path.isdir(data_path):
            data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
        elif os.path.isfile(data_path):
            data_files.append(data_path)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")

    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    with main_process_first():
        dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace)

    return MappingDataset(data=dataset, transform=transform)


def build_iterative_dataset(
    data_path: Union[str, List],
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
    buffer_size: int = 10000,
) -> "IterableDataset":
    data_files = []
    if isinstance(data_path, str):
        data_paths = data_path.split(",")
        for data_path in data_paths:
            if data_path.startswith("hdfs://"):
                raise ValueError(f"Dataset {data_path} is not supported.")
            elif os.path.isdir(data_path):
                data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
            elif os.path.isfile(data_path):
                data_files.append(data_path)
            else:
                raise FileNotFoundError(f"Dataset {data_path} not exists.")
    elif isinstance(data_path, list):
        data_files = data_path
    else:
        raise NotImplementedError
    parallel_state = get_parallel_state()
    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")
    # data files must divided evenly by world_size
    to_add = parallel_state.dp_size - len(data_files) % parallel_state.dp_size
    data_files = data_files + random.sample(data_files, k=to_add)
    random.shuffle(data_files)
    # we shuffle the data files and broadcast
    dist.barrier()
    dist.broadcast_object_list(data_files, src=0)
    logger.info(f"{dist.get_rank()} parquet list: {data_files}")
    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)  # reduce the buffer size because of the video input
    dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)
    logger.info(f"#### dataset has been initialized.")
    # refer to: https://huggingface.co/docs/datasets/v1.11.0/dataset_streaming.html#shuffle-the-dataset
    return IterativeDataset(dataset, transform)


if __name__=="__main__":
    from glob import glob
    # data_dir = "/opt/tiger/rayyang/spatial/data/inhouse/parquet/2D/llavaov_sft_810191"
    data_dir = "/opt/tiger/rayyang/spatial/data/inhouse/parquet/3D_jsonqa/all_json_qa_arkit_train_wo_rotation_20250205-prompt_v4_norm"
    data_files = glob(data_dir + "/*.parquet")
    dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    for sample in dataset:
        print(sample)
        break

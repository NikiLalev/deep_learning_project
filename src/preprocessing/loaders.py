from __future__ import annotations

from typing import Optional, Tuple

from huggingface_hub import login
from datasets import (
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_info,
    load_dataset,
    load_from_disk,
)
from pathlib import Path
from datasets import Image as HFImage
from .transforms import to_torch, to_torch_imagenet

from .preprocess import preprocess_imagenet, preprocess_yolo

DEFAULT_PASCAL_PATH = "data/pascal_voc_yolo_448"
    
def load_pascal_iterable(path=DEFAULT_PASCAL_PATH):
    path = Path(path)
    if not path.exists():
        load_pascal(save_path=str(path))

    ds = load_from_disk(str(path))
    ds = ds.with_format(None)
    ds = dict(ds)

    # Convert each split to HF IterableDataset
    ds["train"] = ds["train"].to_iterable_dataset().map(to_torch)
    ds["test"]  = ds["test"].to_iterable_dataset().map(to_torch)
    return ds

def load_pascal(save_path: str = DEFAULT_PASCAL_PATH) -> DatasetDict:
    """
    Build PASCAL VOC train/test with YOLO-style targets and save to disk.
    """
    voc07 = load_dataset("HuggingFaceM4/pascal_voc", "voc2007_main", trust_remote_code=True, streaming=True)

    voc12_train = load_dataset(
        "HuggingFaceM4/pascal_voc",
        "voc2012_main",
        split="train",
        trust_remote_code=True,
    )
    voc12_val = load_dataset(
        "HuggingFaceM4/pascal_voc",
        "voc2012_main",
        split="validation",
        trust_remote_code=True,
    )

    train_ds = concatenate_datasets(
        [
            voc07["train"],
            voc07["validation"],
            voc12_train,
            voc12_val,
        ]
    )

    test_ds = voc07["test"]

    train_ds = train_ds.map(
        preprocess_yolo,
        fn_kwargs={"img_size": 448},
        remove_columns=train_ds.column_names,
    )
    test_ds = test_ds.map(
        preprocess_yolo,
        fn_kwargs={"img_size": 448},
        remove_columns=test_ds.column_names,
    )

    train_ds = train_ds.cast_column("image", HFImage())
    test_ds = test_ds.cast_column("image", HFImage())

    ds = DatasetDict({"train": train_ds, "test": test_ds})
    ds.save_to_disk(save_path)
    return ds


def load_pascal_from_disk(path: str = DEFAULT_PASCAL_PATH) -> DatasetDict:
    return load_from_disk(path)


def load_imagenet_iterable():
    train, val, test = load_imagenet()
    train = train.map(to_torch_imagenet)
    val = val.map(to_torch_imagenet)
    test = test.map(to_torch_imagenet) if test is not None else None

    ds_imagenet = {
        "train": train,
        "validation": val,
        "test": test,
    }
    return ds_imagenet


def load_imagenet(
    train_n: int = 1000,
    val_n: int = 200,
    test_n: int = 200,
    img_size: int = 224,
):
    """
    Load ImageNet-1k as streaming iterables, map preprocessing, and take first N.
    """
    login()
    dsd = load_dataset("ILSVRC/imagenet-1k", streaming=True)

    train = dsd["train"].map(preprocess_imagenet, fn_kwargs={"img_size": img_size}).take(train_n)
    val = dsd["validation"].map(preprocess_imagenet, fn_kwargs={"img_size": img_size}).take(val_n)
    test = (
        dsd["test"].map(preprocess_imagenet, fn_kwargs={"img_size": img_size}).take(test_n)
        if "test" in dsd
        else None
    )

    # return splits (iterables)
    return train, val, test


def load_imagenet_label_names():
    info = get_dataset_config_info("ILSVRC/imagenet-1k")
    return info.features["label"].names

from __future__ import annotations

from datasets import load_from_disk

from .loaders import DEFAULT_PASCAL_PATH, load_imagenet_iterable, load_pascal_iterable
from .plotting import plot_example, plot_imagenet_example


def import_data():
    # Load imagenet dataset
    ds_imagenet = load_imagenet_iterable()

    # Load pascal voc dataset
    ds_pascal = load_pascal_iterable()

    print(type(ds_imagenet))
    print(type(ds_imagenet["train"]))
    print(type(ds_pascal))
    print(type(ds_pascal["train"]))

    # Optional: plot examples
    # plot_example(ds_pascal, example_num=0)
    # plot_imagenet_example(ds_imagenet, example_num=0)

    print("Data imported successfully.")


    print("Sample Imagenet example:")
    ex = next(iter(ds_imagenet["train"]))
    print(ex["image"].shape)
    print("Label:", ex.get("label", None))
    
    print("Sample PASCAL example:")
    ex_pascal = next(iter(ds_pascal["train"]))
    print("Image shape:", ex_pascal["image"].shape)
    print("Boxes:", ex_pascal["boxes"])
    print("Labels:", ex_pascal["labels"])


    return {
        "imagenet": ds_imagenet,
        "pascal": ds_pascal,
    }



if __name__ == "__main__":
    import_data()

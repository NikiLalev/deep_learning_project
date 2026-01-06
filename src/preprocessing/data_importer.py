from huggingface_hub import login
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
from datasets import Image as HFImage
import torchvision.transforms.functional as F
import torch

def resize_images(example, target_size):
    img = example["image"]
    orig_w, orig_h = img.size

    # Resize image
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)

    # Scale bounding boxes
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h
    boxes = []
    for box in example["objects"]["bboxes"]:
        xmin, ymin, xmax, ymax = box
        boxes.append([
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y,
        ])

    example["image"] = img_resized
    example["objects"]["bboxes"] = boxes
    example["width"] = target_size
    example["height"] = target_size

    return example

def preprocess_yolo(example, img_size=448):
    img = example["image"]

    orig_w = example.get("width", img.size[0])
    orig_h = example.get("height", img.size[1])

    # resize once
    img = img.resize((img_size, img_size), Image.BILINEAR)

    boxes = []
    labels = []

    for box, label in zip(example["objects"]["bboxes"], example["objects"]["classes"]):
        xmin, ymin, xmax, ymax = box

        # scale to resized image
        xmin = xmin * img_size / orig_w
        xmax = xmax * img_size / orig_w
        ymin = ymin * img_size / orig_h
        ymax = ymax * img_size / orig_h

        # YOLO normalized
        xc = ((xmin + xmax) / 2) / img_size
        yc = ((ymin + ymax) / 2) / img_size
        w  = (xmax - xmin) / img_size
        h  = (ymax - ymin) / img_size

        boxes.append([xc, yc, w, h])
        labels.append(int(label))

    return {"image": img, "boxes": boxes, "labels": labels}

def load_pascal():

    voc07 = load_dataset("HuggingFaceM4/pascal_voc", "voc2007_main", trust_remote_code=True)

    voc12_train = load_dataset("HuggingFaceM4/pascal_voc", "voc2012_main",
                               split="train", trust_remote_code=True)
    voc12_val   = load_dataset("HuggingFaceM4/pascal_voc", "voc2012_main",
                               split="validation", trust_remote_code=True)

    train_ds = concatenate_datasets([
        voc07["train"],
        voc07["validation"],
        voc12_train,
        voc12_val,
    ])

    test_ds = voc07["test"]

    train_ds = train_ds.map(preprocess_yolo, fn_kwargs={"img_size": 448},
                            remove_columns=train_ds.column_names)
    test_ds  = test_ds.map(preprocess_yolo, fn_kwargs={"img_size": 448},
                            remove_columns=test_ds.column_names)

    train_ds = train_ds.cast_column("image", HFImage())
    test_ds  = test_ds.cast_column("image", HFImage())

    ds = DatasetDict({"train": train_ds, "test": test_ds})
    ds.save_to_disk("data/pascal_voc_yolo_448")

    return ds 


def save_sample_images(data, num_images=5):

    for i, example in enumerate(data):

        # Access the image and label
        image = example['image']  # This is a PIL Image object
        label = example['label']

        features = data['train'].features
        label = features['label'].int2str(label)

        clean_label = label.replace(" ", "_").replace(",", "")

        print(f"Label: {label}")

        # 4. Display it
        plt.figure()
        plt.imshow(image)
        plt.title(f"ImageNet Label: {label}")
        plt.axis('off')
        plt.savefig(f"data/images/imagenet_{i}_{clean_label}.png")
        plt.close()

        if i > num_images:
            break

def load_imagenet():
    # Login using e.g. `huggingface-cli login` to access this dataset
    login()
    dataset_dict = load_dataset("ILSVRC/imagenet-1k", streaming=True)

    # TODO: Adjust the number of samples as needed
    train_data = dataset_dict['train'].take(1000)
    val_data   = dataset_dict['validation'].take(200)
    test_data  = dataset_dict['test'].take(200)

    #TODO: scale this to 224x224
    
    return train_data, val_data, test_data


def to_torch(example):
    return {
        "image":  F.to_tensor(example["image"]),
        "boxes":  torch.tensor(example["boxes"], dtype=torch.float32),
        "labels": torch.tensor(example["labels"], dtype=torch.long),
    }

def plot_example(ds, example_num):
    VOC_CLASSES = [
        "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
        "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
    ]
        
    # ---- Pick one example ----
    example = ds["train"][example_num]

    image  = example["image"]   # (3, 448, 448)
    boxes  = example["boxes"]   # (N, 4) in YOLO format
    labels = example["labels"]

    # ---- Convert image for matplotlib ----
    img_np = image.permute(1, 2, 0).numpy()  # (448, 448, 3)

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img_np)
    ax.set_axis_off()

    H, W = img_np.shape[:2]

    # ---- Draw bounding boxes ----
    for box, label in zip(boxes, labels):
        xc, yc, w, h = box.tolist()

        # Convert YOLO → pixel corner format
        x_min = (xc - w / 2) * W
        y_min = (yc - h / 2) * H
        box_w = w * W
        box_h = h * H

        rect = patches.Rectangle(
            (x_min, y_min),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        # Class label
        class_name = VOC_CLASSES[label]
        ax.text(
            x_min,
            y_min - 3,
            class_name,
            color="red",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, pad=1),
        )

    plt.show()

def import_data():
    
    # Load imagenet dataset

    # load_imagenet()


    # Load pascal voc dataset

    ds = load_pascal()
    print(type(ds["train"][0]["image"]))  
    ds = load_from_disk("data/pascal_voc_yolo_448")
    print(type(ds["train"][0]["image"]))  
    ds = load_from_disk("data/pascal_voc_yolo_448").with_transform(to_torch)
    plot_example(ds, example_num=10)

    return ds

    
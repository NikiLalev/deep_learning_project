from datasets import load_dataset
from huggingface_hub import login
import matplotlib.pyplot as plt

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

def preprocess_data():
    # Login using e.g. `huggingface-cli login` to access this dataset
    login()
    dataset_dict = load_dataset("ILSVRC/imagenet-1k", streaming=True)

    # TODO: Adjust the number of samples as needed
    train_data = dataset_dict['train'].take(1000)
    val_data   = dataset_dict['validation'].take(200)
    test_data  = dataset_dict['test'].take(200)
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    preprocess_data()
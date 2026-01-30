"""
Data loaders for ImageNet dataset
Uses Local Streaming (Reads local cache files)
"""

import torch
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import os
import io
import numpy as np

def get_imagenet_transforms(split='train'):
    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def preprocess_imagenet(example, transform):
    image = example['image']
    # If image is a dict, extract the bytes and convert to PIL Image
    if isinstance(image, dict):
        if 'bytes' in image:
            image = Image.open(io.BytesIO(image['bytes'])).convert('RGB')
        elif 'path' in image:
            image = Image.open(image['path']).convert('RGB')
        else:
            raise ValueError("Unknown image dict format in example['image']")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    example['image'] = transform(image)
    return example

def load_imagenet_iterable(streaming=True, train_n=None, val_n=None, hf_token=None):
    """
    Load ImageNet dataset via Streaming.
    Since we ran 'huggingface-cli download', this will stream from the LOCAL CACHE.
    """
    print(f"Loading ImageNet (Streaming Mode)...")
    
    train_transform = get_imagenet_transforms('train')
    val_transform = get_imagenet_transforms('validation')
    
    try:
        # Load in streaming mode (Reads from local cache if available)
        # We use the NEW ID: ILSVRC/imagenet-1k
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            token=hf_token,
            streaming=True,
            trust_remote_code=False
        )
        
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        
        print("✓ Successfully loaded ImageNet stream")
        
    except Exception as e:
        print(f"❌ Error loading ImageNet: {e}")
        raise
    
    # Map transforms
    train_dataset = train_dataset.map(lambda x: preprocess_imagenet(x, train_transform))
    val_dataset = val_dataset.map(lambda x: preprocess_imagenet(x, val_transform))
    
    # Shuffle buffer (Required for streaming training)
    train_dataset = train_dataset.shuffle(buffer_size=10000, seed=42)
    
    if train_n is not None:
        train_dataset = train_dataset.take(train_n)
    if val_n is not None:
        val_dataset = val_dataset.take(val_n)
    
    return {
        "train": train_dataset,
        "validation": val_dataset
    }

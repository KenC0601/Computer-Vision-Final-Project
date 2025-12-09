import os
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np

class MetaAlbumDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, organized by class folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._make_dataset()

    def _make_dataset(self):
        images = []
        for target_class in self.classes:
            class_dir = self.root_dir / target_class
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    item = (str(img_path), self.class_to_idx[target_class])
                    images.append(item)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, target = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target

def create_few_shot_splits(dataset, num_shots=5, num_val=5, seed=42):
    """
    Creates 5-shot train, 5-shot val, and rest-test splits.
    
    Returns:
        train_indices, val_indices, test_indices
    """
    random.seed(seed)
    np.random.seed(seed)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Group indices by class
    class_indices = {}
    for idx, (_, target) in enumerate(dataset.images):
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(idx)
        
    for label, indices in class_indices.items():
        # Shuffle indices for this class
        random.shuffle(indices)
        
        if len(indices) < num_shots + num_val:
            print(f"Warning: Class {dataset.classes[label]} has fewer than {num_shots + num_val} images. Using available for train/val.")
            # Handle edge cases if necessary, for now strict split
        
        # Select train
        train_idx = indices[:num_shots]
        # Select val
        val_idx = indices[num_shots:num_shots+num_val]
        # Select test
        test_idx = indices[num_shots+num_val:]
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
        
    return train_indices, val_indices, test_indices

def get_dataloaders(data_dir, batch_size=32, num_shots=5, num_val=5, seed=42, transform=None):
    dataset = MetaAlbumDataset(data_dir, transform=transform)
    
    train_idx, val_idx, test_idx = create_few_shot_splits(dataset, num_shots, num_val, seed)
    
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, dataset.classes

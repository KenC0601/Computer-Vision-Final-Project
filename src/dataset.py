import os
import random
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, SyntaxError, Exception) as e:
            print(f"Warning: Corrupted image found at {img_path}. Skipping. Error: {e}")
            # Return a random other image to maintain batch size
            new_idx = random.randint(0, len(self.images) - 1)
            return self.__getitem__(new_idx)

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
        
        train_idx = indices[:num_shots]
        val_idx = indices[num_shots:num_shots+num_val]
        test_idx = indices[num_shots+num_val:]
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
        
    return train_indices, val_indices, test_indices

def create_full_splits(dataset, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Creates standard train/val/test splits using all data.
    Handles class imbalance/rare classes manually to avoid stratification errors.
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
        n_samples = len(indices)
        
        # Calculate split counts
        n_val = int(n_samples * val_ratio)
        n_test = int(n_samples * test_ratio)
        
        # Ensure at least 1 sample in train if possible
        # If very few samples, prioritize Train > Val > Test
        if n_samples == 1:
            train_indices.extend(indices)
            continue
        elif n_samples == 2:
            train_indices.append(indices[0])
            val_indices.append(indices[1])
            continue
        elif n_samples == 3:
            train_indices.append(indices[0])
            val_indices.append(indices[1])
            test_indices.append(indices[2])
            continue
            
        # Standard split for sufficient samples
        # Ensure train gets the remainder so it's never empty
        if n_val == 0 and val_ratio > 0: n_val = 1
        if n_test == 0 and test_ratio > 0: n_test = 1
        
        if n_val + n_test >= n_samples:
            # Fallback for very small N
            n_val = 1
            n_test = 1
            if n_samples < 3: # Should be caught above
                n_test = 0
        
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test+n_val]
        train_idx = indices[n_test+n_val:]
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
        
    return train_indices, val_indices, test_indices


def create_unseen_5shot_splits(dataset, seed=42, num_shots=5):
    """
    1. Recreates the Full Training Split (Train/Val/Test).
    2. Discards the Train set (which the backbone saw).
    3. Selects 5-shot samples ONLY from the Test set (unseen).
    """
    print("Generating splits to isolate UNSEEN data...")
    
    # 1. Re-generate the exact splits used in full training
    train_idx_full, val_idx_full, test_idx_full = create_full_splits(dataset, seed=seed)
    
    print(f"Full Dataset Split: Train={len(train_idx_full)}, Val={len(val_idx_full)}, Test={len(test_idx_full)}")
    print("Selecting 5-shot samples from the TEST split only...")
    
    # 2. Work only with the Test indices
    test_indices_by_class = {}
    for idx in test_idx_full:
        _, target = dataset.images[idx]
        if target not in test_indices_by_class:
            test_indices_by_class[target] = []
        test_indices_by_class[target].append(idx)
        
    # 3. Select 5 shots per class from this unseen pool
    rng = random.Random(seed + 1) 
    
    final_train_indices = [] # The 5 shots
    final_test_indices = []  # The rest of the unseen data
    
    for cls_idx in range(len(dataset.classes)):
        if cls_idx not in test_indices_by_class:
            continue
            
        indices = test_indices_by_class[cls_idx]
        rng.shuffle(indices)
        
        if len(indices) <= num_shots:
            final_train_indices.extend(indices)
        else:
            final_train_indices.extend(indices[:num_shots])
            final_test_indices.extend(indices[num_shots:])
            
    print(f"5-Shot Split (Unseen): Train={len(final_train_indices)}, Test={len(final_test_indices)}")
    return final_train_indices, final_test_indices


def get_dataloaders(data_dir, batch_size=32, num_shots=5, num_val=5, seed=42, transform=None, use_full_data=False):
    dataset = MetaAlbumDataset(data_dir, transform=transform)
    
    if use_full_data:
        print("Using FULL dataset splits (80/10/10)...")
        train_idx, val_idx, test_idx = create_full_splits(dataset, seed=seed)
    else:
        print(f"Using {num_shots}-shot splits...")
        train_idx, val_idx, test_idx = create_few_shot_splits(dataset, num_shots, num_val, seed)
    
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, dataset.classes

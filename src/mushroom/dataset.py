"""
Define the Mushroom data structure
"""

import os
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MushroomImage:
    img_path: str
    class_id: int


def _dataset_primer(path):
    """Pre-populate a list of MushroomImages to make splitting with sklearn simple"""
    items = []
    classnames = []
    for i, directory in enumerate(os.listdir(path)):
        classnames.append(directory)
        for img in os.listdir(os.path.join(path, directory)):
            items.append(MushroomImage(img_path=os.path.join(path, directory, img),
                                       class_id=i))

    return items, classnames


class MushroomDataset(Dataset):
    def __init__(self, transform, items, classnames):
        self.items = items
        self.transform = transform
        self.classnames = classnames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """Load and return the transformed image associated with the index,
        or None if the image is missing or corrupted"""
        item = self.items[idx]

        try:
            img = Image.open(item.img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            assert (img.mode == 'RGB')
            return self.transform(img), item.class_id, idx

        except AssertionError:
            return None  # Return a None top be filtered later in the collate_fn

        except OSError:
            return None


def constant_ratio_split(items: List[MushroomImage], classnames: List[str],
                        ratio: float) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Split a dataset so that the ratio of classes in the test set equals to the train set.
    Also returns item counts per classes"""

    # Build a list of lists from the item indices
    classes = [[] for i in range(len(classnames))]
    for i, item in enumerate(items):
        classes[item.class_id].append(i)

    # Shuffle and split each class list
    train_idx = []
    train_counts = []
    test_idx = []
    test_counts = []
    for cls in classes:
        np.random.shuffle(cls)
        split_idx = int(len(cls) * ratio)
        train_idx += cls[:split_idx]
        train_counts.append(len(cls[:split_idx]))
        test_idx += cls[split_idx:]
        test_counts.append(len(cls[split_idx:]))

    # Shuffle again for good measure, in case the dataloader doesn't
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return train_idx, train_counts, test_idx, test_counts


def kaggle_mushrooms(path, transform):
    """processes the Kaggle mushroom image classification dataset"""
    # Discover training samples
    dataset_items, classnames = _dataset_primer(path=path)

    # Shuffle and split
    np.random.shuffle(dataset_items)
    train_idx, train_count, test_idx, test_count = constant_ratio_split(dataset_items, classnames, ratio=0.8)
    train_set = MushroomDataset(items=[dataset_items[i] for i in train_idx],
                                transform=transform, classnames=classnames)
    test_set = MushroomDataset(items=[dataset_items[i] for i in test_idx],
                               transform=transform, classnames=classnames)

    return train_set, train_count, test_set, test_count



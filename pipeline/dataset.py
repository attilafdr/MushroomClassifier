"""
Define the Mushroom data structure
"""

import os

from dataclasses import dataclass
from PIL import Image

from torch.utils.data import Dataset


@dataclass(frozen=True)
class MushroomImage:
    img_path: str
    class_id: int


def dataset_primer(path):
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



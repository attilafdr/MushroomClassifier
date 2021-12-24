import torch
import argparse
import numpy as np
import logging

from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomRotation
from torchvision.transforms import RandomApply, RandomEqualize, ColorJitter, RandomResizedCrop

from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18, resnet50

from pipeline.dataset import MushroomDataset, dataset_primer, MushroomImage

import logging
from typing import List, Tuple

def get_model(device, n_classes, counts):
    # The classifier is a ResNet18 with a random top layer dim = n_classes
    classifier = resnet18(pretrained=True)
    classifier.fc = Linear(classifier.fc.in_features, n_classes)
    classifier = classifier.to(device)
    # Invert and normalize counts
    weights = [1 / (count/sum(counts)) * (max(counts)/sum(counts)) for count in counts]
    logging.info(f'Class weights: {weights}')
    criterion = CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    optimizer = SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    return classifier, criterion, optimizer


def representatve_split(items: List[MushroomImage], classnames: List[str],
                        split: float) -> Tuple[List[int], List[int], List[int], List[int]]:
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
        split_idx = int(len(cls) * split)
        train_idx += cls[:split_idx]
        train_counts.append(len(cls[:split_idx]))
        test_idx += cls[split_idx:]
        test_counts.append(len(cls[split_idx:]))

    # Shuffle again for good measure, in case the dataloader doesn't
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return train_idx, train_counts, test_idx, test_counts


def session_config():
    torch.backends.cudnn.enabled = True
    torch.manual_seed(0)
    np.set_printoptions(precision=3)

    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def collate_fn(batch):
    # Use a custom collate_fn to handle corrupted samples dynamically in the DataLoader
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def train():
    # Load config (model, session)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    device = session_config()

    # Load the transformations required for inference
    test_transform = Compose([Resize((300, 300)),
                              ToTensor()])

    # Define dataset and transforms to be applied
    dataset_items, classnames = dataset_primer(path='/home/attila/Datasets/Mushrooms')

    # Shuffle and split
    np.random.shuffle(dataset_items)
    train_idx, train_count, test_idx, test_count = representatve_split(dataset_items, classnames, split=0.8)
    train_set = MushroomDataset(items=[dataset_items[i] for i in train_idx],
                                transform=test_transform, classnames=classnames)
    test_set = MushroomDataset(items=[dataset_items[i] for i in test_idx],
                               transform=test_transform, classnames=classnames)
    logging.info(f'Train set: {len(train_set)} samples, Test set: {len(test_set)} samples')
    logging.info(f'Train class counts: {train_count}, Test class counts: {test_count}')

    # Load or initialise the model
    model, criterion, optimizer = get_model(device=device, n_classes=len(train_set.classnames), counts=train_count)

    '''
    Train the model
    '''

    # Define a step scheduler to reduce learning rate for fine-tuning
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    # Define the train data loader with the custom collate function. A small batch size is used, to increase
    # variation of some random transforms that apply with the same parameters across the whole batch.
    # Set pin_memory=True to speed up data loading to CUDA GPUs
    train_loader = DataLoader(train_set, batch_size=4, collate_fn=collate_fn, pin_memory=True,
                              sampler=RandomSampler(train_set, replacement=False), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=4, collate_fn=collate_fn, pin_memory=True, num_workers=4)

    # Set model to train mode
    model.train()

    for epoch in tqdm(range(1, 31)):
        epoch_loss = 0
        for batch in train_loader:
            data, target, _ = batch
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.squeeze())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()
        logging.info(f'Train Loss: {epoch_loss}')

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                # Track all results
                prediction_all = np.array([], dtype=np.uint8).reshape(0, 1)
                target_all = np.array([], dtype=np.uint8).reshape(0, 1)
                test_epoch_loss = 0
                for test_batch in test_loader:
                    data, target, _ = test_batch
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output.squeeze(), target.squeeze())
                    test_epoch_loss += loss.item()
                    prediction = output.argmax(dim=1, keepdim=True)

                    prediction_all = np.vstack([prediction_all, prediction.cpu().numpy()])
                    target_all = np.vstack([target_all, target.view_as(prediction).cpu().numpy()])

                metrics = {'f1': f1_score(target_all.flatten(), prediction_all.flatten(), average=None)}
                cm = confusion_matrix(target_all.flatten(), prediction_all.flatten())
                logging.info('Confusion matrix:')
                logging.info(cm)
                logging.info(f'Epoch {epoch:04d} F1 Scores:')
                logging.info('\n'.join("{}: {}".format(cls, f1) for cls, f1 in zip(train_set.classnames, metrics['f1'])))
            model.train()

            # Save checkpoint
            torch.save(model.state_dict(), f'../model_repository/ckp{epoch:04d}.pt')


if __name__ == '__main__':
    train()

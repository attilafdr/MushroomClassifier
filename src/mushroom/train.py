
import logging
import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD, lr_scheduler
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Compose, ToTensor, Resize

from mushroom.dataset import kaggle_mushrooms
from mushroom.models.resnet18 import Resnet18Classifier


def get_model(device, n_classes, counts):
    """Return a model, lerarning critetion and optimizer
    TODO: This should be done by a model manager if multiple models are available"""
    classifier = Resnet18Classifier(n_classes=n_classes, pretrained=True)
    classifier = classifier.to(device)
    # Invert and normalize counts
    weights = [1 / (count/sum(counts)) * (max(counts)/sum(counts)) for count in counts]
    logging.info(f'Class weights: {weights}')
    criterion = CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    optimizer = SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    return classifier, criterion, optimizer


def session_config():
    """Config file placeholder"""
    torch.backends.cudnn.enabled = True
    torch.manual_seed(0)
    np.set_printoptions(precision=3)

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def collate_fn(batch):
    """Use a custom collate_fn to handle corrupted samples dynamically in the DataLoader"""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def train():
    # Load config (model, session)
    device = session_config()

    # Define the basic transformations required for minimal training and inference
    transform = Compose([Resize((300, 300)),
                         ToTensor()])

    # Load the Kaggle Mushroom image classification dataset
    train_set, train_counts, test_set, test_counts = kaggle_mushrooms(path='/home/attila/Datasets/Mushrooms',
                                                                      transform=transform)

    # Dataset sanity check
    logging.info(f'Train set: {len(train_set)} samples, Test set: {len(test_set)} samples')
    logging.info(f'Train class counts: {train_counts}, Test class counts: {test_counts}')

    # Load or initialise the model
    model, criterion, optimizer = get_model(device=device, n_classes=len(train_set.classnames), counts=train_counts)

    '''
    Train the model
    '''

    # Define a step scheduler to reduce learning rate for longer trains
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Define the train data loader with the custom collate function
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
            torch.save(model.state_dict(), f'../model-store/ckp{epoch:04d}.pt')


if __name__ == '__main__':
    train()

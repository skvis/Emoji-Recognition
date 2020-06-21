import config
import os
import torch
from torchvision import transforms, datasets


def data_augumentation():

    data_transforms = {
        'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])]),

        'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])])}

    return data_transforms


def data_loader():

    data_transforms = data_augumentation()

    image_datasets = {x: datasets.ImageFolder(os.path.join(config.DATA_PATH, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=config.BATCH_SIZE,
                                                  shuffle=config.SHUFFLE,
                                                  num_workers=config.NUM_WORKER)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    return image_datasets, dataloaders, dataset_sizes, class_names

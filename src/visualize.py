import config
import data_preprocess
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


def load_data(dataloaders, class_names):
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


if __name__ == '__main__':
    _, dataloaders, _, class_names = data_preprocess.data_loader()
    load_data(dataloaders, class_names)

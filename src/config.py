import torch

DATA_PATH = '../input/happy-or-sad/'
MODEL_PATH = '../models/'
MODEL_NAME = 'res_net.pth'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
SHUFFLE = True
NUM_WORKER = 4

LEARNING_RATE = 0.001
MOMENTUM = 0.9

NUM_EPOCHS = 1

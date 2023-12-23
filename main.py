import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import CIFAR10_CNN, MNIST_CNN
from torch.utils.data import Dataset, DataLoader, Subset
from dataset import add_pattern_trigger, add_pixel_trigger, poison_dataset
from utils import train, eval, print_model_perform
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import random
## configure the hyper-parameter
import argparse
parser = argparse.ArgumentParser(description="Badnets: Identifying vulnerabilities in the machine learning model supply chain")
parser.add_argument('--data', default='cifar10', help='Which dataset to use (mnist or cifar10, default: mnist)')
parser.add_argument('--datapath', default='/home/pc/zhujie/data/cifar10', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--batchsize', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of the model, default: 0.001')
parser.add_argument('--attack_type', type=str, default='single_target', help='single_target or all_to_all')
parser.add_argument('--poison_method', type=str, default='pattern', help='pixel or pattern')
parser.add_argument('--trigger_label', type=int, default=0, help='label backdoored image for single target attack type')
parser.add_argument('--epoch', type=int, default=50, help='Number of epochs to train backdoor model, default: 50')
parser.add_argument('--poisoned_portion', type=float, default=0.1, help='posioning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--optmizer', type=str, default='Adam', help='Adam or SGD')
args = parser.parse_args()


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if args.data=='cifar10':
    setup_seed(17) ## for cifar10
elif args.data=='mnist':
    setup_seed(42) ## for mnist


# Initialize the network and load dataset 
if args.data == 'mnist':
    model = MNIST_CNN()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.MNIST(root=args.datapath, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.datapath, train=False, download=True, transform=transform)
elif args.data == 'cifar10':
    model = CIFAR10_CNN()
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)])
    train_set = datasets.CIFAR10(root=args.datapath, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=args.datapath, train=False, download=True, transform=transform)
else:
    raise ValueError(f'currently we do not support dataset:{args.data}')



clean_model = model
bad_model = deepcopy(clean_model)


# Initialize the optimizer and loss function

"""
if we use Adam, it converge faster and the backdoor perfromance is better. 
However, we recommend to use SGD to produce a general condition for mnist
And we recommend to use Adam to produce a general condition for cifar10
"""
if args.optmizer == 'Adam':
    clean_model_optimizer = optim.Adam(clean_model.parameters(), lr=args.learning_rate)
    bad_model_optimizer = optim.Adam(bad_model.parameters(), lr=args.learning_rate)
elif args.optmizer == 'SGD':
    clean_model_optimizer = optim.SGD(clean_model.parameters(), lr=args.learning_rate)
    bad_model_optimizer = optim.SGD(bad_model.parameters(), lr=args.learning_rate)

criteria = nn.CrossEntropyLoss()

if args.poison_method == 'pixel':
    poisoned_train_set = poison_dataset(deepcopy(train_set), args.poisoned_portion, args.attack_type, args.trigger_label, add_pixel_trigger, {'position': (26, 26) if args.data=='mnist' else (30,30), 'size': 1, 'intensity': 255.0})
    poisoned_test_set = poison_dataset(deepcopy(test_set), 1.0, args.attack_type, args.trigger_label, add_pixel_trigger, {'position': (26, 26) if args.data=='mnist' else (30,30), 'size': 1, 'intensity': 255.0})
elif args.poison_method == 'pattern':

    """
    for better implementation, we do not use the proposed pattern in the paper
    Instead, we use a much simpler pattern, i.e., a square pattern
    """
    sample_pattern = torch.ones((1, 2, 2)) * 255  # This is just a placeholder pattern 

    poisoned_train_set = poison_dataset(deepcopy(train_set), args.poisoned_portion, args.attack_type, args.trigger_label, add_pattern_trigger, {'pattern': sample_pattern})
    poisoned_test_set = poison_dataset(deepcopy(test_set), 1.0, args.attack_type, args.trigger_label, add_pattern_trigger, {'pattern': sample_pattern})
else:
    raise ValueError(f'we do not support {args.poison_method}')

train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=True, num_workers=4)

poisoned_train_loader = DataLoader(poisoned_train_set, batch_size=args.batchsize, shuffle=True, num_workers=4)
poisoned_test_loader = DataLoader(poisoned_test_set, batch_size=args.batchsize, shuffle=True, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clean_model.to(device)
bad_model.to(device)


# Train the model on clean data
print('Train the model on clean data')
for epoch in range(args.epoch):
    loss = train(clean_model, device, train_loader, clean_model_optimizer, criteria)
    print("# EPOCH%d   loss: %.4f "% (epoch, loss.item()))
eval(clean_model, test_loader, mode='clean')


# # Train the model on poisoned data
print('Train the model on poisoned data')
for epoch in range(args.epoch):
    loss = train(bad_model, device, poisoned_train_loader, bad_model_optimizer, criteria)
    acc_train = eval(bad_model, poisoned_train_loader, mode='backdoor')
    acc_test_ori = eval(bad_model, test_loader, mode='backdoor')
    acc_test_tri = eval(bad_model, poisoned_test_loader, mode='backdoor')

    print("# EPOCH%d   loss: %.4f  training acc: %.4f, ori testing acc: %.4f, trigger testing acc: %.4f\n"\
              % (epoch, loss.item(), acc_train, acc_test_ori, acc_test_tri))

# # Evaluate the bad model
print("# --------------------------evaluation--------------------------")
print("## original test data performance:")
print_model_perform(bad_model, test_loader)
print("## triggered test data performance:")
print_model_perform(bad_model, poisoned_test_loader)




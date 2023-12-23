import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def print_model_perform(model, data_loader):
    model.eval() # switch to eval mode
    y_true = []
    y_predict = []
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_y_predict = model(batch_x)
            batch_y_predict = torch.argmax(batch_y_predict, dim=1)
            y_predict.append(batch_y_predict)
            y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    try:
        target_names_idx = set.union(set(np.array(y_true.cpu())), set(np.array(y_predict.cpu())))
        target_names = [data_loader.dataset.classes[i] for i in target_names_idx]
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=target_names))
    except ValueError as e:
        print(e)

# Define the training process
def train(model, device, train_loader, optimizer, criteria):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criteria(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss

def eval(model, data_loader, mode='backdoor'):
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_y_predict = model(batch_x)
            batch_y_predict = torch.argmax(batch_y_predict, dim=1)
            y_predict.append(batch_y_predict)
            y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    acc = accuracy_score(y_true.cpu(), y_predict.cpu())

    if mode != 'backdoor':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))
        print(f'Acc:{acc}')

    return acc




import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import pandas as pd
from models.tcn.tcn import TemporalConvNet, HybridTCN
from parse_data import *

class MedicalDataset(Dataset):
    """
    change the pandas format to pytorch
    """
    def __init__(self, dataframe):
        self.data = dataframe.values
    def __get__(self, index):
        return self.data[index]

def load_dataset():
    # df_map = pd.read_excel("data/ACCESS 1853 Dataset.xlsx", sheet_name=None)

    # # sample dataset and convert 
    # encounters_dataset = MedicalDataset(df_map["ENCOUNTERS"])

    data = pd.read_csv("data/new_dt.csv")
    # group by STUDY_ID
    # TODO
    # data = data.groupby('STUDY_ID')

    
    # train, valid and test
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    test_size = len(data) - train_size - val_size

    encounters_train, encounters_val = random_split(data, [train_size, val_size])

    return encounters_train, encounters_val

def train_tcn(epoch,data_loader,model,optimizer, device ):
  for batch_idx, (data, target) in enumerate(data_loader):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    output = model(data)
    # Use MSE loss
    loss_function = nn.MSELoss()
    loss = loss_function(output, target)
    loss.backward()

    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(data_loader.dataset),
        100. * batch_idx / len(data_loader), loss.item()))

def eval(data_loader,model,dataset, device):
    loss = 0
    correct = 0
    with torch.no_grad(): # notice the use of no_grad
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            # MSE loss 
            loss_function = nn.MSELoss()
            loss += loss_function(output,target).item()
    loss /= len(data_loader.dataset)
    print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

def train(model, train_dataset, valid_dataset, device):
    train_loader = DataLoader(train_dataset, batch_size_train = 200, shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size_valid = 200, shuffle = True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    eval(valid_loader,model,"Validation", device)
    for epoch in range(1, n_epochs + 1):
        train_tcn(epoch,train_loader,model,optimizer, device)
        eval(valid_loader,model,"Validation", device)
    

def TCN(device):
    
    model = TemporalConvNet(num_inputs, num_channels = 1)
    train_dataset, valid_dataset = load_dataset()

    results = train(model, train_dataset, valid_dataset, device)

    return results

# need change according to patient dataset 
n_epochs = 50
n_splits =  5
learning_rate =  1e-3
log_interval = 100


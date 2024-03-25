import torch
import pandas as pd

SEQ_START_COL = 8

# def read_csv_sequential(csv_file):
#     # Read the CSV file using pandas
#     data = pd.read_csv(csv_file)
    
#     data_sequential = torch.tensor(data.iloc[:, SEQ_START_COL:].values)
#     data_static = torch.tensor(data.iloc[:, :SEQ_START_COL].values)
#     assert data_sequential.shape == (12)
    
#     # Reshape the data_sequential tensor to a 4x3 tensor
#     # the rows of this tensor are the time steps
#     # the columns are the TOD, TD, and glucose
#     data_sequential = data_sequential.view(4, 3)
    
#     return data_sequential, data_static

def split_sequential(tensor:torch.Tensor):
    
    data_sequential = tensor[:, SEQ_START_COL:]
    data_static = tensor[:, :SEQ_START_COL]
    assert data_sequential.shape == (12)
    
    # Reshape the data_sequential tensor to a 4x3 tensor
    # the rows of this tensor are the time steps
    # the columns are the TOD, TD, and glucose
    data_sequential = data_sequential.view(4, 3)
    
    return data_sequential, data_static
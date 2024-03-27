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
    '''
    Splits the input data instance tensor into sequential and static components, and target glucose
    Args:
        tensor: torch.Tensor - the input data instance tensor
        Returns:
        x_sequential: torch.Tensor - the sequential data tensor containing TOD1, TD1, glucose1, TOD2, TD2, glucose2, TOD3, TD3, glucose3
        x_static: torch.Tensor - the static data tensor containing the patient's attributes
        target: torch.float - the target glucose value
    '''
    slice1 = tensor[SEQ_START_COL:] 
    slice2 = tensor[-3:-1]
    x_sequential = torch.cat((slice1, slice2))
    target = tensor[-1]
    x_static = tensor[:, :SEQ_START_COL]
    assert x_sequential.shape == (9)
    
    # Reshape the x_sequential tensor to a 3x3 tensor
    # the rows of this tensor are the time steps
    # the columns are the TOD, TD, and glucose
    x_sequential = x_sequential.view(3, 3)
    
    return x_sequential, x_static, target
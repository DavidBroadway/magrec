import torch
from magrec.prop.Fourier import FourierTransform2d


class FourierNet(torch.nn.Module):
    
    def __init__(self):
        super(FourierNet, self).__init__()
        self.clu = ctorch.complex_relu()
        
class FieldNet(torch.nn.Module):
    
    def __init__(self):
        self.module = torch.nn.Sequential()
import torch.nn.functional as F
import torch 
import numpy as np

source = torch.rand((32, 40, 501))
shape = np.array(source.shape) 
print("Source Shape", shape)

padding_size = 5 - (shape[-1] % 5)
print("Padding Size", padding_size)

padding = [0]*(2*len(shape))
print("Padding Shape", len(padding))

padding[1] = padding_size
print("Padding", padding)

result = F.pad(input=source, pad=tuple(padding), mode='constant', value=0)
print("Padded", result.shape)

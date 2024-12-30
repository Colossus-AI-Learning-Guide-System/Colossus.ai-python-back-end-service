import torch
print(torch.backends.mps.is_available())  # Check if Metal Performance Shaders (MPS) is available
print(torch.backends.mps.is_built())     # Check if PyTorch is built with MPS support
# Check whether any GPU is available or not

import torch
import os   
cuda_available = torch.cuda.is_available()
print(cuda_available)
if cuda_available:
    gpu_count = torch.cuda.device_count()
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Using CPU")

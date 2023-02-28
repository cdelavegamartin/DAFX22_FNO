import torch

print(f"Torch installation has CUDA: {torch.has_cuda}")
print(f"Torch CUDA version: {torch.version.cuda}")
print(f"Any GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPU available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i} name: {torch.cuda.get_device_name(i)}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(
        f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )

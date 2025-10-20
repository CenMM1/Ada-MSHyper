import torch

# Load the .pt file
data = torch.load('compressed_data/dev.pt')

# Print the type of the loaded data
print(f"Data type: {type(data)}")

# If it's a dictionary, inspect keys and values
if isinstance(data, dict):
    print("Dictionary contents:")
    for key, value in data.items():
        print(f"  Key: {key}")
        print(f"    Type: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"    Shape: {value.shape}")
        if hasattr(value, 'dtype'):
            print(f"    Dtype: {value.dtype}")
        # Print a sample if it's a tensor
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            print(f"    Sample values: {value.flatten()[:10].tolist()}")  # First 10 elements
        print()
# If it's a tensor, inspect directly
elif isinstance(data, torch.Tensor):
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Sample values: {data.flatten()[:10].tolist()}")
# If it's a tuple or list, inspect elements
elif isinstance(data, (tuple, list)):
    print(f"Length: {len(data)}")
    for i, item in enumerate(data):
        print(f"  Item {i}: Type {type(item)}")
        if hasattr(item, 'shape'):
            print(f"    Shape: {item.shape}")
        if hasattr(item, 'dtype'):
            print(f"    Dtype: {item.dtype}")
        if isinstance(item, torch.Tensor) and item.numel() > 0:
            print(f"    Sample values: {item.flatten()[:10].tolist()}")
        print()
else:
    print(f"Data: {data}")
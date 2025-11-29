import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from fer_extractor import load_fer_model, extract_video_embeddings

def process_sample(mosi_root='MOSI', num_dirs=5, output_file='mosi_sample_embeddings.pt'):
    raw_path = os.path.join(mosi_root, 'Raw')
    
    # Get list of all subdirectories
    all_dirs = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
    
    # Select first 5 directories
    target_dirs = all_dirs[:num_dirs]
    print(f"Processing directories: {target_dirs}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_fer_model(device)
    
    results = {}
    
    for video_id in target_dirs:
        video_dir = os.path.join(raw_path, video_id)
        
        # Find all mp4 files in the directory
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        for video_file in video_files:
            clip_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_dir, video_file)
            key = f"{video_id}_{clip_id}"
            
            print(f"Processing {key}...")
            try:
                embeddings = extract_video_embeddings(video_path, model, device)
                if embeddings.size > 0:
                    results[key] = torch.from_numpy(embeddings)
                    print(f"  -> Shape: {embeddings.shape}")
                else:
                    print(f"  -> No faces detected or empty video.")
            except Exception as e:
                print(f"  -> Error: {e}")
                
    # Save results
    print(f"\nSaving results to {output_file}...")
    torch.save(results, output_file)
    print("Done.")

def inspect_results(pt_file='mosi_sample_embeddings.pt'):
    print(f"\n--- Inspecting {pt_file} ---")
    if not os.path.exists(pt_file):
        print("File not found.")
        return

    data = torch.load(pt_file)
    print(f"Total clips processed: {len(data)}")
    
    # Inspect first few items
    for i, (key, tensor) in enumerate(data.items()):
        if i >= 3: break
        print(f"\nKey: {key}")
        print(f"Type: {type(tensor)}")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"First 5 values of first frame:\n{tensor[0, :5]}")
        print(f"Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")

if __name__ == "__main__":
    process_sample(output_file='mosi_sample_embeddings_1.pt')
    inspect_results()

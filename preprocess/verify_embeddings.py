import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

def verify_pt_file(pt_file='mosi_sample_embeddings.pt'):
    if not os.path.exists(pt_file):
        print(f"File {pt_file} not found.")
        return

    print(f"Loading {pt_file}...")
    data = torch.load(pt_file)
    print(f"Loaded {len(data)} clips.")

    if len(data) == 0:
        print("No data to verify.")
        return

    # Verify a few random clips
    keys = list(data.keys())
    sample_keys = keys[:3] # Check first 3

    for key in sample_keys:
        embedding = data[key] # [T, 512]
        
        print(f"\n--- Verifying Clip: {key} ---")
        print(f"Shape: {embedding.shape}")
        
        if embedding.dim() != 2 or embedding.shape[1] != 512:
            print("❌ ERROR: Incorrect shape. Expected [T, 512]")
            continue
            
        # 1. Value Statistics
        print(f"Values - Min: {embedding.min():.4f}, Max: {embedding.max():.4f}, Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
        
        # Check for NaNs
        if torch.isnan(embedding).any():
            print("❌ ERROR: Contains NaNs!")
        else:
            print("✅ No NaNs.")
            
        # Check for Zero vectors (failed inference?)
        norms = torch.norm(embedding, dim=1)
        if (norms == 0).any():
            print(f"⚠️ WARNING: Found {(norms==0).sum()} zero-norm vectors (frames).")
        
        # 2. Temporal Consistency (Cosine Similarity between consecutive frames)
        if embedding.shape[0] > 1:
            # Normalize embeddings
            emb_norm = F.normalize(embedding, p=2, dim=1)
            
            # Compute similarity between t and t+1
            sims = torch.sum(emb_norm[:-1] * emb_norm[1:], dim=1)
            
            avg_sim = sims.mean().item()
            min_sim = sims.min().item()
            
            print(f"Temporal Consistency (Cosine Sim): Avg={avg_sim:.4f}, Min={min_sim:.4f}")
            
            if avg_sim > 0.9:
                print("✅ High temporal consistency (Expected for video).")
            elif avg_sim > 0.7:
                print("⚠️ Moderate temporal consistency.")
            else:
                print("❌ Low temporal consistency. Check alignment stability.")
        else:
            print("⚠️ Too few frames to check temporal consistency.")

if __name__ == "__main__":
    verify_pt_file()
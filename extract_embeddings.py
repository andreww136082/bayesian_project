import torch
import numpy as np
import os
from tqdm import tqdm
from picme_src.argparser import finetune_arg_parser
from picme_src.models import MultiModalBaseline
from picme_src import data, picme_utils

def extract_features_to_numpy(args, dataloader, encoder, device, split="train"):
    encoder.eval()
    all_fused_embeddings = []
    all_labels = []
    all_missing_masks = []
    
    print(f"Extracting {split} embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {split}"):
            # 1. Prep data using your existing utility
            modalities_data, modalities_type, mask_rad, mask_ds, ts_lens, labels, present_mask_batch = \
                picme_utils.prep_data(args.modalities, batch, device, missing=True)
                
            # 2. Get raw embeddings
            embeddings = encoder(modalities_data, modalities_type, mask_rad, mask_ds, ts_lens)
            
            # 3. Apply missingness tokens (Replicating your train_model logic)
            final_embeddings = []
            for i, modality_name in enumerate(args.modalities):
                emb = embeddings[i]
                token = encoder.modality_token_map[modality_name]
                mask = present_mask_batch[:, i].unsqueeze(1).to(device)
                final_emb = torch.where(mask, emb, token)
                final_embeddings.append(final_emb)
            
            # 4. Fuse embeddings (e.g., 'concatenation' -> [B, 1280] assuming 5x256)
            fused_embeddings = picme_utils.secure_fusion(final_embeddings, device, args.fusion_method)
            
            all_fused_embeddings.append(fused_embeddings.cpu().numpy())
            
            # Ensure labels are 1D
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            all_labels.append(labels.cpu().numpy())
            
            # Save the missingness indicator matrix
            all_missing_masks.append(present_mask_batch.cpu().numpy())

    # Stack into flat arrays
    X_matrix = np.concatenate(all_fused_embeddings, axis=0)
    Y_vector = np.concatenate(all_labels, axis=0)
    M_matrix = np.concatenate(all_missing_masks, axis=0)
    
    output_dir = "r_bayesian_data"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f"{split}_X_embeddings.npy"), X_matrix)
    np.save(os.path.join(output_dir, f"{split}_Y_labels.npy"), Y_vector)
    np.save(os.path.join(output_dir, f"{split}_M_indicators.npy"), M_matrix)
    print(f"Saved {split} matrices. X shape: {X_matrix.shape}")

if __name__ == "__main__":
    args = finetune_arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    _DATA_DIR = "/users/awang463/data/awang463/missing_modalities/everything"
    task_name = "in-hospital-mortality"
    
    # Load Static Dataloaders for extraction
    train_input_list, val_input_list, test_input_list = data.get_dataloaders(
        args.modalities, args.batch_size[0], _DATA_DIR, task_name, test=True, 
        dataset_tag="missing", evaluate_with_missing=True
    )
    
    dataloaders_dict = {
        "train": data.MergeLoader(train_input_list),
        "val": data.MergeLoader(val_input_list),
        "test": data.MergeLoader(test_input_list)
    }
    
    # Load Baseline Model
    encoder = MultiModalBaseline(
        ts_input_dim=76, demo_input_dim=44, projection_dim=256, args=args, device=device
    ).to(device)
    
    # Optional: encoder.load_state_dict(torch.load(args.state_dict, map_location=device))
    
    extract_features_to_numpy(args, dataloaders_dict["train"], encoder, device, "train")
    extract_features_to_numpy(args, dataloaders_dict["test"], encoder, device, "test")


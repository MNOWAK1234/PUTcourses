import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from src.model.patch_tst import PatchTST
from src.data_loader import get_dataloader
from config import config

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_loader, test_set = get_dataloader(
        config["file_path"], batch_size=config["batch_size"], flag='test', 
        size=(config["lookback_len"], config["pred_len"])
    )
    
    n_channels = test_set.data_x.shape[1]
    model = PatchTST(
        n_channels=n_channels,
        lookback_len=config["lookback_len"],
        patch_len=config["patch_len"],
        stride=config["stride"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_layers=config["n_layers"],
        pred_len=config["pred_len"]
    ).to(device)
    
    model.load_state_dict(torch.load("patch_tst_traffic_L24_P96.pth", map_location=device))
    model.eval()
    
    mse_fn, mae_fn = nn.MSELoss(), nn.L1Loss()
    mses, maes = [], []
    
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            
            mses.append(mse_fn(output, batch_y).item())
            maes.append(mae_fn(output, batch_y).item())
            
            all_outputs.append(output.cpu())
            all_targets.append(batch_y.cpu())

    print(f"Final Metrics | MSE: {np.mean(mses):.4f}, MAE: {np.mean(maes):.4f}")

    sample_output = all_outputs[0][0].numpy().T
    sample_target = all_targets[0][0].numpy().T
    
    true_unscaled = test_set.scaler.inverse_transform(sample_target)
    pred_unscaled = test_set.scaler.inverse_transform(sample_output)
    
    junctions_to_show = random.sample(range(n_channels), min(10, n_channels))
    num_plots = len(junctions_to_show)
    cols = 2
    rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, j_idx in enumerate(junctions_to_show):
        axes[i].plot(true_unscaled[:, j_idx], label="Ground Truth", marker='o', alpha=0.7)
        axes[i].plot(pred_unscaled[:, j_idx], label="Prediction", marker='x', alpha=0.7)
        axes[i].set_title(f"Junction {j_idx + 1}")
        axes[i].set_xlabel("Steps")
        axes[i].set_ylabel("Value")
        axes[i].legend()
        axes[i].grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
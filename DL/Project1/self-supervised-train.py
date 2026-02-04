import torch
import torch.nn as nn
from src.model.self_supervised_patch_tst import SelfSupervisedPatchTST
from src.data_loader import get_dataloader
from config import config

def masked_mse_loss(pred, target, mask):
    if target.ndim < pred.ndim:
        target = target.unsqueeze(-1)

    mask = mask.unsqueeze(-1).unsqueeze(-1)
    loss = (pred - target) ** 2
    loss = loss * mask
    return loss.sum() / mask.sum()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, train_set = get_dataloader(
        config["file_path"], config["batch_size"], flag='train', 
        size=(config["lookback_len"], config["pred_len"])
    )
    n_channels = train_set.data_x.shape[1]

    model = SelfSupervisedPatchTST(
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

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    model.train()
    for epoch in range(config["epochs"]):
        epoch_loss = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs, patches, mask = model(batch_x)

            loss = masked_mse_loss(outputs, patches, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "patch_tst_traffic.pth")

if __name__ == "__main__":
    train()
import torch
import torch.nn as nn

class InstanceNormalization(nn.Module):
    """
    Standardize each channel independently to have mean 0 and std 1.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode='norm', stats=None):
        if mode == 'norm':
            # x: [Batch, 1, Lookback_Len]
            mean = x.mean(dim=-1, keepdim=True)
            std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
            x = (x - mean) / std
            return x, (mean, std)
        
        elif mode == 'denorm':
            mean, std = stats
            return x * std + mean

class PatchingLayer(nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        """
        Input: [Batch * Channels, 1, Lookback_Len]
        Output: [Batch * Channels, Num_Patches, Patch_Len]
        """
        # Create sliding windows (patches)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.squeeze(1) 
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, d_model, max_patches=1000):
        super().__init__()
        self.projection = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Input: [Batch * Channels, Num_Patches, Patch_Len]
        Output: [Batch * Channels, Num_Patches, d_model]
        """
        n_patches = x.size(1)
        x = self.projection(x)
        x = x + self.pos_embed[:, :n_patches, :]
        return self.dropout(x)
    
class NoOverlapPatchingLayer(nn.Module):
    def __init__(self, patch_len):
        super().__init__()
        self.patch_len = patch_len

    def forward(self, x):
        """
        Input: [Batch * Channels, 1, Lookback_Len]
        Output: [Batch * Channels, Num_Patches, Patch_Len]
        """
        # Create non-overlapping windows (patches)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = x.squeeze(1)
        return x

class MaskingLayer(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B, N, P = x.shape
        num_mask = max(1, int(self.mask_ratio * N))

        mask = torch.ones(B, N, device=x.device, dtype=x.dtype)

        for b in range(B):
            masked_indices = torch.randperm(N)[:num_mask]
            mask[b, masked_indices] = 0

        x_masked = x * mask.unsqueeze(-1)
        return x_masked, mask
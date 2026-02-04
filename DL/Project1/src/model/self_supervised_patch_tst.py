import torch
import torch.nn as nn
from src.model.transformer import TransformerEncoder
from src.model.layers import InstanceNormalization, NoOverlapPatchingLayer, PatchEmbedding, MaskingLayer

class SelfSupervisedPatchTST(nn.Module):
    def __init__(self, n_channels, lookback_len, patch_len, stride, 
                 d_model, n_heads, d_ff, n_layers, pred_len, dropout=0.1, mask_ratio=0.4):
        super().__init__()
        
        self.n_channels = n_channels
        self.patch_len = patch_len
        
        self.revin = InstanceNormalization()

        # make non-overlapping patches and apply mask
        self.patching = NoOverlapPatchingLayer(patch_len)
        self.masking = MaskingLayer(mask_ratio)

        # this is the same
        self.embedding = PatchEmbedding(patch_len, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        
        self.num_patches = lookback_len // patch_len

        # reconstruction head instead of prediction head
        self.head = nn.Linear(d_model, patch_len * n_channels)

    def forward(self, x):
        """
        Input x: [Batch, Channels, Lookback_Len]
        """
        b, c, l = x.shape
        x = x.reshape(b * c, 1, l) # [B, C, L] -> [B*C, 1, L]
        
        x, stats = self.revin(x, mode='norm')
        patches = self.patching(x)       # [B*C, Num_Patches, Patch_Len]
        x, mask = self.masking(patches)  # [B*C, Num_Patches, Patch_Len]
        x = self.embedding(x)            # [B*C, Num_Patches, d_model]
        x = self.encoder(x)              # [B*C, Num_Patches, d_model]
        x = self.head(x)                 # [B*C, Num_patches, Patch_len * c]
        
        x = x.view(b*c, self.num_patches, self.patch_len, self.n_channels)
        x = x.reshape(b*c, 1, -1)
        x = self.revin(x, mode='denorm', stats=stats)
        x = x.view(b*c, self.num_patches, self.patch_len, self.n_channels)
        
        patches = patches.unsqueeze(-1)
        return x, patches, mask
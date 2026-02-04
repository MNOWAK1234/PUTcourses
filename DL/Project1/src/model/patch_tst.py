import torch
import torch.nn as nn
from src.model.transformer import TransformerEncoder
from src.model.layers import InstanceNormalization, PatchingLayer, PatchEmbedding

class PatchTST(nn.Module):
    def __init__(self, n_channels, lookback_len, patch_len, stride, 
                 d_model, n_heads, d_ff, n_layers, pred_len, dropout=0.1):
        super().__init__()
        
        self.n_channels = n_channels
        self.pred_len = pred_len
        
        self.revin = InstanceNormalization()
        self.patching = PatchingLayer(patch_len, stride)
        self.embedding = PatchEmbedding(patch_len, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        
        self.num_patches = (lookback_len - patch_len) // stride + 1
        self.head = nn.Linear(d_model * self.num_patches, pred_len)

    def forward(self, x):
        """
        Input x: [Batch, Channels, Lookback_Len]
        """
        b, c, l = x.shape
        x = x.reshape(b * c, 1, l) # [B, C, L] -> [B*C, 1, L]
        
        x, stats = self.revin(x, mode='norm')
        x = self.patching(x)       # [B*C, Num_Patches, Patch_Len]
        x = self.embedding(x)      # [B*C, Num_Patches, d_model]
        x = self.encoder(x)        # [B*C, Num_Patches, d_model]
        x = x.reshape(b * c, -1)   # [B*C, Num_Patches * d_model]
        x = self.head(x)           # [B*C, Pred_Len]
        
        x = x.reshape(b * c, 1, -1)
        x = self.revin(x, mode='denorm', stats=stats)
        x = x.reshape(b, c, self.pred_len)
        
        return x
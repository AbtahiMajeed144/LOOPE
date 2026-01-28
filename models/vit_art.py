"""
ViT with Art PE (LOOPE)
Vision Transformer with learnable content-aware positional encoding.
"""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from models.positional_encodings import get_zero_pos_embed, ArtPE


class ViTArt(nn.Module):
    """ViT model with Art (LOOPE) positional encoding."""
    
    def __init__(self, num_classes):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        _, self.seq_length, self.hidden_dim = self.model.vit.embeddings.position_embeddings.shape
        
        # Replace position embeddings with zeros (we'll add our own)
        self.model.vit.embeddings.position_embeddings = get_zero_pos_embed(self.hidden_dim, self.seq_length)
        
        # Create Art PE module
        self.pos_embedding = ArtPE(self.seq_length, self.hidden_dim)
        
        # Get model components
        self.patch = self.model.vit.embeddings.patch_embeddings
        self.encoder = self.model.vit.encoder
        self.layernorm = self.model.vit.layernorm
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=num_classes, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
    
    def forward(self, x):
        n = x.shape[0]
        patched = self.patch(x)
        batched_cls_token = self.cls_token.expand(n, -1, -1)
        patched = torch.cat([batched_cls_token, patched], dim=1)
        patched = patched + self.pos_embedding(x)
        patched = self.encoder(patched)
        patched = self.layernorm(patched.last_hidden_state)
        patched = patched[:, 0]
        patched = self.classifier(patched)
        return patched

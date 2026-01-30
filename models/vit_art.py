"""
ViT with Art PE (LOOPE)
Vision Transformer with learnable content-aware positional encoding.
"""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

from models.positional_encodings import get_zero_pos_embed, ArtPE


class ViT_Art(nn.Module):
    """ViT model with Art (LOOPE) positional encoding."""

    def __init__(self, num_classes=1000, image_size=224):
        super().__init__()
        
        # Configure for scratch training
        config = ViTConfig(
            image_size=image_size,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=num_classes,
            
            # DeiT-style: usually keep dropout ~0 and use drop-path + strong aug + mixup/cutmix
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )

        # stochastic depth (drop-path)
        if hasattr(config, "drop_path_rate"):
            config.drop_path_rate = 0.1
        
        # Initialize model from config (scratch)
        self.model = ViTForImageClassification(config)
        
        # Get dimensions
        _, self.seq_length, self.hidden_dim = self.model.vit.embeddings.position_embeddings.shape
        
        # Replace position embeddings with zeros (we'll add our own ArtPE)
        self.model.vit.embeddings.position_embeddings = get_zero_pos_embed(self.hidden_dim, self.seq_length)
        
        # Initialize Art PE
        self.pos_embedding = ArtPE(self.seq_length, self.hidden_dim, img_size=image_size)
    
        # Map components for forward pass
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

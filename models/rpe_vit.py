"""
Relative Positional Encoding ViT (RPE-ViT)
ViT with Swin-style relative position bias.
"""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification

from models.positional_encodings import get_pe_caller


class GlobalRelativePositionBias(nn.Module):
    """Global relative position bias table (Swin-style)."""
    
    def __init__(self, grid_size, num_heads):
        super().__init__()
        self.grid_size = grid_size
        self.num_heads = num_heads

        # Define a learnable relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * grid_size - 1) * (2 * grid_size - 1), num_heads)
        )  # Shape: ((2H-1) * (2W-1), nH)

        # Initialize with truncated normal
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position indices
        self.define_relative_position_index()

    def define_relative_position_index(self):
        """Compute pairwise relative position indices for the full grid."""
        coords_h = torch.arange(self.grid_size)
        coords_w = torch.arange(self.grid_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, H, W)

        coords_flatten = torch.flatten(coords, 1)  # (2, H*W)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, H*W, H*W)

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (H*W, H*W, 2)
        relative_coords[:, :, 0] += self.grid_size - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.grid_size - 1

        relative_coords[:, :, 0] *= 2 * self.grid_size - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # (H*W * H*W)

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        """Retrieve relative position bias."""
        return self.relative_position_bias_table[self.relative_position_index].view(
            self.grid_size * self.grid_size, self.grid_size * self.grid_size, self.num_heads
        ).permute(2, 0, 1)  # (nH, H*W, H*W)


class RifatAttention(nn.Module):
    """Custom attention with relative position encoding."""
    
    def __init__(self, pretrained_query, pretrained_key, pretrained_value, grid_size, num_heads):
        super().__init__()
        self.query_ = nn.Parameter(pretrained_query.weight)
        self.key_ = nn.Parameter(pretrained_key.weight)
        self.value_ = nn.Parameter(pretrained_value.weight)

        self.relative_position_bias_module = GlobalRelativePositionBias(grid_size, num_heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        q = torch.matmul(hidden_states, self.query_.T)
        k = torch.matmul(hidden_states, self.key_.T)
        v = torch.matmul(hidden_states, self.value_.T)

        print(f'query shape:{q.shape}')
        print(f'key shape:{k.shape}')

        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # Standard attention
        print(f'attention score shape:{attn_scores.shape}')

        # Add relative positional bias (broadcasted across batches)
        print(f' RPE shape: {self.relative_position_bias_module().shape}')
        print(attn_scores[:, 1:, 1:].shape)
        attn_scores[:, 1:, 1:] += self.relative_position_bias_module()
        print(f'attention score after RPE shape:{attn_scores.shape}')

        # Apply mask if available
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply head mask if given
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        x = torch.matmul(attn_probs, v)

        if output_attentions:
            return x, attn_probs
        return x


def create_rpe_vit(num_classes, encoder_type='sin_cos_1d'):
    """
    Create ViT with Relative Position Encoding.
    
    Args:
        num_classes: Number of output classes
        encoder_type: Type of base positional encoding
    
    Returns:
        Configured ViTForImageClassification model
    """
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    _, seq_length, hidden_dims = model.vit.embeddings.position_embeddings.shape
    
    pe_caller = get_pe_caller(encoder_type)
    model.vit.embeddings.position_embeddings = pe_caller(hidden_dims, seq_length)
    model.classifier = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    
    # Compute grid size dynamically
    grid_size = model.config.image_size // model.config.patch_size  # 224//16 = 14
    
    # Note: The original notebook modifies attention layers but doesn't complete the implementation
    # The modified attention can be applied here if needed:
    # for i in range(12):
    #     for name, module in model.vit.encoder.layer[i].named_modules():
    #         if 'attention.attention' in name:
    #             query = model.vit.encoder.layer[i].attention.attention.query
    #             key = model.vit.encoder.layer[i].attention.attention.key
    #             value = model.vit.encoder.layer[i].attention.attention.value
    #             module = RifatAttention(query, key, value, grid_size, 1)
    
    return model

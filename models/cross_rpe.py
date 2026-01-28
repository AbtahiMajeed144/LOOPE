"""
Cross-RPE (Experimental)
Cross-Method Relative Positional Encoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from transformers.models.vit.modeling_vit import ViTSelfAttention


class CrossMethodRPE(nn.Module):
    """Cross-method relative positional encoding."""
    
    def __init__(self, image_size, embed_dim, alpha=1.0, beta=10.0, gamma=100.0):
        super(CrossMethodRPE, self).__init__()

        self.image_size = image_size  # (Height, Width)
        self.height, self.width = image_size
        self.embed_dim = embed_dim

        # Learnable scalars for horizontal and vertical directions
        self.px = nn.Parameter(torch.randn(embed_dim, 1, 1))  # Horizontal
        self.py = nn.Parameter(torch.randn(embed_dim, 1, 1))  # Vertical

        # Parameters for piecewise function g(x)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def g(self, dist):
        """Piecewise function for distance mapping."""
        alpha_tensor = torch.tensor(self.alpha, device=dist.device)
        gamma_tensor = torch.tensor(self.gamma, device=dist.device)

        dist = dist.abs()

        # Case 1: |x| ≤ α
        less_than_alpha = dist <= alpha_tensor
        result = torch.round(dist) * less_than_alpha.float()

        # Case 2: |x| > α
        greater_than_alpha = dist > alpha_tensor
        sign_dist = torch.sign(dist)

        log_scaled = alpha_tensor + (torch.log(dist / alpha_tensor) / torch.log(gamma_tensor / alpha_tensor)) * (self.beta - alpha_tensor)
        result += sign_dist * torch.min(self.beta * torch.ones_like(log_scaled), log_scaled) * greater_than_alpha.float()

        return result

    def forward(self):
        """Computes relative positional encoding b_ij."""
        grid_x, grid_y = torch.meshgrid(torch.arange(self.height), torch.arange(self.width), indexing="ij")
        grid_x = grid_x.to(torch.float32).unsqueeze(0).unsqueeze(0)
        grid_y = grid_y.to(torch.float32).unsqueeze(0).unsqueeze(0)

        Ix = self.g(grid_x - grid_x.transpose(-1, -2)) * self.px
        Iy = self.g(grid_y - grid_y.transpose(-1, -2)) * self.py

        encoding = Ix + Iy  # Final RPE b_ij
        return encoding


class ViTSelfAttentionWithRPE(ViTSelfAttention):
    """ViT Self-Attention with Cross-RPE."""
    
    def __init__(self, config, rpe_module):
        super().__init__(config)
        self.rpe_module = rpe_module  # Inject RPE module

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_heads = self.num_attention_heads
        head_dim = hidden_dim // num_heads

        # Project queries, keys, values
        query_layer = self.query(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_layer = self.key(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_layer = self.value(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (head_dim ** 0.5)

        # Add relative positional encoding b_ij
        rpe = self.rpe_module().to(hidden_states.device)  # Compute b_ij
        rpe = rpe.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, height, width]
        attention_scores = attention_scores + rpe

        # Softmax and apply attention
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Compute context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

        # Return values in the expected format
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


def create_cross_rpe_vit():
    """
    Create ViT with Cross-Method RPE.
    
    Returns:
        Modified ViTModel
    """
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    
    # Extract model configuration
    hidden_size = model.config.hidden_size  # dz
    image_size = (14, 14)  # ViT input size after patch embedding (224/16 = 14)
    rpe_module = CrossMethodRPE(image_size=image_size, embed_dim=hidden_size)
    
    # Replace the self-attention module in all transformer blocks
    for layer in model.encoder.layer:
        layer.attention.attention = ViTSelfAttentionWithRPE(model.config, rpe_module)
    
    print("Modified ViT with Cross Method RPE is ready!")
    return model

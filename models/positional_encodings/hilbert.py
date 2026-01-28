"""
Hilbert Curve Positional Encoding
Uses Hilbert/Gilbert space-filling curve for 2D position ordering.
"""
import torch
import torch.nn as nn
import numpy as np


def gilbert_2d(width, height):
    """
    Generate Gilbert 2D space-filling curve indices.
    
    Args:
        width: Grid width
        height: Grid height
    
    Returns:
        numpy array with curve indices
    """
    def sgn(x):
        return 1 if x > 0 else -1 if x < 0 else 0
    
    def generate(x, y, ax, ay, bx, by):
        w = abs(ax + ay)
        h = abs(bx + by)
        dax, day = sgn(ax), sgn(ay)
        dbx, dby = sgn(bx), sgn(by)

        if h == 1:
            return [(x + dax*i, y + day*i) for i in range(w)]
        if w == 1:
            return [(x + dbx*i, y + dby*i) for i in range(h)]

        ax2, ay2 = ax//2, ay//2
        bx2, by2 = bx//2, by//2
        w2, h2 = abs(ax2 + ay2), abs(bx2 + by2)

        if 2*w > 3*h:
            if w2 % 2 and w > 2:
                ax2 += dax
                ay2 += day
            return generate(x, y, ax2, ay2, bx, by) + \
                   generate(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)
        else:
            if h2 % 2 and h > 2:
                bx2 += dbx
                by2 += dby
            return generate(x, y, bx2, by2, ax2, ay2) + \
                   generate(x+bx2, y+by2, ax, ay, bx-bx2, by-by2) + \
                   generate(x+(ax-dax)+(bx2-dbx), 
                            y+(ay-day)+(by2-dby), 
                            -bx2, -by2, -ax+ax2, -ay+ay2)

    # Generate coordinates and create index array
    curve = generate(0, 0, width, 0, 0, height)
    arr = np.zeros((height, width), dtype=int)
    for idx, (x, y) in enumerate(curve):
        arr[y, x] = idx + 1  # Note y is first dimension in numpy arrays
    
    return arr


def generate_hilbert_grid(N):
    """
    Generate standard Hilbert curve grid.
    
    Args:
        N: Grid size (must be power of 2)
    
    Returns:
        2D list with Hilbert indices
    """
    def rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x, y = n - 1 - x, n - 1 - y
            x, y = y, x
        return x, y

    def hilbert_index(n, d):
        x = y = 0
        t = d
        s = 1
        while s < n:
            rx = (t // 2) & 1
            ry = (t ^ rx) & 1
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    grid = [[-1 for _ in range(N)] for _ in range(N)]
    for d in range(N * N):
        x, y = hilbert_index(N, d)
        grid[y][x] = d
    return grid


def get_hilbert_pos_embed(embed_dim, seq_length):
    """
    Create Hilbert curve based positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        seq_length: Sequence length (including CLS token)
    
    Returns:
        nn.Parameter with requires_grad=False
    """
    N = int(np.sqrt(seq_length - 1))
    grid = gilbert_2d(N, N)
    grid = torch.Tensor(grid).reshape(seq_length - 1)
    position = torch.cat((torch.tensor([0]), grid)).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(np.log(10000.0) / embed_dim))
    
    pos_embed = torch.zeros((seq_length, embed_dim), dtype=torch.float32)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)
    
    return nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)

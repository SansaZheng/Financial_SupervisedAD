# Vendored unmodified from kathiresan-jayabalan/sttn-cp-ccfd (Apache-2.0).
# Upstream commit e5d3657d826ca2e42d6b2489e0577afc45f773df.
# See third_party/sttn_cp/LICENSE and NOTICE.
"""STTN-CP model components

The implementation follows a sequential spatial -> residual -> temporal -> residual flow.
Spatial attention receives the original transaction features at each timestep.
Temporal attention operates on the embedded transaction sequence.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for a token sequence."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        if d_model < 2 or d_model % 2 != 0:
            raise ValueError("d_model must be an even integer >= 2")
        if max_len < 1:
            raise ValueError("max_len must be positive")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"sequence length {x.size(1)} exceeds positional encoding "
                f"capacity {self.pe.size(1)}"
            )
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


class SpatialTransformerBlock(nn.Module):
    """Attend across the original transaction features at each timestep."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        n_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")

        self.input_dim = input_dim
        self.feature_embed = nn.Linear(1, embed_dim)
        self.feature_pos = PositionalEncoding(embed_dim, max_len=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("spatial input must have shape (B, T, D)")
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} input features, got {x.size(-1)}"
            )

        batch, steps, features = x.shape
        tokens = self.feature_embed(x.reshape(batch * steps, features, 1))
        tokens = self.feature_pos(tokens)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        return self.norm(pooled).reshape(batch, steps, -1)


class TemporalTransformerBlock(nn.Module):
    """Attend across transaction timesteps in a window."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        mlp_dim: int,
        seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")

        self.positional = PositionalEncoding(embed_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("temporal input must have shape (B, T, E)")
        h = self.positional(x)
        h = self.transformer(h)
        return self.norm(h)


class STBlock(nn.Module):
    """One spatial-temporal block: spatial -> residual -> temporal -> residual."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        n_heads: int,
        mlp_dim: int,
        seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.spatial = SpatialTransformerBlock(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.temporal = TemporalTransformerBlock(
            embed_dim=embed_dim,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            seq_len=seq_len,
            dropout=dropout,
        )

    def forward(self, raw_x: Tensor, h: Tensor) -> Tensor:
        spatial = self.spatial(raw_x)
        h = h + spatial
        temporal = self.temporal(h)
        return h + temporal


class STTNCP(nn.Module):
    """Spatial-Temporal Transformer Network with Contrastive Pretraining."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        embed_dim: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        mlp_dim: int = 256,
        proj_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")

        self.config: Dict[str, int | float] = {
            "input_dim": input_dim,
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "n_blocks": n_blocks,
            "n_heads": n_heads,
            "mlp_dim": mlp_dim,
            "proj_dim": proj_dim,
            "num_classes": num_classes,
            "dropout": dropout,
        }

        self.input_embed = nn.Linear(input_dim, embed_dim)
        self.temporal_positional = PositionalEncoding(embed_dim, max_len=seq_len)
        self.st_blocks = nn.ModuleList(
            [
                STBlock(
                    input_dim=input_dim,
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    mlp_dim=mlp_dim,
                    seq_len=seq_len,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward_backbone(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("model input must have shape (B, T, D)")
        if x.size(1) != self.config["seq_len"]:
            raise ValueError(
                f"expected sequence length {self.config['seq_len']}, got {x.size(1)}"
            )
        if x.size(2) != self.config["input_dim"]:
            raise ValueError(
                f"expected {self.config['input_dim']} features, got {x.size(2)}"
            )

        h = self.input_embed(x)
        h = self.temporal_positional(h)
        for block in self.st_blocks:
            h = block(x, h)
        return h.mean(dim=1)

    def project(self, z: Tensor) -> Tensor:
        return self.projector(z)

    def forward(self, x: Tensor) -> Tensor:
        """Return legitimate/fraud logits for a complete transaction window."""
        z = self.forward_backbone(x)
        return self.classifier(z)


def infonce_loss(
    z1: Tensor,
    z2: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric InfoNCE loss using cosine-normalized representations."""
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("contrastive representations must be 2-D")
    if z1.shape != z2.shape:
        raise ValueError("contrastive representations must have matching shapes")
    if z1.size(0) < 2:
        raise ValueError("InfoNCE requires a batch of at least two samples")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    logits = reps @ reps.T / temperature

    batch = z1.size(0)
    mask = torch.eye(2 * batch, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, float("-inf"))
    targets = torch.arange(2 * batch, device=logits.device)
    targets = (targets + batch) % (2 * batch)
    return F.cross_entropy(logits, targets)


def combined_loss(
    logits: Tensor,
    labels: Tensor,
    z1: Tensor,
    z2: Tensor,
    class_loss: nn.Module,
    lambda_contrastive: float = 0.5,
    temperature: float = 0.07,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return the combined objective: classification + lambda * contrastive."""
    loss_class = class_loss(logits, labels)
    loss_contrastive = infonce_loss(z1, z2, temperature=temperature)
    loss_total = loss_class + lambda_contrastive * loss_contrastive
    return loss_total, loss_class, loss_contrastive


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

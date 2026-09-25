"""The network: a locally connected layer of 64 units (each sees one 5x5 block of the image; the corners of the
blocks form an 8x8 grid over the 28x28 image), a dense layer of 96 units and K outputs, softplus activations,
float64.  The parameters are handled as one flat vector theta (order: W1, b1, fc.weight, fc.bias, out.weight,
out.bias); d = 64 * 26 + 96 * 65 + 97 * K."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMG, PATCH, GRID = 28, 5, 8          # image side, block side, blocks per side
UNITS, HIDDEN = GRID * GRID, 96      # 64 locally connected units, 96 dense units


def patch_indices() -> np.ndarray:
    """(64, 25) pixel indices of the 5x5 blocks."""
    corners = np.round(np.linspace(0, IMG - PATCH, GRID)).astype(int)
    idx = []
    for r0 in corners:
        for c0 in corners:
            idx.append([(r0 + r) * IMG + (c0 + c) for r in range(PATCH) for c in range(PATCH)])
    return np.asarray(idx, dtype=np.int64)


class PatchNet(torch.nn.Module):
    def __init__(self, K: int):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.zeros(UNITS, PATCH * PATCH, dtype=torch.float64))
        self.b1 = torch.nn.Parameter(torch.zeros(UNITS, dtype=torch.float64))
        self.fc = torch.nn.Linear(UNITS, HIDDEN, dtype=torch.float64)
        self.out = torch.nn.Linear(HIDDEN, K, dtype=torch.float64)
        self.register_buffer("pidx", torch.from_numpy(patch_indices()))

    def forward(self, x):                       # x: (B, 784)
        patches = x[:, self.pidx]               # (B, 64, 25)
        z = torch.einsum("bup,up->bu", patches, self.W1) + self.b1
        z = F.softplus(z)
        z = F.softplus(self.fc(z))
        return self.out(z)                      # logits (B, K)


def initial_point(K: int, seed: int = 8) -> np.ndarray:
    """He initialization of the flat parameter vector (hidden biases 0.01, output biases 0)."""
    rng = np.random.RandomState(seed)
    parts = [rng.randn(UNITS, PATCH * PATCH).ravel() * np.sqrt(2.0 / (PATCH * PATCH)),
             np.full(UNITS, 0.01),
             rng.randn(HIDDEN, UNITS).ravel() * np.sqrt(2.0 / UNITS),
             np.full(HIDDEN, 0.01),
             rng.randn(K, HIDDEN).ravel() * np.sqrt(2.0 / HIDDEN),
             np.zeros(K)]
    return np.concatenate(parts)


def load_theta(net: torch.nn.Module, theta: np.ndarray) -> None:
    """Copy the flat vector theta into the network parameters."""
    t = torch.from_numpy(theta)
    offset = 0
    for param in net.parameters():
        n_params = param.numel()
        param.data.copy_(t[offset: offset + n_params].view_as(param))
        offset += n_params
    assert offset == t.numel(), f"theta has length {t.numel()}, the network {offset} parameters"


def flatten_grads(grads) -> np.ndarray:
    return torch.cat([g.reshape(-1) for g in grads]).detach().cpu().numpy()

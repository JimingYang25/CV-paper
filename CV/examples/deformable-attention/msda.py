"""Readable multi-scale deformable attention for learning and CPU verification.

Independent teaching implementation of the equations in Deformable DETR.
No custom CUDA extension, padding/valid-ratio handling, or reference boxes.
Inputs must use a common unpadded image frame. See the accompanying lesson.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sample_and_aggregate(values: list[Tensor], locations: Tensor,
                         weights: Tensor) -> Tensor:
    """Aggregate already projected feature maps.

    values[l]: [B, C, H_l, W_l]; C = M * D
    locations: [B, Q, M, L, K, 2], normalized (x, y), not clamped
    weights:   [B, Q, M, L, K], already normalized across L*K
    returns:   [B, Q, C], before the output projection
    """
    b, q, m, levels, points, xy = locations.shape
    if xy != 2 or len(values) != levels:
        raise ValueError("locations must end in 2 and match the feature levels")
    c = values[0].shape[1]
    if c % m:
        raise ValueError("C must be divisible by the number of heads")
    d = c // m
    if weights.shape != locations.shape[:-1]:
        raise ValueError("weights and locations disagree")
    # Keep the level reduction explicit instead of materializing all samples.
    result = values[0].new_zeros(b, m, d, q)
    for level, value in enumerate(values):
        if value.ndim != 4 or value.shape[:2] != (b, c):
            raise ValueError("each feature level must have shape [B,C,H,W]")
        h, w = value.shape[-2:]
        per_head = value.reshape(b, m, d, h, w).reshape(b * m, d, h, w)
        # grid_sample regards Q and K as output spatial axes; neither is an
        # image coordinate here. Their last dimension holds actual (x,y).
        grid = locations[:, :, :, level].permute(0, 2, 1, 3, 4)
        grid = (2 * grid - 1).reshape(b * m, q, points, 2)
        sampled = F.grid_sample(per_head, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        sampled = sampled.reshape(b, m, d, q, points)
        alpha = weights[:, :, :, level].permute(0, 2, 1, 3).unsqueeze(2)
        result = result + (sampled * alpha).sum(dim=-1)
    return result.permute(0, 3, 1, 2).reshape(b, q, c)


class MultiScaleDeformableAttention(nn.Module):
    """Two-dimensional point references, with no padding.

    query: [B,Q,C]
    reference: [B,Q,2] in normalized image coordinates (x,y)
    features: L tensors [B,C,H_l,W_l]
    """

    def __init__(self, channels: int = 256, heads: int = 8,
                 levels: int = 4, points: int = 4):
        super().__init__()
        if min(channels, heads, levels, points) < 1 or channels % heads:
            raise ValueError("positive sizes and channels % heads == 0 required")
        self.channels, self.heads = channels, heads
        self.levels, self.points = levels, points
        self.offsets = nn.Linear(channels, heads * levels * points * 2)
        self.logits = nn.Linear(channels, heads * levels * points)
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Linear(channels, channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        # Spread initial samples over directions and radii. The precise
        # normalization follows the convention used in the official module.
        angle = torch.arange(self.heads) * (2 * math.pi / self.heads)
        direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
        direction = direction / direction.abs().amax(dim=-1, keepdim=True)
        radius = torch.arange(1, self.points + 1)
        initial = direction[:, None, None, :] * radius[None, None, :, None]
        initial = initial.expand(self.heads, self.levels, self.points, 2)
        with torch.no_grad():
            self.offsets.bias.copy_(initial.reshape(-1))
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query: Tensor, reference: Tensor,
                features: list[Tensor], return_details: bool = False):
        b, q, c = query.shape
        if c != self.channels or reference.shape != (b, q, 2):
            raise ValueError("query/reference shapes disagree with module")
        if len(features) != self.levels:
            raise ValueError("wrong number of feature levels")
        if any(f.ndim != 4 or f.shape[:2] != (b, c) for f in features):
            raise ValueError("features must share batch and channel sizes")
        offsets = self.offsets(query).reshape(
            b, q, self.heads, self.levels, self.points, 2)
        alpha = self.logits(query).reshape(
            b, q, self.heads, self.levels * self.points).softmax(dim=-1)
        alpha = alpha.reshape(b, q, self.heads, self.levels, self.points)
        wh = query.new_tensor([(f.shape[-1], f.shape[-2]) for f in features])
        locations = reference[:, :, None, None, None, :] + offsets / wh[
            None, None, None, :, None, :]
        projected = [self.value(f) for f in features]
        result = self.output(sample_and_aggregate(projected, locations, alpha))
        if return_details:
            return result, {"locations": locations, "weights": alpha,
                            "offsets": offsets}
        return result

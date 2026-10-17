"""Independent NumPy reference for RAFT's correlation/upsampling mathematics.

This is not a pretrained RAFT estimator. It implements feature correlation,
target-only pooling, fractional lookup, the equivalent on-demand computation,
and convex flow upsampling. Coordinates are feature-pixel centers in (x,y).
Lookup channels use level-major, dy-major, dx-minor ordering; do not load
upstream learned weights without matching their channel ordering explicitly.
"""

import numpy as np


def coordinate_grid(batch, height, width, dtype=np.float64):
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.broadcast_to(np.stack((x, y))[None],
                           (batch, 2, height, width)).astype(dtype).copy()


def all_pairs(feature1, feature2):
    """[B,D,H1,W1], [B,D,H2,W2] -> [B,H1,W1,H2,W2]."""
    if feature1.ndim != 4 or feature2.ndim != 4:
        raise ValueError("features must be rank four")
    b, d, h1, w1 = feature1.shape
    if feature2.shape[:2] != (b, d) or d == 0:
        raise ValueError("features must share batch and channel sizes")
    h2, w2 = feature2.shape[-2:]
    left = feature1.reshape(b, d, h1*w1).transpose(0, 2, 1)
    right = feature2.reshape(b, d, h2*w2)
    return (left @ right / np.sqrt(d)).reshape(b, h1, w1, h2, w2)


def pool_last_two(array):
    """Non-overlapping 2x2 average pooling, flooring odd target dimensions."""
    h, w = array.shape[-2:]
    if min(h, w) < 2:
        raise ValueError("not enough target pixels for another pooling level")
    h2, w2 = h//2, w//2
    cropped = array[..., :2*h2, :2*w2]
    return cropped.reshape(*array.shape[:-2], h2, 2, w2, 2).mean(axis=(-3,-1))


def correlation_pyramid(feature1, feature2, levels=4):
    if levels < 1:
        raise ValueError("levels must be positive")
    pyramid = [all_pairs(feature1, feature2)]
    for _ in range(levels-1):
        pyramid.append(pool_last_two(pyramid[-1]))
    return pyramid


def bilinear_sample(maps, points):
    """maps [A,H,W], points [A,P,2] -> [A,P], zero outside support.

    Pixel-center coordinates avoid a normalized-grid convention internally.
    For non-degenerate maps this is equivalent to an align_corners=True grid
    with gx=2*x/(W-1)-1 and gy=2*y/(H-1)-1, with zero padding.
    """
    if maps.ndim != 3 or points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("expected maps [A,H,W] and points [A,P,2]")
    a, h, w = maps.shape
    if points.shape[0] != a or not np.isfinite(points).all():
        raise ValueError("batch mismatch or nonfinite coordinates")
    x, y = points[..., 0], points[..., 1]
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    dx, dy = x-x0, y-y0
    out = np.zeros(points.shape[:2], dtype=np.result_type(maps, points))
    for ox, oy, weight in ((0,0,(1-dx)*(1-dy)), (1,0,dx*(1-dy)),
                            (0,1,(1-dx)*dy), (1,1,dx*dy)):
        xx, yy = x0+ox, y0+oy
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        value = maps[np.arange(a)[:, None], np.clip(yy,0,h-1), np.clip(xx,0,w-1)]
        out += np.where(valid, value*weight, 0)
    return out


def offset_grid(radius):
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    dy, dx = np.meshgrid(np.arange(-radius,radius+1),
                         np.arange(-radius,radius+1), indexing="ij")
    return np.stack((dx,dy),axis=-1).reshape(-1,2)


def lookup(pyramid, coordinates, radius=4):
    """Target coordinates [B,2,H1,W1] -> [B,L*(2r+1)^2,H1,W1]."""
    b, h1, w1 = pyramid[0].shape[:3]
    if coordinates.shape != (b,2,h1,w1):
        raise ValueError("coordinate/source-grid shape mismatch")
    centers = coordinates.transpose(0,2,3,1).reshape(b*h1*w1,1,2)
    delta = offset_grid(radius)
    outputs = []
    for level, corr in enumerate(pyramid):
        h2,w2 = corr.shape[-2:]
        points = centers/(2**level) + delta[None]
        values = bilinear_sample(corr.reshape(b*h1*w1,h2,w2),points)
        outputs.append(values.reshape(b,h1,w1,-1).transpose(0,3,1,2))
    return np.concatenate(outputs,axis=1)


def lookup_on_demand(feature1, feature2, coordinates, levels=4, radius=4):
    """Pool target FEATURES, sample them, then dot with each source feature.

    Computes the same responses as lookup(correlation_pyramid(...)) without
    constructing the quadratic all-pairs array. This vectorized NumPy version
    still materializes sampled-feature intermediates; it is not a fused kernel.
    """
    b,d,h1,w1 = feature1.shape
    if feature2.shape[:2] != (b,d) or coordinates.shape != (b,2,h1,w1):
        raise ValueError("feature or coordinate shape mismatch")
    if levels < 1:
        raise ValueError("levels must be positive")
    n = h1*w1
    delta = offset_grid(radius)
    k = len(delta)
    centers = coordinates.transpose(0,2,3,1).reshape(b,n,1,2)
    source = feature1.reshape(b,d,n)
    target = feature2
    responses = []
    for level in range(levels):
        h2,w2 = target.shape[-2:]
        points = centers/(2**level)+delta[None,None]
        points = np.broadcast_to(points[:,None],(b,d,n,k,2)).reshape(b*d,n*k,2)
        sampled = bilinear_sample(target.reshape(b*d,h2,w2),points).reshape(b,d,n,k)
        response = np.einsum("bdn,bdnk->bnk",source,sampled)/np.sqrt(d)
        responses.append(response.reshape(b,h1,w1,k).transpose(0,3,1,2))
        if level+1 < levels:
            target = pool_last_two(target)
    return np.concatenate(responses,axis=1)


def argmax_flow(correlation):
    """Unregularized integer matching baseline, not RAFT's learned update."""
    b,h1,w1,h2,w2 = correlation.shape
    index = correlation.reshape(b,h1,w1,h2*w2).argmax(axis=-1)
    target = np.stack((index % w2,index//w2),axis=1)
    return target-coordinate_grid(b,h1,w1)


def convex_upsample(flow, weights, scale=8):
    """flow [B,2,H,W], weights [B,9,s,s,H,W] -> [B,2,sH,sW].

    weights must be nonnegative and sum to 1 over the nine 3x3 neighbors.
    Boundary neighbors are zero-padded, matching the official unfold path.
    Multiplication by scale converts feature-pixel to input-pixel units.
    """
    b,c,h,w = flow.shape
    if c != 2 or scale < 1 or weights.shape != (b,9,scale,scale,h,w):
        raise ValueError("invalid flow, scale, or weights shape")
    if (weights < 0).any() or not np.allclose(weights.sum(axis=1),1):
        raise ValueError("weights must be convex over the neighbor axis")
    padded = np.pad(flow,((0,0),(0,0),(1,1),(1,1)))
    neighbors = np.stack([padded[:,:,dy:dy+h,dx:dx+w]
                          for dy in range(3) for dx in range(3)],axis=2)
    mixed = (scale*neighbors[:,:,:,None,None]*weights[:,None]).sum(axis=2)
    return mixed.transpose(0,1,4,2,5,3).reshape(b,2,h*scale,w*scale)

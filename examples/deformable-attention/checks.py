"""Mathematical checks against an independent scalar interpolation oracle."""

import json
import platform
import unittest

import torch

from msda import MultiScaleDeformableAttention, sample_and_aggregate


def scalar_oracle(values, locations, weights):
    """Slow forward-only oracle; deliberately no grid_sample or reshapes."""
    b, q, m, levels, points, _ = locations.shape
    c = values[0].shape[1]
    d = c // m
    result = torch.zeros(b, q, c, dtype=values[0].dtype)
    for bi in range(b):
        for qi in range(q):
            for mi in range(m):
                for li in range(levels):
                    h, w = values[li].shape[-2:]
                    for ki in range(points):
                        u, v = locations[bi, qi, mi, li, ki].tolist()
                        x, y = u * w - 0.5, v * h - 0.5
                        ix, iy = int(x // 1), int(y // 1)
                        for yy in (iy, iy + 1):
                            for xx in (ix, ix + 1):
                                if 0 <= xx < w and 0 <= yy < h:
                                    a = (1 - abs(x - xx)) * (1 - abs(y - yy))
                                    for di in range(d):
                                        ci = mi * d + di
                                        result[bi, qi, ci] += (
                                            weights[bi, qi, mi, li, ki]
                                            * a * values[li][bi, ci, yy, xx])
    return result


class SamplingChecks(unittest.TestCase):
    def test_known_bilinear_value(self):
        value = torch.tensor([[[[1., 3.], [5., 7.]]]], dtype=torch.float64)
        loc = torch.tensor([0.375, 0.625], dtype=torch.float64).reshape(1,1,1,1,1,2)
        alpha = torch.ones(1,1,1,1,1, dtype=torch.float64)
        actual = sample_and_aggregate([value], loc, alpha)
        self.assertAlmostEqual(actual.item(), 4.5, places=12)

    def test_multiscale_non_square_batch_head_oracle(self):
        torch.manual_seed(17)
        values = [torch.randn(2, 6, 3, 5, dtype=torch.float64),
                  torch.randn(2, 6, 2, 4, dtype=torch.float64)]
        # Includes near-boundary and outside-map samples.
        loc = torch.rand(2, 3, 2, 2, 3, 2, dtype=torch.float64) * 1.5 - 0.25
        weights = torch.randn(2,3,2,6, dtype=torch.float64).softmax(-1)
        weights = weights.reshape(2,3,2,2,3)
        torch.testing.assert_close(sample_and_aggregate(values, loc, weights),
                                   scalar_oracle(values, loc, weights),
                                   rtol=1e-11, atol=1e-11)

    def test_zero_padding_boundary_is_not_clamping(self):
        value = torch.ones(1,1,2,2, dtype=torch.float64)
        loc = torch.tensor([[0.,0.],[-1.,-1.]], dtype=torch.float64)
        loc = loc.reshape(1,2,1,1,1,2)
        output = sample_and_aggregate([value], loc, torch.ones(1,2,1,1,1))
        torch.testing.assert_close(output.flatten(), torch.tensor([0.25,0.],
                                                               dtype=torch.float64))

    def test_normalization_and_xy_offset_units(self):
        model = MultiScaleDeformableAttention(8, 2, 2, 2).double()
        query = torch.zeros(1,3,8, dtype=torch.float64)
        reference = torch.full((1,3,2), 0.5, dtype=torch.float64)
        feats = [torch.randn(1,8,4,10, dtype=torch.float64),
                 torch.randn(1,8,2,5, dtype=torch.float64)]
        with torch.no_grad():
            model.offsets.bias.copy_(torch.tensor([2.,1.],dtype=torch.float64)
                                     .repeat(model.offsets.bias.numel() // 2))
        _, details = model(query, reference, feats, return_details=True)
        torch.testing.assert_close(details['weights'].sum(dim=(-1,-2)),
                                   torch.ones(1,3,2, dtype=torch.float64))
        torch.testing.assert_close(details['locations'][0,0,0,0,0],
                                   torch.tensor([0.7,0.75], dtype=torch.float64))
        torch.testing.assert_close(details['locations'][0,0,0,1,0],
                                   torch.tensor([0.9,1.0], dtype=torch.float64))

    def test_gradients_of_values_locations_and_logits(self):
        torch.manual_seed(3)
        value = torch.randn(1,2,3,4, dtype=torch.float64, requires_grad=True)
        # Fractional positions away from pixel-center knots.
        loc = torch.tensor([0.37,0.43,0.62,0.71],dtype=torch.float64)
        loc = loc.reshape(1,1,1,1,2,2).requires_grad_()
        logits = torch.randn(1,1,1,1,2,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda val, pos, log: sample_and_aggregate([val], pos, log.softmax(-1)),
            (value, loc, logits), eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_full_module_backpropagation(self):
        torch.manual_seed(19)
        model = MultiScaleDeformableAttention(8, 2, 2, 2).double()
        query = torch.randn(2,3,8,dtype=torch.float64,requires_grad=True)
        ref = torch.full((2,3,2),0.5,dtype=torch.float64,requires_grad=True)
        feats = [torch.randn(2,8,6,8,dtype=torch.float64,requires_grad=True),
                 torch.randn(2,8,3,4,dtype=torch.float64,requires_grad=True)]
        out = model(query,ref,feats)
        self.assertEqual(out.shape,(2,3,8))
        out.square().mean().backward()
        for tensor in [query,ref,*feats,*model.parameters()]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(model.offsets.weight.grad.abs().sum().item(),0.)
        self.assertGreater(model.logits.weight.grad.abs().sum().item(),0.)


if __name__ == '__main__':
    print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
                      'device':'cpu','dtype':'float64'}))
    unittest.main(verbosity=2)

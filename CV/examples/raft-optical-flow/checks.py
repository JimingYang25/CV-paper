"""Independent numerical checks for correlation and coordinate geometry."""

import unittest

import numpy as np

from correlation import (all_pairs, pool_last_two, bilinear_sample, lookup,
                         correlation_pyramid, lookup_on_demand, coordinate_grid,
                         convex_upsample)
from demo import run


class NumericalChecks(unittest.TestCase):
    def test_all_pairs_against_scalar_dot_products(self):
        rng = np.random.default_rng(1)
        f1,f2 = rng.normal(size=(2,3,3,5)),rng.normal(size=(2,3,4,7))
        c = all_pairs(f1,f2)
        oracle = np.zeros((2,3,5,4,7))
        for b,y1,x1,y2,x2 in np.ndindex(oracle.shape):
            oracle[b,y1,x1,y2,x2] = sum(f1[b,d,y1,x1]*f2[b,d,y2,x2]
                                       for d in range(3))/np.sqrt(3)
        np.testing.assert_allclose(c,oracle,atol=1e-12,rtol=1e-12)

    def test_pooling_preserves_source_indices_and_floors_odd_sizes(self):
        data = np.arange(2*3*5*7,dtype=float).reshape(2,3,5,7)
        pooled = pool_last_two(data)
        self.assertEqual(pooled.shape,(2,3,2,3))
        for a,b,y,x in np.ndindex(pooled.shape):
            self.assertEqual(pooled[a,b,y,x],data[a,b,2*y:2*y+2,2*x:2*x+2].mean())

    def test_fractional_sample_and_zero_padding(self):
        maps = np.array([[[1.,3.],[5.,7.]]])
        points = np.array([[[.25,.75],[-.5,-.5],[-2.,0.]]])
        np.testing.assert_allclose(bilinear_sample(maps,points),[[4.5,.25,0.]])

    def test_bilinear_coordinate_jacobian_by_finite_difference(self):
        maps = np.array([[[1.,3.],[5.,7.]]])
        p = np.array([[[.25,.75]]])
        numerical = []
        for axis in range(2):
            step = np.zeros_like(p); step[...,axis] = 1e-6
            numerical.append(((bilinear_sample(maps,p+step)-
                                bilinear_sample(maps,p-step))/(2e-6)).item())
        np.testing.assert_allclose(numerical,[2.,4.],atol=1e-8,rtol=0)

    def test_lookup_xy_order_and_level_coordinate_scaling(self):
        # Every source point sees a target map V(y,x)=x+2y+source_id.
        yy,xx = np.meshgrid(np.arange(8),np.arange(12),indexing='ij')
        corr = np.stack([xx+2*yy+i for i in range(6)]).reshape(1,2,3,8,12)
        coords = np.zeros((1,2,2,3)); coords[:,0] = 4.25; coords[:,1] = 3.5
        values = lookup([corr,pool_last_two(corr)],coords,radius=1)
        self.assertEqual(values.shape,(1,18,2,3))
        for y,x in np.ndindex(2,3):
            source_id = y*3+x
            # The center channel of each 3x3 local lookup has offset (0,0).
            self.assertAlmostEqual(values[0,4,y,x],4.25+2*3.5+source_id)
            # Pooling V gives 2*x_level+4*y_level+1.5+source_id.
            self.assertAlmostEqual(values[0,13,y,x],4.25+7.+1.5+source_id)

    def test_materialized_matches_on_demand_with_odd_maps_and_boundaries(self):
        rng = np.random.default_rng(3)
        f1,f2 = rng.normal(size=(2,5,3,4)),rng.normal(size=(2,5,9,13))
        coords = rng.uniform([-1.,-1.],[14.,10.],size=(2,3,4,2)).transpose(0,3,1,2)
        expected = lookup(correlation_pyramid(f1,f2,3),coords,radius=2)
        actual = lookup_on_demand(f1,f2,coords,3,radius=2)
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

    def test_convex_upsampling_units_layout_and_boundary(self):
        flow = coordinate_grid(1,3,4)
        weights = np.zeros((1,9,2,2,3,4)); weights[:,4] = 1.
        up = convex_upsample(flow,weights,scale=2)
        expected = np.repeat(np.repeat(flow,2,axis=2),2,axis=3)*2
        np.testing.assert_allclose(up,expected)
        ones = np.ones((1,2,3,4)); weights.fill(1/9)
        up = convex_upsample(ones,weights,scale=2)
        np.testing.assert_allclose(up[:,:,2:4,2:4],2.)
        np.testing.assert_allclose(up[:,:,0,0],2*4/9)

    def test_translation_demo_flow_sign_and_valid_mask(self):
        report,*_ = run()
        self.assertEqual(report['valid_pixels'],176)
        self.assertEqual(report['valid_mean_epe_feature_pixels'],0.)
        self.assertEqual(report['local_peak_displacement_xy'],[2,-1])


if __name__ == '__main__':
    unittest.main(verbosity=2)

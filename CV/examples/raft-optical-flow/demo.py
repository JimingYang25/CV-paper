"""Ideal-descriptor translation experiment; no neural network or weights."""

import json
import numpy as np

from correlation import (argmax_flow, coordinate_grid,
                         correlation_pyramid, lookup, lookup_on_demand, offset_grid)


def translated_features(seed=18, height=12, width=18, channels=32, dx=2, dy=-1):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=(1,channels,height,width))
    f2 = rng.normal(size=f1.shape)
    f1 /= np.linalg.norm(f1,axis=1,keepdims=True)
    f2 /= np.linalg.norm(f2,axis=1,keepdims=True)
    ys,ye = max(0,-dy),min(height,height-dy)
    xs,xe = max(0,-dx),min(width,width-dx)
    f2[:,:,ys+dy:ye+dy,xs+dx:xe+dx] = f1[:,:,ys:ye,xs:xe]
    valid = np.zeros((height,width),dtype=bool)
    valid[ys:ye,xs:xe] = True
    return f1,f2,valid,np.array([dx,dy],dtype=np.float64)


def run():
    f1,f2,valid,truth = translated_features()
    pyramid = correlation_pyramid(f1,f2,levels=3)
    centers = coordinate_grid(1,*f1.shape[-2:])
    stored = lookup(pyramid,centers,radius=2)
    ondemand = lookup_on_demand(f1,f2,centers,levels=3,radius=2)
    predicted = argmax_flow(pyramid[0])
    error = np.linalg.norm(predicted[0]-truth[:,None,None],axis=0)
    point_y,point_x = 6,7
    local_delta = offset_grid(2)[stored[0,:25,point_y,point_x].argmax()]
    report = {
        "seed":18,"feature_shape":list(f1.shape),"dtype":str(f1.dtype),
        "true_flow_feature_pixels":truth.tolist(),"valid_pixels":int(valid.sum()),
        "valid_mean_epe_feature_pixels":float(error[valid].mean()),
        "valid_max_epe_feature_pixels":float(error[valid].max()),
        "local_peak_displacement_xy":local_delta.tolist(),
        "lookup_shape":list(stored.shape),
        "stored_vs_on_demand_max_abs_error":float(np.abs(stored-ondemand).max()),
        "correlation_pyramid_bytes":sum(p.nbytes for p in pyramid),
        "scope":"Ideal synthetic descriptors; no trained RAFT or real-image benchmark"
    }
    assert report["valid_max_epe_feature_pixels"] == 0
    np.testing.assert_allclose(local_delta,truth)
    np.testing.assert_allclose(stored,ondemand,atol=1e-12,rtol=1e-12)
    return report,pyramid[0][0,point_y,point_x],valid,predicted


if __name__ == "__main__":
    report,*_ = run()
    print(json.dumps(report,indent=2))

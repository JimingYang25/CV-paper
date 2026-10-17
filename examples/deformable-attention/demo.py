"""A tiny end-to-end differentiable spatial retrieval experiment.

Learn a global sampling displacement by maximizing a known Gaussian's value.
This isolates offset learning; it is not detector training or a benchmark.
"""

import json

import torch

from msda import MultiScaleDeformableAttention


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    h, w = 24, 32
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    center = torch.tensor([17., 10.])
    signal = torch.exp(-((xx-center[0])**2+(yy-center[1])**2)/(2*3.0**2))
    features = [signal.reshape(1,1,h,w)]
    reference = ((center - torch.tensor([2.,-1.]) + 0.5)
                 / torch.tensor([w,h])).reshape(1,1,2)
    model = MultiScaleDeformableAttention(1,1,1,1)
    # Freeze value/output transforms to identity so they cannot amplify the
    # answer. Only the offset bias moves. With K=L=1, alpha is necessarily 1.
    with torch.no_grad():
        model.value.weight.fill_(1.)
        model.value.bias.zero_()
        model.output.weight.fill_(1.)
        model.output.bias.zero_()
        model.offsets.bias.zero_()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.offsets.bias.requires_grad_(True)
    optimizer = torch.optim.Adam([model.offsets.bias],lr=0.08)
    query = torch.zeros(1,1,1)
    initial = model(query,reference,features).item()
    for _ in range(160):
        optimizer.zero_grad()
        response = model(query,reference,features)
        (-response.mean()).backward()
        optimizer.step()
    final, detail = model(query,reference,features,return_details=True)
    point = detail['locations'].reshape(2).detach()*torch.tensor([w,h])-0.5
    report = {'seed':7,'steps':160,'device':'cpu',
              'initial_response':initial,'final_response':final.item(),
              'target_xy':center.tolist(),'learned_sample_xy':point.tolist(),
              'learned_offset_xy':model.offsets.bias.detach().tolist(),
              'error_pixels':torch.linalg.vector_norm(point-center).item(),
              'scope':'synthetic offset optimization; no detection accuracy measured'}
    assert report['final_response'] > report['initial_response']
    assert report['error_pixels'] < 0.2
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

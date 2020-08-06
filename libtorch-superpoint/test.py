import torch
import torchvision
model=torchvision.models.resnet18()
s=torch.randn(1,3,224,224)
ts=torch.jit.trace(model,s)
ts.save('test.pt')
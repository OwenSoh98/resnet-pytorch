
import torch
t = torch.tensor([[[1, 2],[3, 4]], [[5, 6],[7, 8]]])

print(t.shape)
print(t[:,1,:])
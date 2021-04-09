import torch
from leaf_torch.frontend import Leaf

x = torch.ones((1, 16000))
y = Leaf()(x)
print(y)

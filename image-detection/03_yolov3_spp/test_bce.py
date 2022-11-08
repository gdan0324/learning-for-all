import torch
import numpy as np

loss = torch.nn.BCELoss(reduction='none')   # 默认是进行均值处理，二值交叉熵处理之后就不进行处理
po = [[0.1, 0.8, 0.9], [0.2, 0.7, 0.8]]
p = torch.tensor(po, requires_grad=True)

go = [[0., 0., 1.], [0., 0., 1.]]
g = torch.tensor(go)

l = loss(input=p, target=g)
print(np.round(l.detach().numpy(), 5))


def bce(c, o):
    return np.round(-(o * np.log(c) + (1 - o) * np.log(1 - c)), 5)


pn = np.array(po)
gn = np.array(go)

print(bce(pn, gn))

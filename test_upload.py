import torch
#
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
# labels = torch.Tensor([[0,1,1],[0,0,1],[1,0,1]])
# print(labels)
# print(torch.matmul(labels, labels.T).bool().float())

# aa = np.arange(10)
# print(aa)
#
# res = np.where(aa == 5)
# print(type(res), res[0])

# y_pred = np.array([[0.1, 0.1, 0.9], [0.6,0.1, 0.3], [0.2,0.3,0.6]])
# y_true = np.array([2,0,1])
#
# average_precision = metrics.accuracy_score(
#             y_true, np.argmax(y_pred, axis=1))
#
# print("average_precision: ", average_precision)

# random_state = np.random.RandomState(1234)
# mixup_alpha = 1.
# batch_size = 64
# mixup_lambdas = []
# for n in range(0, batch_size, 2):
#     lam = random_state.beta(mixup_alpha, mixup_alpha, 1)[0]
#     mixup_lambdas.append(lam)
#     mixup_lambdas.append(1. - lam)
#
# print(mixup_lambdas)

a = torch.Tensor([0,1,2])

print(F.normalize(a, dim=0))
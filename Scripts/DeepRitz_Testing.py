from Utils.Algorithms import *
from Utils.ANN_Models import *
import json


torch.set_default_tensor_type(torch.DoubleTensor)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {dev}')

activation = nn.GELU()


def phi(x):
    return torch.sum(x[:, ::2] * x[:, 1::2], dim=1)
    # return torch.sum(x.square(), dim=1)


d_i = 10
width = 16
depth = 8
beta = 500.
f = 0.

residual = True

ann = HeatRitz_ANN(d_i, width, depth, phi, dev, activation,
                   beta=beta, res=residual, f_term=f).to(dev)

n_test = 100000
n_test_reset = 8192

train_steps = 80000
eval_steps = train_steps // 200
lr = .001
bs = 4096

tolerance = 400
lr_factor = 4
search_list = [50]
k_list = [5]

results = test_adaptive_adam(ann, train_steps, eval_steps, bs, n_test, n_test_reset,
                             search_list, k_list, tolerance, lr_factor, init_lr=lr)

json_path = 'ritz_test.json'
with open(json_path, 'w') as f:
    json.dump(results, f)

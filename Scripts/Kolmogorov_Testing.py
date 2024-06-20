from Utils.Algorithms import *
from Utils.ANN_Models import *
import json


torch.set_default_tensor_type(torch.DoubleTensor)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {dev}')

activation = nn.GELU()

d_i = 25
neurons = [d_i, 50, 100, 50, 1]

T = 2.
rho = 1.


def phi(x):  # initial value of solution of heat PDE
    return x.square().sum(axis=1, keepdim=True)


def u_T(x):  # value of heat PDE solution at final time T
    return x.square().sum(axis=1, keepdim=True) + 2. * rho * T * d_i


space_bounds = [-1, 1]

n_test = 100000

train_steps = 80000
eval_steps = train_steps // 200
lr = .001
bs = 4096

tolerance = 400
lr_factor = 4
search_list = [50]
k_list = [5]

ann = Heat_PDE_ANN(neurons, phi, space_bounds, T, rho, dev,
                   activation, lr, u_T).to(dev)

results = test_adaptive_adam(ann, train_steps, eval_steps, bs, n_test, n_test,
                             search_list, k_list, tolerance, lr_factor, init_lr=lr)

json_path = 'heat_test.json'
with open(json_path, 'w') as f:
    json.dump(results, f)

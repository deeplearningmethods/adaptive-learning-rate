from Utils.Algorithms import *
from Utils.ANN_Models import *
import json


torch.set_default_tensor_type(torch.DoubleTensor)

activation = nn.GELU()

neurons = [3, 32, 64, 32, 1]

space_bounds = [2., 1.]


def init_func(x):
    return 1.5 * np.prod(np.sin(np.pi * x), axis=0) ** 2
    # return np.prod(np.sin(np.pi * x), axis=0)


def sine_torch(x):
    return torch.sin(x)


def sine_nonlin(x):
    return np.sin(x)


def allen_cahn_nonlin(x):
    return x - x ** 3


T = 1.
alpha = 0.05
ann = SemilinHeat_PINN_2d(neurons, init_func, sine_nonlin, alpha, space_bounds,
                          T, activation=activation, torch_nonlin=sine_torch)
# to test Allen-Cahn equation: use allen_cahn_nonlin here
n_test = 2000

train_steps = 100000
eval_steps = train_steps // 100
lr = 0.001
bs = 256

tolerance = 400
lr_factor = 4
search_list = [30, 50]
k_list = [3, 5]

results = test_adaptive_adam(ann, train_steps, eval_steps, bs, n_test, n_test,
                             search_list, k_list, tolerance, lr_factor)

json_path = 'pinn_sine_test.json'
with open(json_path, 'w') as f:
    json.dump(results, f)

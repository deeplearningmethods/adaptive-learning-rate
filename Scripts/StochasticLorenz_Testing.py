from Utils.Algorithms import *
from Utils.ANN_Models import *
import json


torch.set_default_tensor_type(torch.DoubleTensor)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {dev}')

activation = nn.GELU()

d_i = 3
neurons = [d_i, 32, 64, 32, 1]

T = 1.
alphas = [10., 14., 8. / 3.]
beta = 0.15
space_bounds = [[0.5, 1.5], [8., 10.], [10., 12.]]
nt = 100


def phi(x):  # initial value of solution of PDE
    return x.square().sum(axis=-1, keepdim=True)


n_test_reset = 4096
n_test = 4096
mc_samples = 4000
mc_rounds = 200

train_steps = 100000
eval_steps = train_steps // 100
lr = .001
bs = 512

tolerance = 1000
lr_factor = 3
search_list = [50]
k_list = [5]

ann = StochasticLorenz_ANN(neurons, phi, space_bounds, alphas, beta, T, nt, dev,
                           activation, lr, mc_samples=mc_samples,
                           mc_rounds=mc_rounds, test_size=n_test).to(dev)

results = test_adaptive_adam(ann, train_steps, eval_steps, bs, n_test,
                             n_test_reset, search_list, k_list, tolerance, lr_factor, init_lr=lr)

json_path = 'lorenz_test.json'
with open(json_path, 'w') as f:
    json.dump(results, f)

from Utils.Algorithms import *
from Utils.ANN_Models import *
import json


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {dev}')
torch.set_default_tensor_type(torch.DoubleTensor)

activation = nn.GELU()

d_i = 10
neurons = [d_i, 50, 100, 50, 1]

T = 1.
r, c, K = 0.05, 0.1, 100.
sigma = torch.linspace(start=1. / d_i, end=.5, steps=d_i)


def phi(x):  # initial value of solution of heat PDE
    return np.exp(-r * T) * torch.maximum(torch.max(x, dim=-1,
                            keepdim=True)[0] - K, torch.tensor(0.))


space_bounds = [90, 110]

n_test_reset = 4096
n_test = 4096
mc_samples = 1024
mc_rounds = 400

train_steps = 80000
eval_steps = train_steps // 100
lr = .001
bs = 2048

tolerance = 500
lr_factor = 3
search_list = [50]
k_list = [5]

ann = BlackScholes_ANN(neurons, phi, space_bounds, T, c, r, sigma, dev,
                       activation, lr, mc_samples=mc_samples,
                       mc_rounds=mc_rounds, test_size=n_test)

results = test_adaptive_adam(ann, train_steps, eval_steps, bs, n_test,
            n_test_reset, search_list, k_list, tolerance, lr_factor, init_lr=lr)

json_path = 'black_scholes_test.json'
with open(json_path, 'w') as f:
    json.dump(results, f)

from Utils.Algorithms import *
from Utils.ANN_Models import *
import json


torch.set_default_tensor_type(torch.DoubleTensor)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {dev}')

activation = nn.ReLU()

neurons = [6, 128, 1]
d_i = neurons[0]

A = torch.reshape(torch.arange(1. - d_i, d_i + 1., 2.), [d_i, 1])


def f(x):  # target function
    return torch.matmul(x ** 2, A) + 1.


ann = Supervised_ANN(neurons, f, [-1., 1.], dev, activation=activation)

n_test = 2000
n_test_reset = 10000

train_steps = 20000
eval_steps = train_steps // 200
lr = 0.001
bs = 256

tolerance = 400
lr_factor = 4
search_list = [50]
k_list = [5]

results = test_adaptive_adam(ann, train_steps, eval_steps, bs, n_test,
                             n_test_reset, search_list, k_list, tolerance, lr_factor, init_lr=lr)

json_path = 'supervised_test.json'
with open(json_path, 'w') as f:
    json.dump(results, f)
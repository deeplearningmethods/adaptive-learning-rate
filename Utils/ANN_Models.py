import torch
from torch import nn
import math
from Utils.PDEs_ScikitFEM import *
from Utils.Initializers import *


class Supervised_ANN(nn.Module):
    """
    Approximates deterministic target function f by neural network.
    neurons: List/Tuple specifying the layer dimensions.
    """
    def __init__(self, neurons, f, space_bounds, dev, activation, lr=0.001):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))

        self.initializer = UniformValueSampler(neurons[0], space_bounds, dev)
        self.target_fn = f

        self.done_steps = 0

        self.losses = []
        self.lr_list = []
        self.lr = lr

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, x):
        y = self.forward(x)
        l = (y - self.target_fn(x)).square().mean()
        return l

    def test_loss(self, x):
        y = self.forward(x)
        l = (y - self.target_fn(x)).square().mean().sqrt()
        return l


class Heat_PDE_ANN(nn.Module):
    """
    Implements deep Kolmogorov method for standard heat PDE.
    neurons: List/Tuple specifying the layer dimensions of the considered ANN.
    phi: Function representing the initial value.
    """
    def __init__(self, neurons, phi, space_bounds, T, rho, dev,
                 activation=nn.ReLU(), lr=0.0001, final_u=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))

        self.initializer = UniformValueSampler(neurons[0], space_bounds, dev)
        self.phi = phi
        self.rho = rho
        self.T = T
        self.space_bounds = space_bounds
        self.final_u = final_u

        self.done_steps = 0

        self.losses = []
        self.lr_list = []
        self.lr = lr

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        W = torch.randn_like(data)
        return (self.phi(math.sqrt(2 * self.rho * self.T) * W + data)
                - self.forward(data)).square().mean()

    def test_loss(self, data):
        output = self.forward(data)
        u_T = self.final_u(data)
        return ((u_T - output) / u_T).square().mean().sqrt()


class BlackScholes_ANN(nn.Module):
    """
    Implements deep Kolmogorov method for Black-Scholes PDE.
    """
    def __init__(self, neurons, phi, space_bounds, T, c, r, sigma, dev,
                 activation=nn.ReLU(), lr=0.0001, final_u=None, mc_samples=1024,
                 test_size=10000, mc_rounds=100):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        self.depth = len(neurons) - 1
        self.layers.append(nn.BatchNorm1d(neurons[0]))
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))

        self.initializer = UniformValueSampler(neurons[0], space_bounds, dev)
        self.phi = phi
        self.x_test = self.initializer.sample(test_size)

        self.mu = r - c
        self.r = r
        self.sigma = sigma.to(dev)
        self.mc_samples = mc_samples
        self.T = T
        self.space_bounds = space_bounds
        self.final_u = final_u

        self.done_steps = 0

        self.losses = []
        self.lr_list = []
        self.lr = lr

        if self.final_u is None:  # approximate true solution by Monte Carlo
            u_ref = torch.zeros([test_size, neurons[-1]]).to(dev)
            for i in range(mc_rounds):
                x = torch.stack([self.x_test for _ in range(self.mc_samples)])
                w = torch.randn_like(x)
                u = self.phi(self.x_test *
                             torch.exp((self.mu - 0.5 * self.sigma ** 2)
                                * self.T + self.sigma * math.sqrt(self.T) * w))
                u = torch.mean(u, dim=0)
                u_ref += u
            self.u_test = u_ref / mc_rounds
        else:
            self.u_test = self.final_u(self.x_test)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        W = torch.randn_like(data)
        X = data * torch.exp((self.mu - 0.5 * self.sigma ** 2) * self.T +
                             self.sigma * math.sqrt(self.T) * W)
        return (self.phi(X) - self.forward(data)).square().mean()

    def test_loss(self, data):
        output = self.forward(self.x_test)

        return ((self.u_test - output) / self.u_test).square().mean().sqrt()


class StochasticLorenz_ANN(nn.Module):
    """Specific example of deep Kolmogorov method (Lorenz-type PDE),
     taken from section 4.5 of Beck et al. paper."""
    def __init__(self, neurons, phi, space_bounds, alphas, beta, T, nr_timesteps,
                 dev, activation=nn.ReLU(), lr=0.0001, mc_samples=1024,
                 test_size=10000, mc_rounds=100):
        super().__init__()
        assert neurons[0] == 3

        self.layers = nn.ModuleList()
        self.dims = neurons
        self.depth = len(neurons) - 1
        self.layers.append(nn.BatchNorm1d(neurons[0]))
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))

        self.alphas = alphas
        self.beta = beta
        self.nt = nr_timesteps
        self.T = T
        self.space_bounds = torch.Tensor(space_bounds)
        self.dt = self.T / self.nt

        self.mc_samples = mc_samples
        self.lr = lr
        self.phi = phi

        self.initializer = UniformValueSamplerGeneral(3, self.space_bounds, dev)
        self.x_test = self.initializer.sample(test_size)

        u_ref = torch.zeros([test_size, neurons[0]]).to(dev)
        for i in range(mc_rounds): # approximate true solution by Monte Carlo
            x = torch.stack([self.x_test for _ in range(self.mc_samples)])
            for _ in range(self.nt):
                w = math.sqrt(self.dt) * torch.stack([torch.randn_like(self.x_test) for _ in range(self.mc_samples)])
                mu_norm = torch.sqrt(torch.sum(self.mu(x) ** 2, dim=-1, keepdim=True))
                x += self.mu(x) * self.dt * torch.where(mu_norm <= 1. / self.dt, 1., 0.)
                x += self.beta * w
            u = torch.mean(self.phi(x), dim=0)
            u_ref += u
        self.u_test = u_ref / mc_rounds

    def mu(self, x):
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        x3 = x[:, :, 2]

        return torch.stack([self.alphas[0] * (x2 - x1),
                            self.alphas[1] * x1 - x2 - x1 * x3,
                            x1 * x2 - self.alphas[2] * x3], dim=-1)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        x = torch.unsqueeze(data, dim=0)
        for _ in range(self.nt):
            w = math.sqrt(self.dt) * torch.randn_like(data)
            mu_norm = torch.sqrt(torch.sum(self.mu(x) ** 2, dim=-1, keepdim=True))
            x += self.mu(x) * self.dt * torch.where(mu_norm <= 1. / self.dt, 1., 0.)
            x += self.beta * w
        return (self.phi(x) - self.forward(data)).square().mean()

    def test_loss(self, data):
        output = self.forward(self.x_test)

        return ((self.u_test - output) / self.u_test).square().mean().sqrt()


class SemilinHeat_PINN_2d(nn.Module):
    """ Neural network to solve semilinear heat equation
     du/dt = alpha * Laplace(u) + nonlin(u) on rectangle [0, a] x [0, b]
     with either Dirichlet or periodic boundary conditions using PINN method."""
    def __init__(self, neurons, f, nonlin, alpha, space_bounds, T=1.,
                 test_discr=100, test_timesteps=500, activation=nn.Tanh(),
                 nonlin_name=None, torch_nonlin=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dims = neurons
        assert neurons[0] == 3
        self.depth = len(neurons) - 1
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(neurons[-2], neurons[-1]))
        self.f = f
        self.T = T
        self.alpha = alpha
        self.space_bounds = space_bounds
        self.spacetime_bounds = space_bounds + [T]

        self.nonlin = nonlin
        self.nonlin_name = nonlin_name
        if torch_nonlin is None:
            self.torch_nonlin = nonlin
        else:
            self.torch_nonlin = torch_nonlin

        self.base = FEM_Basis(space_bounds, test_discr)
        self.ref_method = ReferenceMethod(Second_order_linear_implicit_RK_FEM,
                                          self.alpha, self.nonlin, self.nonlin_name, (0.5, 0.5), 'LIRK2')
        self.pde = self.ref_method.create_ode(self.base)
        self.init_values = self.base.project_cont_function(f)
        self.final_sol = self.ref_method.compute_sol(T, self.init_values, test_timesteps, self.pde, self.base)
        self.u_t_fem = self.base.basis.interpolator(self.final_sol)

        self.initializer = RectangleValueSampler(3, self.spacetime_bounds)

    def forward(self, x):
        for fc in self.layers:
            x = fc(x)
        return x

    def loss(self, data):
        x = torch.Tensor(data[:, 0:2])
        t = torch.Tensor(data[:, 2:3])

        f0 = torch.Tensor(np.transpose(self.f(np.transpose(data[:, 0:2]))))
        x.requires_grad_()
        t.requires_grad_()

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        x0 = torch.cat((x, torch.zeros_like(t)), 1)
        u0 = torch.squeeze(self.forward(x0))

        initial_loss = (u0 - f0).square().mean()

        u = self.forward(torch.cat((x1, x2, t), 1))
        u_x1 = torch.autograd.grad(u, x1, torch.ones_like(u),
                                   create_graph=True)[0]
        u_xx1 = torch.autograd.grad(u_x1, x1, torch.ones_like(u_x1),
                                    create_graph=True)[0]
        u_x2 = torch.autograd.grad(u, x2, torch.ones_like(u),
                                   create_graph=True)[0]
        u_xx2 = torch.autograd.grad(u_x2, x2, torch.ones_like(u_x2),
                                    create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u),
                                  create_graph=True)[0]

        loss = (self.alpha * (u_xx1 + u_xx2) +
                self.torch_nonlin(u) - u_t).square().mean()

        xa0 = torch.cat((torch.zeros_like(x1), x2, t), 1)
        xa1 = torch.cat((self.space_bounds[0] * torch.ones_like(x1), x2, t), 1)

        xb0 = torch.cat((x1, torch.zeros_like(x2), t), 1)
        xb1 = torch.cat((x1, self.space_bounds[1] * torch.ones_like(x2), t), 1)

        boundary_loss = (self.forward(xa0)).square().mean() \
                        + (self.forward(xa1)).square().mean() \
                        + (self.forward(xb0)).square().mean() \
                        + (self.forward(xb1)).square().mean()

        loss_value = loss + boundary_loss + 2. * initial_loss
        return loss_value

    def test_loss(self, x):
        x = x[:, 0:2]
        y_t_fem = torch.Tensor(self.u_t_fem(np.transpose(x)))
        x = torch.Tensor(x)
        x_t = torch.cat((x, self.T * torch.ones_like(x[:, 0:1])), 1)
        y_t_net = torch.squeeze(self.forward(x_t))
        l = (y_t_fem - y_t_net).square().mean()
        return l.sqrt()


class HeatRitz_ANN(nn.Module):
    """ Deep Ritz method (E & Yu) to solve heat PDE.
    d_i-dimensional input, ANN layers of dimension depth,
    option to add residual connections skipping every second layer. """
    def __init__(self, d_i, width, depth, phi, dev, activation=nn.ReLU(),
                 f_term=0., lr=0.0001, res=False, beta=1.):
        super().__init__()
        self.layers = nn.ModuleList()
        self.width = width
        self.d_i = d_i
        self.activation = activation
        self.depth = depth
        self.layers.append(nn.Linear(self.d_i, self.width))
        for i in range(self.depth - 1):
            self.layers.append(nn.Linear(self.width, self.width))
        self.layers.append(nn.Linear(self.width, 1))

        self.initializer = CubeSampler(d_i, dev)

        self.phi = phi

        self.done_steps = 0

        self.losses = []
        self.lr_list = []
        self.lr = lr
        self.res = res
        self.beta = beta
        self.f = f_term

    def forward(self, x):
        for fc in range(self.depth + 1):
            x_temp = self.layers[fc](x)
            if fc < self.depth - 1:
                x_temp = self.activation(x_temp)
                if self.res and fc > 0:
                    x_temp += x
            x = x_temp
        return x

    def loss(self, data):
        x_i = data[:, 0:self.d_i]
        x_b = data[:, self.d_i:2 * self.d_i]
        x_i.requires_grad_()

        y_i = self.forward(x_i)
        y_b = self.forward(x_b)

        dfdx = torch.autograd.grad(y_i, x_i, grad_outputs=torch.ones_like(y_i),
                                   create_graph=True)[0]

        ssum = 0.5 * torch.sum(dfdx.square(), dim=1) + self.f * y_i
        l_i = ssum.mean()
        l_b = (self.phi(x_b) - y_b.squeeze()).square().mean()

        return l_i + self.beta * l_b

    def test_loss(self, data):
        x = data[:, 0:self.d_i]
        y = self.forward(x).squeeze()
        y_true = self.phi(x)
        l2_err = (y - y_true).square().mean()
        ref = y_true.square().mean()
        return (l2_err / ref).sqrt()

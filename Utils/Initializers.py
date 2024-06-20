import torch
import numpy as np


def initial_values_sampler_uniform(batch_size, space_dim, space_bounds):
    # samples points uniformly in hypercube [a, b ] ^dim
    a, b = space_bounds
    return (b - a) * torch.rand([batch_size, space_dim]) + a


def initial_values_sampler_rectangular(batch_size, space_dim, space_bounds):
    # samples points uniformly in hyper-cuboid [0, a_1] x ... x [0, a_d]
    return space_bounds * np.random.uniform(0., 1., (batch_size, space_dim))


def sample_from_boundary(dim, bs):
    # samples points from boundary of hypercube [-1, 1]^d
    data = 2. * torch.rand(bs, dim) - 1.
    norm = torch.norm(data, float('inf'), dim=1, keepdim=True)
    if torch.min(norm) == 0:
        return sample_from_boundary(dim, bs)
    else:
        array = data / norm
        return array


class RectangleValueSampler:
    # class implementing uniform sampling from hyper-cuboid.
    def __init__(self, space_dim, space_bounds):
        self.space_dim = space_dim
        assert len(space_bounds) == space_dim
        self.space_bounds = space_bounds

    def sample(self, batch_size):
        values = initial_values_sampler_rectangular(batch_size, self.space_dim,
                                                    self.space_bounds)
        return values


class UniformValueSampler:
    # class implementing uniform sampling from hyper-cube.
    def __init__(self, space_dim, space_bounds, dev):
        self.space_dim = space_dim

        assert len(space_bounds) == 2
        self.space_bounds = space_bounds
        self.dev = dev

    def sample(self, batch_size):
        values = initial_values_sampler_uniform(batch_size, self.space_dim,
                                                self.space_bounds).to(self.dev)
        return values


class UniformValueSamplerGeneral:
    def __init__(self, space_dim, space_bounds, dev):
        self.space_dim = space_dim
        self.lower_bounds = space_bounds[:, 0]
        self.side_lengths = space_bounds[:, 1] - space_bounds[:, 0]

        assert len(self.lower_bounds) == space_dim
        assert len(self.side_lengths) == space_dim

        self.dev = dev

    def sample(self, batch_size):
        x = torch.rand([batch_size, self.space_dim])
        values = self.lower_bounds + self.side_lengths * x
        values = values.to(self.dev)
        return values


class CubeSampler:
    """ Returns concatenation of values inside and on boundary
     of hypercube (for deep Ritz method). """
    def __init__(self, dim, dev):
        self.dim = dim
        self.dev = dev

    def sample(self, bs):
        x = initial_values_sampler_uniform(bs, self.dim, [-1., 1.]).to(self.dev)
        y = sample_from_boundary(self.dim, bs).to(self.dev)
        return torch.cat([x, y], dim=1)
# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch

class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *args):
        self.saved_tensors = args

class LinearKernel:

    def __call__(self, ctx, x, y):
        dx = x[:, 1:, :] - x[:, :-1, :]
        dy = y[:, 1:, :] - y[:, :-1, :]
        ctx.save_for_backward(dx, dy)
        gram = torch.bmm(dx, dy.permute(0, 2, 1))
        return gram

    def grad_x(self, ctx, derivs):
        dx, dy = ctx.saved_tensors
        out = torch.empty((dx.shape[0], dx.shape[1] + 1, dy.shape[1]), dtype=torch.float64, device=derivs.device)
        out[:, 0, :] = 0
        out[:, 1:, :] = derivs
        out[:, :-1, :] -= derivs
        out = torch.bmm(out, dy)
        return out

    def grad_y(self, ctx, derivs):
        dx, dy = ctx.saved_tensors
        out = torch.empty((dx.shape[0], dx.shape[1], dy.shape[1] + 1), dtype=torch.float64, device=derivs.device)
        out[:, :, 0] = 0
        out[:, :, 1:] = derivs
        out[:, :, :-1] -= derivs
        out = torch.bmm(out.permute(0, 2, 1), dx)
        return out

class ScaledLinearKernel:
    def __init__(self, scale : float = 1.):
        self.linear_kernel = LinearKernel()
        self.scale = scale
        self._scale_sq = scale ** 2

    def __call__(self, ctx, x, y):
        return self.linear_kernel(ctx, x * self._scale_sq, y)

    def grad_x(self, ctx, derivs):
        return self.linear_kernel.grad_x(ctx, derivs) * self._scale_sq

    def grad_y(self, ctx, derivs):
        return self.linear_kernel.grad_y(ctx, derivs)

class RBFKernel:
    def __init__(self, sigma : float):
        self.sigma = sigma
        self._one_over_sigma = 1. / sigma
        self._scale = 2 * self._one_over_sigma

    def __call__(self, ctx, x, y):
        dist = torch.bmm(x * self._scale, y.permute(0, 2, 1))
        torch.pow(x, 2, out=x)
        torch.pow(y, 2, out=y)
        x2 = torch.sum(x, dim=2) * self._one_over_sigma
        y2 = torch.sum(y, dim=2) * self._one_over_sigma
        dist -= torch.reshape(x2, (x.shape[0], x.shape[1], 1)) + torch.reshape(y2, (x.shape[0], 1, y.shape[1]))
        torch.exp(dist, out=dist)
        buff = torch.empty_like(dist[:, :-1, :])
        torch.diff(dist, dim=1, out=buff)
        dist.resize_((dist.shape[0], dist.shape[1] - 1, dist.shape[2] - 1))
        torch.diff(buff, dim=2, out=dist)
        return dist

    def grad_x(self, ctx, derivs):
        pass

    def grad_y(self, ctx, derivs):
        pass

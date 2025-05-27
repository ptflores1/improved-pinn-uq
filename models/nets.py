import os
from typing import Type, Tuple, List, Union
import time
import dill
import numpy as np
import torch
from torch import nn
import pyro
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, MCMC
from pyro.infer.autoguide import AutoNormal
import pyro.distributions as dist
from neurodiffeq.generators import SamplerGenerator
from tqdm.auto import tqdm

from models.callbacks import BVICallback, HMCCallback
from models.nuts import NUTS

def save_hmc_samples(samples, solver_id):
    old_samples = {}
    if os.path.exists(f"checkpoints/running_hmc_samples_{solver_id}.dill"):
        with open(f"checkpoints/running_hmc_samples_{solver_id}.dill", "rb") as f:
            old_samples = dill.load(f)
    
    new_samples = {k: torch.cat([old_samples[k], v]) if k in old_samples else v for k, v in samples.items()}
    with open(f"checkpoints/running_hmc_samples_{solver_id}.dill", "wb") as f:
        dill.dump(new_samples, f)

def load_hmc_samples(solver_id):
    with open(f"checkpoints/running_hmc_samples_{solver_id}.dill", "rb") as f:
        return dill.load(f)

def clear_hmc_samples(solver_id):
    os.remove(f"checkpoints/running_hmc_samples_{solver_id}.dill")

class FCNN(nn.Module):
    def __init__(
            self, input_features: int, output_features: int, hidden_units: Tuple[int, ...],
            actv: Type[nn.Module] = nn.Tanh, id=None, device="cpu") -> None:
        super().__init__()
        self.hidden_units = hidden_units
        self.output_features = output_features
        self.id = id

        units = (input_features,) + hidden_units
        layers: List[nn.Module] = []
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1], device=device))
            layers.append(actv())
        self.feature_map = torch.nn.Sequential(*layers)
        self.output = nn.Linear(units[-1], output_features, device=device)

        # Needed to be able to save the neurodiffeq solver
        self.NN = []

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.TensorType):
        x = self.feature_map(x)
        x = self.output(x)
        return x


class BVINN(PyroModule):
    def __init__(
        self,
        input_features,
        output_features,
        hidden_units,
        prior_std: float,
        actv=nn.Tanh,
        pretrained_weights: Union[FCNN, None] = None,
        device="cpu"
    ):
        super().__init__()
        self.activation = actv
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_units = (hidden_units)
        self.prior_std = prior_std

        # Define the layer sizes and the PyroModule layer list
        self.layer_sizes = [input_features] + list(hidden_units) + [output_features]
        layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in range(1, len(self.layer_sizes))]

        self.sequential = PyroModule[nn.Sequential](*sum([[l, self.activation()] for l in layer_list[:-1]], []) + [layer_list[-1]])

        for layer_idx, layer in enumerate(layer_list):
            layer.weight = PyroSample(dist.Normal(0., prior_std * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_std).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))
        
        if pretrained_weights:
            self._use_pretrained_weigths_asd(pretrained_weights)

    def forward(self, x):
        mu = self.sequential(x)
        return mu

    def _use_pretrained_weigths_asd(self, pretrained_weights):
        det_linears = filter(lambda l: isinstance(l, nn.Linear), pretrained_weights.feature_map)
        bbb_linears = filter(lambda l: isinstance(l, nn.Linear), self.sequential)
        for det_layer, var_layer in zip(det_linears, bbb_linears):
            assert var_layer.weight.size() == det_layer.weight.size()
            assert var_layer.bias.size() == det_layer.bias.size()
            var_layer.weight = det_layer.weight
            var_layer.bias = det_layer.bias

        assert self.sequential[-1].weight.size() == pretrained_weights.output.weight.size()
        assert self.sequential[-1].bias.size() == pretrained_weights.output.bias.size()
        self.sequential[-1].weight = pretrained_weights.output.weight
        self.sequential[-1].bias = pretrained_weights.output.bias

class BVIDetSolver(PyroModule):
    def __init__(self, nets, conditions, train_generator, det_solution, diff_eqs, lr=1e-3, get_likelihood_std=None, output_variance=False):
        super().__init__()
        self.nets = PyroModule[torch.nn.ModuleList](nets)
        self.conditions = conditions
        self.train_generator = SamplerGenerator(train_generator)
        self.det_solution = det_solution
        self.lr = lr
        self.get_likelihood_std = get_likelihood_std
        self.output_variance = output_variance
        self.training = True
        self.auto_guide = AutoNormal(self)
        self.callbacks = [BVICallback()]
        self.diff_eqs = diff_eqs

    def _sample_residuals(self, samples, coords):
        n_samples = samples[0].size(0)
        n_funcs = len(samples)

        residuals = [[] for _ in range(n_funcs)]
        for si in range(n_samples):
            res = self.diff_eqs(*[samples[fi][si] for fi in range(n_funcs)], *coords)
            for fi in range(n_funcs):
                residuals[fi].append(res[fi].detach())
        return [torch.stack(r) for r in residuals]

    def forward(self, x):
        mus = [cond.enforce(net, *x) for net, cond in zip(self.nets, self.conditions)]
        
        targets = self.det_solution(*x) if self.training else [None]*len(self.conditions)
        targets = targets if isinstance(targets, list) else [targets]

        stds = self.get_likelihood_std(x)

        for i, (mu, y) in enumerate(zip(mus, targets)):
            obs = pyro.sample(f"obs_{i}", dist.Normal(mu, stds).to_event(2), obs=y)
        return mus

    def fit(self, max_epochs: int):
        optimizer = pyro.optim.Adam({"lr": self.lr})
        svi = SVI(self, self.auto_guide, optimizer, loss=Trace_ELBO())

        losses = []
        for epoch in tqdm(range(max_epochs)):
            batch = [v.reshape(-1, 1) for v in self.train_generator.get_examples()]

            loss = svi.step(batch)
            losses.append(loss)

            for cb in self.callbacks:
                cb(loss, epoch)

            if epoch % 1000 == 0:
                tqdm.write("Elbo loss: {}".format(loss))
        return losses
    
    def posterior_predictive(self, x, param_samples=None, num_samples=1000, to_numpy=False, include_residuals=False):
        self.training = False
        return_sites = tuple(f"obs_{i}" for i in range(len(self.conditions)))
        if param_samples:
            predictive = pyro.infer.Predictive(self, num_samples=num_samples, posterior_samples=param_samples, return_sites=return_sites)
        else:
            predictive = pyro.infer.Predictive(self, guide=self.auto_guide, num_samples=num_samples, return_sites=return_sites)

        coords = [torch.tensor(t).requires_grad_(include_residuals) for t in x]
        samples = predictive(coords)
        samples = [samples[obs] for obs in return_sites]

        if include_residuals:
            residuals = self._sample_residuals(samples, coords)

        self.training = True
        if include_residuals:
            if to_numpy:
                return [sample.detach().numpy() for sample in samples], [res.detach().numpy() for res in residuals]
            return samples, residuals
        if to_numpy:
            return [sample.detach().numpy() for sample in samples]
        return samples
    
    def sample_posterior(self, n_samples=100):
        with pyro.plate("samples", n_samples, dim=-1):
            samples = self.auto_guide(None)
        return samples
    

class HMCDetSolver(PyroModule):
    def __init__(self, nets, conditions, train_generator, det_solution, diff_eqs, get_likelihood_std=None, output_variance=False, det_nets=None, chains=1, step_size=1):
        super().__init__()
        self.nets = PyroModule[torch.nn.ModuleList](nets)
        self.conditions = conditions
        self.train_generator = SamplerGenerator(train_generator)
        self.det_solution = det_solution
        self.get_likelihood_std = get_likelihood_std
        self.output_variance = output_variance
        self.mcmc = None
        self.callbacks = [HMCCallback()]
        self._id = str(time.time()).split(".")[0] + str(id(self))
        self.det_nets = det_nets
        self.chains = chains
        self.step_size = step_size
        print("HMCDetSolver ID:", self._id)

    def _get_initial_params(self):
        if self.det_nets is None:
            return None
        initial_params = {}
        for n, net in enumerate(self.det_nets):
            for pk, pv in net.named_parameters():
                key = f"nets.{n}.{pk}"
                key = key.replace("feature_map", "sequential")
                key = key.replace("output", "sequential.4")
                initial_params[key] = pv.detach().clone()
        return initial_params

    def _sample_residuals(self, samples, coords):
        n_samples = samples[0].size(0)
        n_funcs = len(samples)

        residuals = [[] for _ in range(n_funcs)]
        for si in range(n_samples):
            res = self.diff_eqs(*[samples[fi][si] for fi in range(n_funcs)], *coords)
            for fi in range(n_funcs):
                residuals[fi].append(res[fi].detach())
        return [torch.stack(r) for r in residuals]

    def forward(self, xs, ys=None, stds=None):
        mus = [cond.enforce(net, *xs) for net, cond in zip(self.nets, self.conditions)]
        targets = ys if ys is not None else [None]*len(self.conditions)
        targets = targets if isinstance(targets, list) else [targets]

        for i, (mu, y) in enumerate(zip(mus, targets)):
            obs = pyro.sample(f"obs_{i}", dist.Normal(mu, stds).to_event(2), obs=y)
        return mus

    def fit(self, n_samples: int):
        x_train = [v.reshape(-1, 1) for v in self.train_generator.get_examples()]
        y_train = self.det_solution(*x_train)
        stds = self.get_likelihood_std(x_train)

        def hook(*args, **kwargs):
            for cb in self.callbacks:
                cb(*args, **kwargs)

        initial_params = self._get_initial_params()
        print("FIXED SAMPLES")
        nuts_kernel = NUTS(self, step_size=self.step_size)
        self.mcmc = MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=int(n_samples*.1), hook_fn=hook, num_chains=self.chains, initial_params=initial_params)
        self.mcmc.run(x_train, y_train, stds)
        samples = self.mcmc.get_samples()
        return samples

        # samples_per_iteration = 1000
        # iterations = n_samples // samples_per_iteration

        # nuts_kernel = NUTS(self)
        # self.mcmc = MCMC(nuts_kernel, num_samples=samples_per_iteration, warmup_steps=int(n_samples*.1), hook_fn=hook)
        # for _ in range(iterations):
        #     self.mcmc.run(x_train, y_train, stds)
        #     save_hmc_samples(self.mcmc.get_samples(), self._id)
        #     self.mcmc.sampler.initial_params = { k: v[-1].unsqueeze(0) for  k, v in self.mcmc.get_samples().items() }
        #     self.mcmc._samples = None
        #     self.mcmc.sampler.warmup_steps = 0
        # samples = load_hmc_samples(self._id)
        # clear_hmc_samples(self._id)
        # return samples
    
    def posterior_predictive(self, x, param_samples=None, to_numpy=False, include_residuals=False):
        return_sites = tuple(f"obs_{i}" for i in range(len(self.conditions)))
        x = [torch.tensor(t) for t in x]
        predictive = pyro.infer.Predictive(self, posterior_samples=param_samples, return_sites=return_sites)
        stds = self.get_likelihood_std(x)

        coords = [torch.tensor(t).requires_grad_(include_residuals) for t in x]
        samples = predictive(coords, stds=stds)
        samples = [samples[obs] for obs in return_sites]

        if include_residuals:
            residuals = self._sample_residuals(samples, coords)
            if to_numpy:
                return [sample.detach().numpy() for sample in samples], [res.detach().numpy() for res in residuals]
            return samples, residuals
        
        if to_numpy:
            return [sample.detach().numpy() for sample in samples]
        return samples
    
class BVISolver(PyroModule):
    def __init__(self, nets, system, conditions, train_generator, likelihood_std=1., lr=1e-3):
        super().__init__()
        self.nets = PyroModule[torch.nn.ModuleList](nets)
        self.system = system
        self.conditions = conditions
        self.train_generator = SamplerGenerator(train_generator)
        self.likelihood_std = likelihood_std
        self.lr = lr
        self.training = True
        self.auto_guide = AutoNormal(self)
        self.callbacks = [BVICallback()]

    def forward(self, x):
        functions = [cond.enforce(net, *x) for net, cond in zip(self.nets, self.conditions)]
        residuals = self.system(*functions, *x)

        for i, r in enumerate(residuals):
            obs = pyro.sample(f"obs_{i}", dist.Normal(r, self.likelihood_std).to_event(2), obs=torch.tensor(0))
        return functions

    def fit(self, max_epochs: int):
        optimizer = pyro.optim.Adam({"lr": self.lr})
        svi = SVI(self, self.auto_guide, optimizer, loss=Trace_ELBO())

        losses = []
        for epoch in tqdm(range(max_epochs)):
            batch = [v.reshape(-1, 1) for v in self.train_generator.get_examples()]

            loss = svi.step(batch)
            losses.append(loss)

            for cb in self.callbacks:
                cb(loss, epoch)

            if epoch % 1000 == 0:
                tqdm.write("Elbo loss: {}".format(loss))
        return losses
    
    def posterior_predictive(self, x, param_samples=None, num_samples=1000, to_numpy=False):
        self.training = False
        return_sites = tuple(f"obs_{i}" for i in range(len(self.conditions)))
        if param_samples:
            predictive = pyro.infer.Predictive(self, num_samples=num_samples, posterior_samples=param_samples, return_sites=return_sites)
        else:
            predictive = pyro.infer.Predictive(self, guide=self.auto_guide, num_samples=num_samples, return_sites=return_sites)
        samples = predictive([torch.tensor(t) for t in x])
        self.training = True
        if to_numpy:
            return [samples[obs].numpy() for obs in return_sites]
        return [samples[obs] for obs in return_sites]
    
    def sample_posterior(self, n_samples=100):
        with pyro.plate("samples", n_samples, dim=-1):
            samples = self.auto_guide(None)
        return samples

class HMCSolver(PyroModule):
    def __init__(self, nets, system, conditions, train_generator, likelihood_std=1.):
        super().__init__()
        self.nets = PyroModule[torch.nn.ModuleList](nets)
        self.system = system
        self.conditions = conditions
        self.train_generator = SamplerGenerator(train_generator)
        self.likelihood_std = likelihood_std
        self.mcmc = None
        self.callbacks = [HMCCallback()]
        self._id = str(time.time()).split(".")[0] + str(id(self))
        print("HMCSolver ID:", self._id)

    def forward(self, xs):
        functions = [cond.enforce(net, *xs) for net, cond in zip(self.nets, self.conditions)]
        residuals = self.system(*functions, *xs)

        for i, r in enumerate(residuals):
            obs = pyro.sample(f"obs_{i}", dist.Normal(r, self.likelihood_std).to_event(2), obs=torch.tensor(0))
        return functions

    def fit(self, n_samples: int):
        x_train = [v.reshape(-1, 1) for v in self.train_generator.get_examples()]

        def hook(*args, **kwargs):
            for cb in self.callbacks:
                cb(*args, **kwargs)

        samples_per_iteration = 1000
        iterations = n_samples // samples_per_iteration

        nuts_kernel = NUTS(self)
        self.mcmc = MCMC(nuts_kernel, num_samples=samples_per_iteration, warmup_steps=int(n_samples*.1), hook_fn=hook)
        for it in range(iterations):
            self.mcmc.run(x_train)
            save_hmc_samples(self.mcmc.get_samples(), self._id)
            self.mcmc.sampler.initial_params = { k: v[-1].unsqueeze(0) for  k, v in self.mcmc.get_samples().items() }
            self.mcmc._samples = None
            self.mcmc.sampler.warmup_steps = 0
        samples = load_hmc_samples(self._id)
        clear_hmc_samples(self._id)
        return samples
    
    def posterior_predictive(self, x, param_samples=None, to_numpy=False):
        return_sites = tuple(f"obs_{i}" for i in range(len(self.conditions)))
        x = [torch.tensor(t) for t in x]
        predictive = pyro.infer.Predictive(self, posterior_samples=param_samples, return_sites=return_sites)
        stds = self.get_likelihood_std(x)
        samples = predictive(x, stds=stds)
        if to_numpy:
            return [samples[obs].numpy() for obs in return_sites]
        return [samples[obs] for obs in return_sites]
    
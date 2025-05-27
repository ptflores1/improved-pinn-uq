from typing import Callable, Sequence, Tuple, Type, Union

import torch
from torch.distributions import Normal
from pyro import poutine

from neurodiffeq.solvers import BaseSolver


class BBBLoss:
    def __init__(
            self, det_solution: Callable, solver: Type[BaseSolver],
            get_error_tolerance: Callable[[Sequence[torch.TensorType]],
                                          torch.Tensor],
            n_samples=1, device="cpu") -> None:
        self.det_solution = det_solution
        self.solver = solver
        self.n_samples = n_samples
        self.get_error_tolerance = get_error_tolerance
        self.device = device
        self.train_history = {"loss": [], "log_varpost": [], "log_prior": [], "log_likelihood": []}
        self.valid_history = {"loss": [], "log_varpost": [], "log_prior": [], "log_likelihood": []}

    def __call__(self, residuals, funcs, coords, take_mean=True) -> Union[Tuple, torch.Tensor]:
        targets = self.det_solution(*coords)
        targets = [targets] if len(self.solver.nets) == 1 else targets

        log_varposts = torch.zeros(self.n_samples, len(self.solver.nets), device=self.device)
        log_priors = torch.zeros(self.n_samples, len(self.solver.nets), device=self.device)
        log_likes = torch.zeros(self.n_samples, len(self.solver.nets), device=self.device)

        for i in range(self.n_samples):
            # make predictions
            funcs = [self.solver.compute_func_val(n, c, *coords)
                     for n, c in zip(self.solver.nets, self.solver.conditions)]

            residuals = self.solver.diff_eqs(*funcs, *coords)
            residuals = torch.cat(residuals, dim=1)

            # get log variational posterior and log prior
            log_varposts[i] = torch.stack([net.log_varpost() for net in self.solver.nets])
            log_priors[i] = torch.stack([net.log_prior() for net in self.solver.nets])

            err_tol = self.get_error_tolerance(*coords)
            # calculate log likelihood
            log_likes[i] = torch.stack([Normal(funcs[idx], err_tol).log_prob(targets[idx]).sum()
                                       for idx in range(len(self.solver.nets))])

        if not take_mean:
            return log_priors, log_varposts, log_likes

        # calculate Monte Carlo estimate of the log variational posterior, prior and likelihood
        log_varpost = log_varposts.mean(dim=0)
        log_prior = log_priors.mean(dim=0)
        log_like = log_likes.mean(dim=0)

        # calculate the negative ELBO

        weight = torch.tensor(2 ** (self.solver._max_local_epoch - self.solver.local_epoch) /
                              (2 ** self.solver._max_local_epoch - 1))
        loss = weight*(log_varpost - log_prior) - log_like
        if self.solver._phase == "train":
            self.train_history["loss"].append(loss.tolist())
            self.train_history["log_varpost"].append(log_varpost.tolist())
            self.train_history["log_prior"].append(log_prior.tolist())
            self.train_history["log_likelihood"].append(log_like.tolist())
        else:
            self.valid_history["loss"].append(loss.tolist())
            self.valid_history["log_varpost"].append(log_varpost.tolist())
            self.valid_history["log_prior"].append(log_prior.tolist())
            self.valid_history["log_likelihood"].append(log_like.tolist())

        return loss[0]


class BBBResidualsLoss:
    def __init__(
            self, solver: Type[BaseSolver],
            error_tolerance: float, n_samples=1, device="cpu") -> None:
        self.solver = solver
        self.n_samples = n_samples
        self.error_tolerance = error_tolerance
        self.device = device
        self.train_history = {"loss": [], "log_varpost": [], "log_prior": [], "log_likelihood": []}
        self.valid_history = {"loss": [], "log_varpost": [], "log_prior": [], "log_likelihood": []}

    def __call__(self, residuals, funcs, coords, take_mean=True) -> Union[Tuple, torch.Tensor]:
        log_varposts = torch.zeros(self.n_samples, len(self.solver.nets), device=self.device)
        log_priors = torch.zeros(self.n_samples, len(self.solver.nets), device=self.device)
        log_likes = torch.zeros(self.n_samples, len(self.solver.nets), device=self.device)

        for i in range(self.n_samples):
            # make predictions
            funcs = [self.solver.compute_func_val(n, c, *coords)
                     for n, c in zip(self.solver.nets, self.solver.conditions)]

            residuals = self.solver.diff_eqs(*funcs, *coords)
            residuals = torch.cat(residuals, dim=1)

            # get log variational posterior and log prior
            log_varposts[i] = torch.stack([net.log_varpost() for net in self.solver.nets])
            log_priors[i] = torch.stack([net.log_prior() for net in self.solver.nets])

            # calculate log likelihood
            log_likes[i] = torch.stack([Normal(0, self.error_tolerance).log_prob(residuals[:, idx]).sum()
                                       for idx in range(len(self.solver.nets))])

        if not take_mean:
            return log_priors, log_varposts, log_likes

        # calculate Monte Carlo estimate of the log variational posterior, prior and likelihood
        log_varpost = log_varposts.mean(dim=0)
        log_prior = log_priors.mean(dim=0)
        log_like = log_likes.mean(dim=0)

        # calculate the negative ELBO
        weight = torch.tensor(2 ** (self.solver._max_local_epoch - self.solver.local_epoch) /
                              (2 ** self.solver._max_local_epoch - 1))
        loss = weight*(log_varpost - log_prior) - log_like
        if self.solver._phase == "train":
            self.train_history["loss"].append(loss.item())
            self.train_history["log_varpost"].append(log_varpost.item())
            self.train_history["log_prior"].append(log_prior.item())
            self.train_history["log_likelihood"].append(log_like.item())
        else:
            self.valid_history["loss"].append(loss.item())
            self.valid_history["log_varpost"].append(log_varpost.item())
            self.valid_history["log_prior"].append(log_prior.item())
            self.valid_history["log_likelihood"].append(log_like.item())

        return loss[0]

class BVIPINNLoss:
    def __init__(self, solver, det_solution, n_samples=1, device="cpu") -> None:
        self.det_solution = det_solution
        self.n_samples = n_samples
        self.solver = solver
        self.device = device

    def __call__(self, residuals, funcs, coords):
        targets = self.det_solution(*coords)
        targets = [targets] if len(funcs) == 1 else targets

        log_varposts = torch.zeros(self.n_samples, len(funcs), device=self.device)
        log_priors = torch.zeros(self.n_samples, len(funcs), device=self.device)
        log_likes = torch.zeros(self.n_samples, len(funcs), device=self.device)

        for i in range(self.n_samples):
            # get log variational posterior and log prior
            log_varposts[i] = torch.stack([net.log_varpost() for net in self.solver.nets])
            log_priors[i] = torch.stack([net.log_prior() for net in self.solver.nets])

            # calculate log likelihood
            log_likes[i] = torch.stack([net.log_likelihood(coords, targets[i]) for idx, net in enumerate(self.solver.nets)])

        # calculate Monte Carlo estimate of the log variational posterior, prior and likelihood
        log_varpost = log_varposts.mean(dim=0)
        log_prior = log_priors.mean(dim=0)
        log_like = log_likes.mean(dim=0)

        # calculate the negative ELBO
        weight = torch.tensor(2 ** (self.solver._max_local_epoch - self.solver.local_epoch) /
                              (2 ** self.solver._max_local_epoch - 1), device=self.device)
        loss = weight * (log_varpost - log_prior) - log_like
        return loss[0]
    
class BVINNLoss():
    def __call__(self, model, guide, *args, **kwargs):
        # run the guide and trace its execution
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        # run the model and replay it against the samples from the guide
        model_trace = poutine.trace(
            poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        # construct the elbo loss function
        return -1*(model_trace.log_prob_sum() - guide_trace.log_prob_sum())
        
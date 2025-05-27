import numpy as np
import torch
import emcee
from neurodiffeq.solvers import BundleSolver1D, Solver1D
from neurodiffeq.solvers_utils import SolverConfig
from inverse.bayesian_adapters import nlm_sample_solutions

from models.nlm import NLMModel
from models.utils import StdGetter, StdGetterEB
from utils import Equation, NLMConfig, InverseConfig

def run_forward(equation: Equation, method_config: NLMConfig, experiment_name: str, args=None):
    print("Runnning forward...")
    base_method_name = args.basemethod
    det_nets = torch.load(f"checkpoints/nets_{args.equation}_{base_method_name}.pt", map_location=method_config.device)
    det_solver = Solver1D.load(f"checkpoints/solver_{args.equation}_{base_method_name}.ndeq")
    det_solver.nets = [n.to(method_config.device) for n in det_solver.nets]
    det_solver.best_nets = [n.to(method_config.device) for n in det_solver.best_nets]
    det_solution = det_solver.get_solution(best=False)

    if args.errorbounds:
        get_likelihood_stds = StdGetterEB(
            equation.coords_train_min[0],
            equation.coords_test_max[0],
            equation.int_eP,
            equation.eP,
            lambda z: det_solver.get_residuals(z, best=False),
            number_of_stds=equation.system_size,
            experiment_name=args.equation + "_foward")
    else:
        get_likelihood_stds = StdGetter(method_config.likelihood_std, equation.coords_train_min[0], equation.system_size)

    nlm_model = NLMModel(det_nets, det_solution, get_likelihood_stds, equation.system, device=method_config.device)
    
    priors = torch.linspace(*method_config.prior_calibration_range, method_config.prior_calibration_points, device=method_config.device)
    t_test = torch.linspace(equation.coords_test_min[0], equation.coords_test_max[0], 100, device=method_config.device).unsqueeze(-1)
    t_train = torch.linspace(equation.coords_train_min[0], equation.coords_train_max[0], 100, device=method_config.device).unsqueeze(-1)

    valid_priors, prior_opts, error_norms = nlm_model.calibrate_prior(priors, [t_test], [t_train])
    best_priors = [prior_opt if prior_opt is not None else 1 for prior_opt in prior_opts]
    nlm_model.fit(sigma_priors=best_priors, coordinates=[t_train])

    nlm_model.save(f"checkpoints/model_{experiment_name}.pt")

def run_bundle(equation: Equation, method_config: NLMConfig, inverse_config: InverseConfig, experiment_name: str, args):
    base_method_name = args.basemethod
    if args.inverse in [0, 1]:
        print("Runnning bundle...")
        det_nets = torch.load(f"checkpoints/nets_bundle_{args.equation}_{base_method_name}.pt", map_location=method_config.device)
        det_solver = BundleSolver1D.load(f"checkpoints/solver_bundle_{args.equation}_{base_method_name}.ndeq")
        det_solver.nets = [n.to(method_config.device) for n in det_solver.nets]
        det_solver.best_nets = [n.to(method_config.device) for n in det_solver.best_nets]
        det_solution = det_solver.get_solution()

        if args.errorbounds:
            eb_parameters_min = equation.eb_parameters_min if equation.eb_parameters_min is not None else equation.bundle_parameters_min
            eb_parameters_max = equation.eb_parameters_max if equation.eb_parameters_max is not None else equation.bundle_parameters_max
            bounded_params = np.linspace(eb_parameters_min, eb_parameters_max, method_config.eb_dimension_batch_size)
            get_likelihood_stds = StdGetterEB(
                equation.coords_train_min[0],
                equation.coords_test_max[0],
                equation.int_eP,
                equation.eP,
                lambda *x: det_solver.get_residuals(*x, best=False),
                bounded_params,
                equation.system_size,
                device=method_config.device,
                experiment_name=args.equation + "_bundle")
        else:
            get_likelihood_stds = StdGetter(method_config.likelihood_std, equation.coords_train_min[0], equation.system_size)
        
        
        nlm_model = NLMModel(det_nets, det_solution, get_likelihood_stds, equation.system_bundle, device=method_config.device)

        priors = torch.linspace(*method_config.prior_calibration_range, method_config.prior_calibration_points, device=method_config.device)
        coords_test = torch.meshgrid(
            torch.linspace(equation.coords_test_min[0], equation.coords_test_max[0], method_config.dimension_batch_size[0], device=method_config.device),
            *[torch.linspace(p_min, p_max, method_config.dimension_batch_size[i+1], device=method_config.device) for i, (p_min, p_max) in enumerate(zip(equation.bundle_parameters_min, equation.bundle_parameters_max))],
            indexing='ij')
        coords_train = torch.meshgrid(
            torch.linspace(equation.coords_train_min[0], equation.coords_train_max[0], method_config.dimension_batch_size[0], device=method_config.device),
            *[torch.linspace(p_min, p_max, method_config.dimension_batch_size[i+1], device=method_config.device) for i, (p_min, p_max) in enumerate(zip(equation.bundle_parameters_min, equation.bundle_parameters_max))],
            indexing='ij')
        coords_test = [c.reshape(-1, 1) for c in coords_test]
        coords_train = [c.reshape(-1, 1) for c in coords_train]

        # valid_priors, prior_opts, error_norms = nlm_model.calibrate_prior(priors, coords_test, coords_train)
        # best_priors = [prior_opt if prior_opt is not None else 1 for prior_opt in prior_opts]
        best_priors = [1] * len(det_solver.nets)
        
        nlm_model.fit(sigma_priors=best_priors, coordinates=coords_train)
        nlm_model.save(f"checkpoints/model_{experiment_name}.pt")
    else:
        print("Skipping bundle...")

    if args.inverse in [1, 2]:
        print("Running inverse...")
        nlm_model = NLMModel.load(f"checkpoints/model_{experiment_name}.pt", device=inverse_config.device)
        nlm_model.det_nets = [n.to(inverse_config.device) for n in nlm_model.det_nets]
        nlm_model.device = inverse_config.device
        nlm_model.get_likelihood_stds.device = inverse_config.device

        adapted_sampler = nlm_sample_solutions(nlm_model, inverse_config.solution_samples, len(equation.bundle_parameters_min))

        log_posterior = inverse_config.log_posterior_evaluator(adapted_sampler, "bayesian", f"datasets/{args.inverse_dataset}.csv").log_posterior
        backend = emcee.backends.HDFBackend(f"checkpoints/inverse_samples_{experiment_name}.h5")
        sampler = emcee.EnsembleSampler(inverse_config.chains, len(inverse_config.inverse_params_min), log_posterior, backend=backend)

        p0 = np.random.uniform(inverse_config.inverse_params_min, inverse_config.inverse_params_max, (inverse_config.chains, len(inverse_config.inverse_params_min)))
        sampler.run_mcmc(p0, inverse_config.samples, progress=True, tune=True)
        samples = sampler.get_chain(flat=True)

        print("Inverse samples mean and std:")
        for i, (m, s) in  enumerate(zip(samples.mean(axis=0), samples.std(axis=0))):
            print(f"Parameter {i+1}: {m} +- {s}")
        np.save(f"checkpoints/inverse_samples_{experiment_name}_{args.inverse_dataset}.npy", samples)
    else:
        print("Skipping inverse...")
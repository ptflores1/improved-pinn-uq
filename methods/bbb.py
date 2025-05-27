import numpy as np
import emcee
import torch
import dill
from neurodiffeq.solvers import BundleSolver1D, Solver1D
from neurodiffeq.generators import Generator1D
from inverse.bayesian_adapters import bbb_sample_solutions

from models.nets import BVINN, BVIDetSolver, BVISolver
from models.utils import StdGetter, StdGetterEB
from utils import Equation, BBBConfig, InverseConfig

def run_forward_det(equation: Equation, method_config: BBBConfig, experiment_name: str, args):
    print("Runnning forward...")
    base_method_name = args.basemethod
    det_nets = torch.load(f"checkpoints/nets_{args.equation}_{base_method_name}.pt", map_location=method_config.device)
    det_solver = Solver1D.load(f"checkpoints/solver_{args.equation}_{base_method_name}.ndeq")
    det_solver.conditions = equation.conditions
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
            experiment_name=args.equation + "_foward",
            device=method_config.device)
    else:
        get_likelihood_stds = StdGetter(method_config.likelihood_std, equation.coords_train_min[0])

    train_generator = Generator1D(method_config.dimension_batch_size, t_min=equation.coords_train_min[0], t_max=equation.coords_train_max[0], method='uniform')
    valid_generator = Generator1D(method_config.dimension_batch_size, t_min=equation.coords_train_min[0], t_max=equation.coords_train_max[0], method='uniform')

    nets = [
        BVINN(method_config.input_features, method_config.output_features,
            method_config.hidden_units,
            prior_std=method_config.prior_std,
            actv=method_config.activation,
            pretrained_weights=det_net).to(method_config.device)
            for det_net in det_nets
    ]

    bvi_solver = BVIDetSolver(nets, equation.conditions, train_generator, det_solution, equation.system, lr=method_config.lr, get_likelihood_std=get_likelihood_stds, output_variance=args.output_variance)
    bvi_solver.fit(max_epochs=method_config.iterations)

    torch.save([n.to("cpu") for n in nets], f"checkpoints/nets_{experiment_name}.pt")
    torch.save(bvi_solver.to("cpu"), f"checkpoints/solver_{experiment_name}.pyro", pickle_module=dill)

def run_forward_res(equation: Equation, method_config: BBBConfig, experiment_name: str, args):
    print("Runnning forward...")
    train_generator = Generator1D(method_config.dimension_batch_size, t_min=equation.coords_train_min[0], t_max=equation.coords_train_max[0], method='uniform')

    nets = [
        BVINN(method_config.input_features, method_config.output_features,
            method_config.hidden_units,
            prior_std=method_config.prior_std,
            actv=method_config.activation,
            pretrained_weights=det_net).to(method_config.device)
            for det_net in range(equation.system_size)
    ]
    
    bvi_solver = BVISolver(nets, equation.system, equation.conditions, train_generator, method_config.res_likelihood_std, lr=method_config.lr)
    bvi_solver.fit(max_epochs=method_config.iterations)

    torch.save([n.to("cpu") for n in nets], f"checkpoints/nets_{experiment_name}.pt")
    torch.save(bvi_solver.to("cpu"), f"checkpoints/solver_{experiment_name}.pyro", pickle_module=dill)

def run_forward(equation: Equation, method_config: BBBConfig, experiment_name: str, args):
    if args.res_loss:
        run_forward_res(equation, method_config, experiment_name, args)
    else:
        run_forward_det(equation, method_config, experiment_name, args)

def run_bundle_det(equation: Equation, method_config: BBBConfig, inverse_config: InverseConfig, experiment_name: str, args):
    base_method_name = args.basemethod
    if args.inverse in [0, 1]:
        print("Runnning bundle...")
        print(f"checkpoints/nets_bundle_{args.equation}_{base_method_name}.pt")
        det_nets = torch.load(f"checkpoints/nets_bundle_{args.equation}_{base_method_name}.pt", map_location=method_config.device)
        det_solver = BundleSolver1D.load(f"checkpoints/solver_bundle_{args.equation}_{base_method_name}.ndeq")
        det_solver.conditions = equation.bundle_conditions
        det_solver.nets = [n.to(method_config.device) for n in det_solver.nets]
        det_solver.best_nets = [n.to(method_config.device) for n in det_solver.best_nets]
        det_solution = det_solver.get_solution(best=False)

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
                experiment_name=args.equation + "_bundle",
                device=method_config.device)
        else:
            get_likelihood_stds = StdGetter(method_config.likelihood_std, equation.coords_train_min[0])

        nets = [
            BVINN(method_config.input_features, method_config.output_features,
                method_config.hidden_units,
                prior_std=method_config.prior_std,
                actv=method_config.activation,
                pretrained_weights=det_net).to(method_config.device)
                for det_net in det_nets
        ]

        train_generator = Generator1D(method_config.dimension_batch_size[0], t_min=equation.coords_train_min[0], t_max=equation.coords_train_max[0], method='uniform')
        for i in range(len(equation.bundle_parameters_min)):
            train_generator ^= Generator1D(method_config.dimension_batch_size[i+1], t_min=equation.bundle_parameters_min[i], t_max=equation.bundle_parameters_max[i], method='uniform')

        bvi_solver = BVIDetSolver(nets, equation.bundle_conditions, train_generator, det_solution, equation.system_bundle, lr=method_config.lr, get_likelihood_std=get_likelihood_stds, output_variance=args.output_variance)        
        bvi_solver.fit(max_epochs=method_config.iterations)

        torch.save([n.to("cpu") for n in nets], f"checkpoints/nets_{experiment_name}.pt", pickle_module=dill)
        torch.save(bvi_solver.to("cpu"), f"checkpoints/solver_{experiment_name}.pyro", pickle_module=dill)
    else:
        print("Skipping bundle...")

    if args.inverse in [1, 2]:
        print("Running inverse...")
        nets = torch.load(f'checkpoints/nets_{experiment_name}.pt', map_location=inverse_config.device)
        solver = torch.load(f'checkpoints/solver_{experiment_name}.pyro', map_location=inverse_config.device)
        solver.conditions = equation.bundle_conditions

        if "get_likelihood_std" in solver.__dict__:
            solver.get_likelihood_std.device = "cpu"

        adapted_sampler = bbb_sample_solutions(solver, inverse_config.solution_samples, len(equation.bundle_parameters_min))
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

def run_bundle_res(equation: Equation, method_config: BBBConfig, inverse_config: InverseConfig, experiment_name: str, args):
    if args.inverse in [0, 1]:
        print("Runnning bundle...")
        print(f"checkpoints/nets_bundle_{args.equation}.pt")

        nets = [
            BVINN(method_config.input_features, method_config.output_features,
                method_config.hidden_units,
                prior_std=method_config.prior_std,
                actv=method_config.activation).to(method_config.device)
                for _ in range(equation.system_size)
        ]

        train_generator = Generator1D(method_config.dimension_batch_size[0], t_min=equation.coords_train_min[0], t_max=equation.coords_train_max[0], method='uniform')
        for i in range(len(equation.bundle_parameters_min)):
            train_generator ^= Generator1D(method_config.dimension_batch_size[i+1], t_min=equation.bundle_parameters_min[i], t_max=equation.bundle_parameters_max[i], method='uniform')

        bvi_solver = BVISolver(nets, equation.system_bundle, equation.bundle_conditions, train_generator, method_config.res_likelihood_std, lr=method_config.lr)        
        bvi_solver.fit(max_epochs=method_config.iterations)

        torch.save([n.to("cpu") for n in nets], f"checkpoints/nets_{experiment_name}.pt", pickle_module=dill)
        torch.save(bvi_solver.to("cpu"), f"checkpoints/solver_{experiment_name}.pyro", pickle_module=dill)
    else:
        print("Skipping bundle...")

    if args.inverse in [1, 2]:
        print("Running inverse...")
        nets = torch.load(f'checkpoints/nets_{experiment_name}.pt', map_location=inverse_config.device)
        solver = torch.load(f'checkpoints/solver_{experiment_name}.pyro', map_location=inverse_config.device)
        solver.conditions = equation.bundle_conditions
        if "get_likelihood_std" in solver.__dict__:
            solver.get_likelihood_std.device = "cpu"

        adapted_sampler = bbb_sample_solutions(solver, inverse_config.solution_samples, len(equation.bundle_parameters_min))

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

def run_bundle(equation: Equation, method_config: BBBConfig, inverse_config: InverseConfig, experiment_name: str, args):
    if args.res_loss:
        run_bundle_res(equation, method_config, inverse_config, experiment_name, args)
    else:
        run_bundle_det(equation, method_config, inverse_config, experiment_name, args)
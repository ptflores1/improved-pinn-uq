import numpy as np
import torch
import emcee
from neurodiffeq.solvers import BundleSolver1D, Solver1D
from neurodiffeq.generators import Generator1D
from models.callbacks import NeurodiffeqWNBCallback

from models.nets import FCNN
from utils import Equation, FCNNConfig, InverseConfig


def run_forward(equation: Equation, method_config: FCNNConfig, experiment_name: str, args):
    print("Runnning forward...")
    nets = [FCNN(input_features=method_config.input_features, output_features=method_config.output_features,
                hidden_units=method_config.hidden_units, actv=method_config.activation, device=method_config.device)
            for _ in range(equation.system_size)]
    
    train_generator = Generator1D(method_config.dimension_batch_size, t_min=equation.coords_train_min[0], t_max=equation.coords_train_max[0], method='uniform')
    valid_generator = Generator1D(method_config.dimension_batch_size, t_min=equation.coords_train_min[0], t_max=equation.coords_train_max[0], method='uniform')
    
    solver = Solver1D(ode_system=equation.system,
                    conditions=equation.conditions,
                    t_min=equation.coords_train_min[0], t_max=equation.coords_train_max[0],
                    train_generator=train_generator,
                    valid_generator=valid_generator,
                    nets=nets,
                    loss_fn=equation.loss_fn
                    )
    solver.fit(method_config.iterations, callbacks=(NeurodiffeqWNBCallback(), ))

    solver.nets = [n.to("cpu") for n in solver.nets]
    solver.best_nets = [n.to("cpu") for n in solver.best_nets]
    torch.save(solver._get_internal_variables()['best_nets'], f"checkpoints/nets_{experiment_name}.pt")
    solver.save(f"checkpoints/solver_{experiment_name}.ndeq")

def run_bundle(equation: Equation, method_config: FCNNConfig, inverse_config: InverseConfig, experiment_name: str, args):
    if args.inverse in [0, 1]:
        print("Runnning bundle...")
        nets = [FCNN(input_features=method_config.input_features, output_features=method_config.output_features,
                    hidden_units=method_config.hidden_units, actv=method_config.activation, device=method_config.device).to(method_config.device)
                for _ in range(equation.system_size)]
        
        train_generator = Generator1D(method_config.dimension_batch_size[0], t_min=equation.coords_train_min[0], t_max=equation.coords_train_max[0], method='equally-spaced-noisy')
        for i in range(len(equation.bundle_parameters_min)):
            train_generator ^= Generator1D(method_config.dimension_batch_size[i+1], t_min=equation.bundle_parameters_min[i], t_max=equation.bundle_parameters_max[i], method='equally-spaced')

        solver = BundleSolver1D(ode_system=equation.system_bundle,
                        conditions=equation.bundle_conditions,
                        t_min=None, t_max=None,
                        train_generator=train_generator,
                        valid_generator=train_generator,
                        nets=nets,
                        loss_fn=equation.loss_fn,
                        eq_param_index=tuple(range(len(equation.bundle_parameters_min)))
                        )
        solver.fit(method_config.iterations, callbacks=(NeurodiffeqWNBCallback(), ))

        solver.nets = [n.to("cpu") for n in solver.nets]
        solver.best_nets = [n.to("cpu") for n in solver.best_nets]
        torch.save(solver._get_internal_variables()['best_nets'], f"checkpoints/nets_{experiment_name}.pt")
        solver.save(f"checkpoints/solver_{experiment_name}.ndeq")
        print(f"checkpoints/solver_{experiment_name}.ndeq")
    else:
        print("Skipping bundle...")

    if args.inverse in [1, 2]:
        print("Running inverse...")
        solver = BundleSolver1D.load(f"checkpoints/solver_{experiment_name}.ndeq")
        solver.nets = [n.to(inverse_config.device) for n in solver.nets]
        solver.conditions = equation.bundle_conditions
        solution = solver.get_solution(best=False)

        log_posterior = inverse_config.log_posterior_evaluator(solution, "deterministic", f"datasets/{args.inverse_dataset}.csv").log_posterior
        backend = emcee.backends.HDFBackend(f"checkpoints/inverse_samples_{experiment_name}.h5")
        sampler = emcee.EnsembleSampler(inverse_config.chains, len(inverse_config.inverse_params_min), log_posterior, backend=backend)

        p0 = np.random.uniform(inverse_config.inverse_params_min, inverse_config.inverse_params_max, (inverse_config.chains, len(inverse_config.inverse_params_min)))
        sampler.run_mcmc(p0, inverse_config.samples, progress=True, tune=True) #skip_initial_state_check=True
        samples = sampler.get_chain(flat=True)

        print("Inverse samples mean and std:")
        for i, (m, s) in  enumerate(zip(samples.mean(axis=0), samples.std(axis=0))):
            print(f"Parameter {i+1}: {m} +- {s}")
        np.save(f"checkpoints/inverse_samples_{experiment_name}_{args.inverse_dataset}.npy", samples)
    else:
        print("Skipping inverse...")

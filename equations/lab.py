import torch
from neurodiffeq import diff
from neurodiffeq.conditions import IVP, BundleIVP
from inverse.lab import LABBayesian, LABDeterministic
from utils import BBBConfig, Equation, FCNNConfig, HMCConfig, InverseConfig, NLMConfig

alpha = 8.96e-2
beta = 3.12704e-4
gamma = 2.1899e-3
tau = 6.49e-4
delta = 1.76e-2
theta = 4.4e-3
rho = 2.47e-4
sigma = 6.7e-2
phi = 6.55e-3
omega = 1.5e-4
params = [alpha, delta, sigma]


coords_train_min = (0,)
coords_train_max = (24,)
coords_test_min = (0,)
coords_test_max = (48,)
bundle_parameters_min = tuple(p - .5 * p for p in params)
bundle_parameters_max = tuple(p + .5 * p for p in params)
non_bundle_parameters_min = ()
non_bundle_parameters_max = ()
inverse_params_min = bundle_parameters_min
inverse_params_max = bundle_parameters_max
bundle_parameters_min_plot = bundle_parameters_min
bundle_parameters_max_plot = bundle_parameters_max
bundle_plot_dimension_sizes = (50, 50, 50, 50)

initial_conditions = [7.5797, 6.44, 1.9]
conditions = conditions = [IVP(0, initial_conditions[0]), IVP(0, initial_conditions[1]), IVP(0, initial_conditions[2])]
bundle_conditions = [BundleIVP(0, initial_conditions[0]), BundleIVP(0, initial_conditions[1]), BundleIVP(0, initial_conditions[2])]

def system_bundle(x, y, z, t, alpha, delta, sigma):
    return [
        alpha*x - beta*x**2 - gamma*x*y**2 - tau*x*z**2 - diff(x, t),
        delta*y - theta*y**2 - rho*y*z - diff(y, t),
        sigma*z - phi*z**2 - omega*x*z - diff(z, t),
    ] 

def system(x, y, z, t):
    return [
        alpha*x - beta*x**2 - gamma*x*y**2 - tau*x*z**2 - diff(x, t),
        delta*y - theta*y**2 - rho*y*z - diff(y, t),
        sigma*z - phi*z**2 - omega*x*z - diff(z, t),
    ] 

equation = Equation(coords_train_min=coords_train_min, coords_train_max=coords_train_max,
                    coords_test_min=coords_test_min, coords_test_max=coords_test_max,
                    bundle_parameters_min=bundle_parameters_min, bundle_parameters_max=bundle_parameters_max,
                    non_bundle_parameters_min=non_bundle_parameters_min, non_bundle_parameters_max=non_bundle_parameters_max,
                    system=system, system_bundle=system_bundle, loss_fn=None, analytic=None,
                    system_size=3, conditions=conditions, bundle_conditions=bundle_conditions,
                    int_eP=None, eP=None)

## METHOD RELATED INFORMATION ##

fcnn_config = FCNNConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=64, lr=0.001)
nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=500, likelihood_std=.01, res_likelihood_std=.01)
bbb_config = BBBConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=100, prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01)
hmc_config = HMCConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01)

fcnn_config_bundle = FCNNConfig(input_features=4, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=(16, 16, 16, 16), lr=0.001)
nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=(12, 12, 12, 12), likelihood_std=.01, res_likelihood_std=.01)
bbb_config_bundle = BBBConfig(input_features=4, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=(16, 16, 16, 16), prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01)
hmc_config_bundle = HMCConfig(input_features=4, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=(16, 16, 16, 16), target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01)

# fcnn_config = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=1000, dimension_batch_size=16, lr=0.001)
# nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=16, dimension_batch_size=500, likelihood_std=.01, res_likelihood_std=.01)
# bbb_config = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=16, prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01)
# hmc_config = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=20, tune_samples=10, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01)

# fcnn_config_bundle = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100, dimension_batch_size=16, lr=0.001)
# nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=10, dimension_batch_size=16, likelihood_std=.01, res_likelihood_std=.01)
# bbb_config_bundle = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=32, prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01)
# hmc_config_bundle = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=20, tune_samples=10, chains=1, dimension_batch_size=32, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01)

methods_configs = {
    "fcnn": fcnn_config,
    "nlm": nlm_config,
    "bbb": bbb_config,
    "hmc": hmc_config,
}

methods_configs_bundle = {
    "fcnn": fcnn_config_bundle,
    "nlm": nlm_config_bundle,
    "bbb": bbb_config_bundle,
    "hmc": hmc_config_bundle,
}

## INVERSE RELATED INFORMATION ##

def get_inverse_posterior_evaluator(solution, type, dataset_path):
    parameters_min = equation.bundle_parameters_min + equation.non_bundle_parameters_min
    parameters_max = equation.bundle_parameters_max + equation.non_bundle_parameters_max
    if type == "deterministic":
        return LABDeterministic(parameters_min, parameters_max, solution, dataset_path)
    return LABBayesian(parameters_min, parameters_max, solution, dataset_path)


inverse_configs = {
    "fcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
    "nlm": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cpu"),
    "bbb": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=16, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
    "hmc": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
}

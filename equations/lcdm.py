import torch
import numpy as np
from neurodiffeq import diff
from neurodiffeq.conditions import IVP, BundleIVP
from inverse.cc import CCBayesian, CCDeterministic
from utils import FCNNConfig, InverseConfig, NLMConfig, BBBConfig, HMCConfig, Equation

## EQUATION RELATED INFORMATION ##

# Set the range of the independent variable:
z_0 = 0.0
z_f = 3.0
z_f_test = 6.0

# Fized parameter for non-bundle solutions
Om_m_0 = 0.2

# Set the range of the parameter of the bundle:
Om_m_0_min = 0.1
Om_m_0_max = 0.4

H_0_min = 50
H_0_max = 80

coords_train_min = (z_0,)
coords_train_max = (z_f,)
coords_test_min = (z_0,)
coords_test_max = (z_f_test,)
bundle_parameters_min = (Om_m_0_min,)
bundle_parameters_max = (Om_m_0_max,)
bundle_parameters_min_plot = (Om_m_0_min,)
bundle_parameters_max_plot = (Om_m_0_max,)
non_bundle_parameters_min = (H_0_min,)
non_bundle_parameters_max = (H_0_max,)
inverse_params_min = (Om_m_0_min, H_0_min)
inverse_params_max = (Om_m_0_max, H_0_max)
bundle_plot_dimension_sizes = (200, 200)

conditions = [IVP(t_0=z_0, u_0=Om_m_0)]
bundle_conditions = [BundleIVP(t_0=z_0, bundle_param_lookup={'u_0': 0}), ]

# Define the differential equation:
def system_bundle(x, z, Om_m_0):
    r"""Function that defines the differential equation of the system, by defining the residual of it. In this case:
    :math:`\displaystyle \mathcal{R}\left(\tilde{x},z\right)=\dfrac{d\tilde{x}}{dz} - \dfrac{3\tilde{x}}{1+z}.`
    :param x: The reparametrized output of the network corresponding to the dependent variable.
    :type x: `torch.Tensor`.
    :param z: The independent variable.
    :type z: `torch.Tensor`.
    :return: The residual of the differential equation.
    :rtype: list[`torch.Tensor`].
    """
    res = diff(x, z) - 3*x/(1 + z)
    return [res]

def system(x, z):
    r"""Function that defines the differential equation of the system, by defining the residual of it. In this case:
    :math:`\displaystyle \mathcal{R}\left(\tilde{x},z\right)=\dfrac{d\tilde{x}}{dz} - \dfrac{3\tilde{x}}{1+z}.`
    :param x: The reparametrized output of the network corresponding to the dependent variable.
    :type x: `torch.Tensor`.
    :param z: The independent variable.
    :type z: `torch.Tensor`.
    :return: The residual of the differential equation.
    :rtype: list[`torch.Tensor`].
    """
    res = diff(x, z) - 3*x/(1 + z)
    return [res]

# Define a custom loss function:
def weighted_loss_LCDM(res, x, t):
    r"""A custom loss function. While the default loss is the square of the residual,
    here a weighting function is added:
    :math:`\displaystyle L\left(\tilde{x},z\right)=\mathcal{R}\left(\tilde{x},z\right)^2e^{-2\left(z-z_0\right)}.`
    :param res: The residuals of the differential equation.
    :type res: `torch.Tensor`.
    :param x: The reparametrized output of the network corresponding to the dependent variable.
    :type x: `torch.Tensor`.
    :type t: The inputs of the neural network: i.e, the independent variable and the parameter of the bundle.
    :param t: list[`torch.Tensor`, `torch.Tensor`].
    :return: The mean value of the loss across the training points.
    :rtype: `torch.Tensor`.
    """
    z = t[0]
    w = 2

    loss = (res ** 2) * torch.exp(-w * (z - z_0))
    return loss.mean()

def analytic(z, Om_m_0=Om_m_0):
    c = (Om_m_0 * 3) ** (1 / 3)
    return ((1 + z) * c) ** 3 / 3

def H_LCDM(z, Om_m_0, H_0, x):
    r"""The Hubble parameter, :math:`H`, as a function of the redshift :math:`z`, the parameters of the funcion,
    and the reparametrized output of the network:

    :math:`\displaystyle H=H_0\sqrt{\tilde{x}+1-\Omega_{m,0}}.`

    :param z: The redshift.
    :type z: float or `numpy.array`.
    :param Om_m_0: The first parameter of the function.
    :type Om_m_0: float.
    :param H_0: The second parameter of the function.
    :type H_0: float.
    :param x:
        The reparametrized output of the network that represents the dependent variable
        of the differential system of :math:`\Lambda\mathrm{CDM}`.
    :type x function.
    :return: The value of the Hubble parameter.
    :rtype: float or `numpy.array`.
    """
    
    shape = np.ones(z.shape)

    if callable(x):
        Om_m_0s = Om_m_0*shape
        xs = x(z, Om_m_0s, to_numpy=True)
        xs = xs[0] if isinstance(xs, list) else xs
    else:
        xs = x

    # assert not ((xs + 1 - Om_m_0) < 0).any()
    H = H_0 * ((xs + 1 - Om_m_0) ** (1/2))
    #H = H_0 * (torch.tensor(xs + 1 - Om_m_0, device="cpu").clamp(min=0) ** (1/2)).numpy()
    #print("WARNING: H_LCDM is not correct!")
    return H

def int_eP(z, params=None):
    return .5 - 1/(2 * (1 + z)**2)

def eP(z, params=None):
    return (1+z)**(-3)

equation = Equation(coords_train_min=coords_train_min, coords_train_max=coords_train_max,
                    coords_test_min=coords_test_min, coords_test_max=coords_test_max,
                    bundle_parameters_min=bundle_parameters_min, bundle_parameters_max=bundle_parameters_max,
                    non_bundle_parameters_min=non_bundle_parameters_min, non_bundle_parameters_max=non_bundle_parameters_max,
                    system=system, system_bundle=system_bundle, loss_fn=weighted_loss_LCDM, analytic=analytic,
                    system_size=1, conditions=conditions, bundle_conditions=bundle_conditions,
                    int_eP=int_eP, eP=eP)

## METHOD RELATED INFORMATION ##

fcnn_config = FCNNConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=64, lr=0.001, device="cuda")
nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=500, likelihood_std=.1, res_likelihood_std=.1, device="cuda")
bbb_config = BBBConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=100, prior_std=1, lr=0.001, likelihood_std=.1, res_likelihood_std=.1, device="cuda")
hmc_config = HMCConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.1, res_likelihood_std=.1, device="cuda")

fcnn_config_bundle = FCNNConfig(input_features=2, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=(64, 64), lr=0.001, device="cuda")
nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=(100, 100), likelihood_std=.1, res_likelihood_std=.1, device="cuda")
bbb_config_bundle = BBBConfig(input_features=2, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=(64, 64), prior_std=1, lr=0.001, likelihood_std=.1, res_likelihood_std=.1, device="cuda")
hmc_config_bundle = HMCConfig(input_features=2, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=(32, 32), target_acceptance_rate=0.95, prior_std=1, likelihood_std=.1, res_likelihood_std=.1, device="cuda")

# fcnn_config = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=1000, dimension_batch_size=16, lr=0.001, device="cuda")
# nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=16, dimension_batch_size=500, likelihood_std=.01, device="cuda")
# bbb_config = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=16, prior_std=1, lr=0.001, likelihood_std=.01, device="cuda")
# hmc_config = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=20, tune_samples=10, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, device="cuda")

# fcnn_config_bundle = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100, dimension_batch_size=16, lr=0.001, device="cuda")
# nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=10, dimension_batch_size=16, likelihood_std=.01, device="cuda")
# bbb_config_bundle = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=32, prior_std=1, lr=0.001, likelihood_std=.01, device="cuda")
# hmc_config_bundle = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10, tune_samples=10, chains=1, dimension_batch_size=32, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, device="cuda")

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
        return CCDeterministic(parameters_min, parameters_max, solution, H_LCDM, dataset_path)
    return CCBayesian(parameters_min, parameters_max, solution, H_LCDM, dataset_path)

inverse_configs = {
    "fcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
    "nlm": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cpu"),
    "bbb": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
    "hmc": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
}


# inverse_configs = {
#     "fcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
#     "nlm": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
#     "bbb": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
#     "hmc": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
#     "clfcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator, device="cuda"),
# }

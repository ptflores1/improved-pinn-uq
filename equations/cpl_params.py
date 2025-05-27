import torch
import numpy as np
from neurodiffeq import diff
from neurodiffeq.conditions import IVP, BundleIVP
from inverse.cc import CCBayesian, CCDeterministic
from models.reparameterizations import CustomCondition, reparam_CPL
from utils import FCNNConfig, InverseConfig, NLMConfig, BBBConfig, HMCConfig, Equation

## EQUATION RELATED INFORMATION ##

z_0 = 0.0
z_f = 3.0
z_f_test = 6.0

w_0 = -.9
w_1 = -.3

w_0_min, w_0_max = -1.6, 0
w_1_min, w_1_max = -8, 3

Om_m_0_min = 0.0
Om_m_0_max = 0.6

H_0_min = 50
H_0_max = 80


coords_train_min = (z_0,)
coords_train_max = (z_f,)
coords_test_min = (z_0,)
coords_test_max = (z_f_test,)
# bundle_parameters_min = (w_0_min, w_1_min)
# bundle_parameters_max = (w_0_max, w_1_max)
bundle_parameters_min = (-1, -.6)
bundle_parameters_max = (-.8, 0)
eb_parameters_min = (-1.6, -8)
eb_parameters_max = (0, 3)
bundle_parameters_min_plot = (-1.6, -8)
bundle_parameters_max_plot = (0, 3)
non_bundle_parameters_min = (Om_m_0_min, H_0_min)
non_bundle_parameters_max = (Om_m_0_max, H_0_max)
inverse_params_min = (-1.6, -8, Om_m_0_min, H_0_min)
inverse_params_max = (-.4, 3, Om_m_0_max, H_0_max)
bundle_plot_dimension_sizes = (100, 4, 100)


conditions = [IVP(z_0, 1)]
bundle_conditions = [BundleIVP(z_0, 1)]


def system_bundle(x, z, w_0, w_1):
    r"""Function that defines the differential equation of the system, by defining the residual of it. In this case:
    :math:`\displaystyle \mathcal{R}\left(\tilde{x},z,\omega_0,\omega_1\right)=\dfrac{d\tilde{x}}{dz}
    - \dfrac{3\tilde{x}}{1+z}\left(1+\omega_0 + \dfrac{\omega_1 z}{1+z}\right).`
    :param x: The reparametrized outputs of the network corresponding to the dependent variable.
    :type x: `torch.Tensor`
    :param z: The independent variable.
    :type z: `torch.Tensor`
    :param w_0: The first parameter of the bundle.
    :type w_0: `torch.Tensor`
    :param w_1: The second parameter of the bundle.
    :type w_1: `torch.Tensor`
    :return: The residual of the differential equation.
    :rtype: list[`torch.Tensor`]
    """

    w = w_0 + (w_1*z/(1 + z))
    res = diff(x, z) - 3*((1 + w)/(1 + z))*x
    return [res]

def system(x, z):
    r"""Function that defines the differential equation of the system, by defining the residual of it. In this case:
    :math:`\displaystyle \mathcal{R}\left(\tilde{x},z,\omega_0,\omega_1\right)=\dfrac{d\tilde{x}}{dz}
    - \dfrac{3\tilde{x}}{1+z}\left(1+\omega_0 + \dfrac{\omega_1 z}{1+z}\right).`
    :param x: The reparametrized outputs of the network corresponding to the dependent variable.
    :type x: `torch.Tensor`
    :param z: The independent variable.
    :type z: `torch.Tensor`
    :param w_0: The first parameter of the bundle.
    :type w_0: `torch.Tensor`
    :param w_1: The second parameter of the bundle.
    :type w_1: `torch.Tensor`
    :return: The residual of the differential equation.
    :rtype: list[`torch.Tensor`]
    """

    w = w_0 + (w_1*z/(1 + z))

    res = diff(x, z) - 3*((1 + w)/(1 + z))*x
    return [res]

def analytic(z, w_0=None, w_1=None):
    w_0 = w_0 if w_0 is not None else -.9
    w_1 = w_1 if w_1 is not None else -.3
    exp = 3*(torch.log(z+1) + w_1*(torch.log(z+1) + 1/(1+z) - 1) + w_0*torch.log(z+1))
    return torch.exp(exp)

def H_CPL(z, w_0, w_1, Om_m_0, H_0, x):
    r"""The Hubble parameter, :math:`H`, as a function of the redshift :math:`z`, the parameters of the funcion,
    and the reparametrized outputs of the network:

    :math:`\displaystyle H=H_0\sqrt{\Omega_{m,0}\left(1+z\right)^3
    +\left(1-\Omega_{m,0}\right)\tilde{x}}.`

    :param z: The redshift.
    :type z: float or `numpy.array`.
    :param w_0: The first parameter of the function.
    :type w_0: float.
    :param w_1: The second parameter of the function.
    :type w_1: float.
    :param Om_m_0: The thrid parameter of the function.
    :type Om_m_0: float.
    :param H_0: The fourth parameter of the function.
    :type H_0: float.
    :param x:
        The reparametrized outputs of the network that represents the dependent variable
        of the differential system of CPL.
    :type x function.
    :return: The value of the Hubble parameter.
    :rtype: float or `numpy.array`.
    """

    shape = np.ones_like(z)
    if callable(x):
        xs = x(z, w_0*shape, w_1*shape, to_numpy=True)[0]
    else:
        xs = x
    H = H_0*((Om_m_0*((1+z)**3) + (1-Om_m_0)*xs) ** (1/2))
    #H = H_0*(torch.tensor(Om_m_0*((1+z)**3) + (1-Om_m_0)*xs, device="cpu").clamp(min=0) ** (1/2)).numpy()
    #print("WARNING: H_LCDM is not correct!")
    return H

def int_eP(z, w_0=w_0, w_1=w_1):
    z = z.detach().cpu()
    w_0 = w_0.detach().cpu() if isinstance(w_0, torch.Tensor) else w_0
    w_1 = w_1.detach().cpu() if isinstance(w_1, torch.Tensor) else w_1
    c = 3* (1 + w_0 + w_1)
    expint1 = float(sympy.re(expint(2 - c, 3*w_1)))
    b = (1 + z)**(c - 1)
    expint2 = float(sympy.re(expint(2 - c, 3*w_1/(1 + z))))
    
    return np.exp(3*w_1) / b *(-b * expint1 + expint2)

def eP(Z, w_0=w_0, w_1=w_1):
    print(w_0, w_1)
    Z = Z.detach().cpu()
    w_0 = w_0.detach().cpu() if isinstance(w_0, torch.Tensor) else w_0
    w_1 = w_1.detach().cpu() if isinstance(w_1, torch.Tensor) else w_1
    return np.exp(-3 *(-w_1 + w_1/(1+Z) + (1+w_0+w_1)*np.log(1 + Z)))

equation = Equation(coords_train_min=coords_train_min, coords_train_max=coords_train_max,
                    coords_test_min=coords_test_min, coords_test_max=coords_test_max,
                    bundle_parameters_min=bundle_parameters_min, bundle_parameters_max=bundle_parameters_max,
                    eb_parameters_min=eb_parameters_min, eb_parameters_max=eb_parameters_max,
                    non_bundle_parameters_min=non_bundle_parameters_min, non_bundle_parameters_max=non_bundle_parameters_max,
                    system=system, system_bundle=system_bundle, loss_fn=None, analytic=analytic,
                    system_size=1, conditions=conditions, bundle_conditions=bundle_conditions,
                    int_eP=int_eP, eP=eP)

## METHOD RELATED INFORMATION ##

fcnn_config = FCNNConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=64, lr=0.001)
nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=500, likelihood_std=.01, res_likelihood_std=.01)
bbb_config = BBBConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=100, prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01)
hmc_config = HMCConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01)

fcnn_config_bundle = FCNNConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=(32, 32, 32), lr=0.001)
nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=(32, 32, 32), likelihood_std=.01, res_likelihood_std=.01)
bbb_config_bundle = BBBConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=(32, 32, 32), prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01)
hmc_config_bundle = HMCConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=(32, 32, 32), target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01)

# fcnn_config = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=1000, dimension_batch_size=16, lr=0.001)
# nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=16, dimension_batch_size=500, likelihood_std=.01, res_likelihood_std=.01)
# bbb_config = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=16, prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01)
# hmc_config = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=20, tune_samples=10, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01)

# fcnn_config_bundle = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100, dimension_batch_size=16, lr=0.001)
# nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=10, dimension_batch_size=16, likelihood_std=.01, res_likelihood_std=.01, eb_dimension_batch_size=10)
# bbb_config_bundle = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=32, prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01, eb_dimension_batch_size=10)
# hmc_config_bundle = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=20, tune_samples=10, chains=1, dimension_batch_size=32, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01, eb_dimension_batch_size=10)

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
        return CCDeterministic(parameters_min, parameters_max, solution, H_CPL, dataset_path)
    return CCBayesian(parameters_min, parameters_max, solution, H_CPL, dataset_path)

inverse_configs = {
    "fcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "nlm": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "bbb": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "hmc": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
}
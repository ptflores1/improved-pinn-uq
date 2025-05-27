import torch
import numpy as np
from neurodiffeq import diff
from inverse.cc import CCBayesian, CCDeterministic

from models.reparameterizations import CustomCondition, quint_reparams
from utils import FCNNConfig, InverseConfig, NLMConfig, BBBConfig, HMCConfig, Equation

## EQUATION RELATED INFORMATION ##

# Set the range of the independent variable:
N_prime_0 = 0.0
N_prime_f = .5
N_prime_f_test = 1.0

# Set the range of the parameters of the bundle:
Om_m_0 = 0.25
lam_prime = 0.5

Om_m_0_min = 0.1
Om_m_0_max = 0.4

lam_prime_min = 0.0
lam_prime_max = 1.0

H_0_min = 50
H_0_max = 80

# Define the differential equation:
z_0 = 10.0 
N_0_abs = np.abs(np.log(1/(1 + z_0)))
lam_max = 3.0

coords_train_min = (N_prime_0,)
coords_train_max = (N_prime_f,)
coords_test_min = (N_prime_0,)
coords_test_max = (N_prime_f_test,)
bundle_parameters_min = (lam_prime_min, Om_m_0_min)
bundle_parameters_max = (lam_prime_max, Om_m_0_max)
bundle_parameters_min_plot = (lam_prime_min, Om_m_0_min)
bundle_parameters_max_plot = (lam_prime_max, Om_m_0_max)
non_bundle_parameters_min = (H_0_min,)
non_bundle_parameters_max = (H_0_max,)
inverse_params_min = (0, Om_m_0_min, H_0_min)
inverse_params_max = (3, Om_m_0_max, H_0_max)
bundle_plot_dimension_sizes = (100, 100, 4)

quint = quint_reparams(N_0_abs, lam_prime, Om_m_0)
conditions = bundle_conditions = [CustomCondition(quint.x_reparam), CustomCondition(quint.y_reparam)]


def system_bundle(x, y, N_prime, lam_prime, Om_m_0):
    r"""Function that defines the differential equations of the system, by defining the residuals of it. In this case:
    :math:`\displaystyle \mathcal{R}_1\left(\tilde{x},\tilde{y},N^{\prime},\lambda^{\prime}\right)=
    \dfrac{1}{\left|N_0\right|}\dfrac{d\tilde{x}}{dN^{\prime}}
    +3\tilde{x}-3\dfrac{\sqrt{6}}{2}\lambda^{\prime} \tilde{y}^{2}
    -\dfrac{3}{2}\tilde{x}\left(1+\tilde{x}^{2}-\tilde{y}^{2}\right),`
    :math:`\displaystyle \mathcal{R}_2\left(\tilde{x},\tilde{y},N^{\prime},\lambda^{\prime}\right)=
    \dfrac{1}{\left|N_0\right|}\dfrac{d\tilde{y}}{dN^{\prime}}
    +3\dfrac{\sqrt{6}}{2}\tilde{x}\tilde{y}\lambda^{\prime}
    -\dfrac{3}{2}\tilde{y}\left(1+\tilde{x}^{2}-\tilde{y}^{2}\right).`
    :param x: The reparametrized output of the network corresponding to the first dependent variable.
    :type x: `torch.Tensor`
    :param y: The reparametrized output of the network corresponding to the second dependent variable.
    :type y: `torch.Tensor`
    :param N_prime: The independent variable.
    :type N_prime: `torch.Tensor`
    :param theta: The parameters of the bundle.
    :type theta: list[`torch.Tensor`,`torch.Tensor`]
    :return: The residuals of the differential equations.
    :rtype: list[`torch.Tensor`, `torch.Tensor`]
    """

    res_1 = (diff(x, N_prime)/N_0_abs) + 3*x - lam_max*(np.sqrt(6)/2)*lam_prime*(y ** 2) - (3/2)*x*(1 + (x**2) - (y**2))
    res_2 = (diff(y, N_prime)/N_0_abs) + lam_max*(np.sqrt(6)/2)*lam_prime*(y * x) - (3/2)*y*(1 + (x**2) - (y**2))
    return res_1, res_2

def system(x, y, N_prime):
    res_1 = (diff(x, N_prime)/N_0_abs) + 3*x - lam_max*(np.sqrt(6)/2)*lam_prime*(y ** 2) - (3/2)*x*(1 + (x**2) - (y**2))
    res_2 = (diff(y, N_prime)/N_0_abs) + lam_max*(np.sqrt(6)/2)*lam_prime*(y * x) - (3/2)*y*(1 + (x**2) - (y**2))
    return res_1, res_2

# Define a custom loss function:

def weighted_loss_quint(res, f, t):
    r"""A custom loss function. While the default loss is the sum of the squares of the residuals,
    here a weighting function is added:
    :math:`\displaystyle L\left(\tilde{x},\tilde{y},N^{\prime},\lambda^{\prime}\right)
    =\sum^2_{i=1}\mathcal{R}_i\left(\tilde{x},\tilde{y},N^{\prime},\lambda^{\prime}\right)^2e^{-2N^{\prime}\lambda^{\prime}}`
    :param res: The residuals of the differential equation.
    :type res: `torch.Tensor`.
    :param f: The reparametrized outputs of the networks corresponding to the dependent variables.
    :type f: list[`torch.Tensor`, `torch.Tensor`, ...].
    :type t: The inputs of the neural network: i.e, the independent variable and the parameter of the bundle.
    :param t: list[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`].
    :return: The mean value of the loss across the training points.
    :rtype: `torch.Tensor`.
    """

    N_prime = t[0]
    w = 0
    mean_res = torch.mean(((torch.exp(-w * N_prime * lam_prime) * (res ** 2))), dim=0)
    return mean_res.sum()

def H_quint(z, lam, Om_m_0, H_0, solution):
    r"""The Hubble parameter, :math:`H`, as a function of the redshift :math:`z`, the parameters of the funcion,
    and the reprarametrized outputs of the neural networks:

    :math:`\displaystyle H=H^{\Lambda}_0\sqrt{\dfrac{\Omega_{m,0}^{\Lambda}\left(1+z\right)^3}
    {1-\tilde{x}^2-\tilde{y}^2}}.`

    :param z: The redshift.
    :type z: float or `numpy.array`.
    :param lam: The first parameter of the function.
    :type lam: float.
    :param Om_m_0: The second parameter of the function.
    :type Om_m_0: float.
    :param H_0: The third parameter of the function.
    :type H_0: float.
    :param x:
        The reparametrized output of the network that represents the first dependent variable
        of the differential system of Quintessence.
    :type x function.
    :param y:
        The reparametrized output of the network that represents the second dependent variable
        of the differential system of Quintessence.
    :type y function.
    :return: The value of the Hubble parameter.
    :rtype: float or `numpy.array`.
    """

    shape = np.ones_like(z)

    Ns = np.log(1/(1 + z))
    N_primes = (Ns/N_0_abs) + 1
    lam_prime = lam/lam_max
    lam_primes = lam_prime*shape

    Om_m_0s = Om_m_0*shape
    xs, ys = solution(N_primes, lam_primes, Om_m_0s, to_numpy=True) if callable(solution) else solution
    H = H_0*((Om_m_0 * ((1 + z) ** 3))/(1 - (xs ** 2) - (ys ** 2))) ** (1/2)
    #H = H_0*torch.tensor((Om_m_0 * ((1 + z) ** 3))/(1 - (xs ** 2) - (ys ** 2)), device="cpu").clamp(min=0).numpy() ** (1/2)
    #print("WARNING: H_LCDM is not correct!")
    return H

equation = Equation(coords_train_min=coords_train_min, coords_train_max=coords_train_max,
                    coords_test_min=coords_test_min, coords_test_max=coords_test_max,
                    bundle_parameters_min=bundle_parameters_min, bundle_parameters_max=bundle_parameters_max,
                    non_bundle_parameters_min=non_bundle_parameters_min, non_bundle_parameters_max=non_bundle_parameters_max,
                    system=system, system_bundle=system_bundle, loss_fn=weighted_loss_quint, analytic=None,
                    system_size=2, conditions=conditions, bundle_conditions=bundle_conditions,
                    int_eP=None, eP=None)

## METHOD RELATED INFORMATION ##

fcnn_config = FCNNConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=64, lr=0.001)
nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=500, likelihood_std=.005, res_likelihood_std=.005)
bbb_config = BBBConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=100, prior_std=1, lr=0.001, likelihood_std=.005, res_likelihood_std=.005)
hmc_config = HMCConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.005, res_likelihood_std=.005)

fcnn_config_bundle = FCNNConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=(32, 32, 32), lr=0.001)
nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=(32, 32, 32), likelihood_std=.005, res_likelihood_std=.005)
bbb_config_bundle = BBBConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=(32, 32, 32), prior_std=1, lr=0.001, likelihood_std=.005, res_likelihood_std=.005)
hmc_config_bundle = HMCConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=(32, 32, 32), target_acceptance_rate=0.95, prior_std=1, likelihood_std=.005, res_likelihood_std=.005)

# fcnn_config = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=1000, dimension_batch_size=16, lr=0.001)
# nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=16, dimension_batch_size=500, likelihood_std=.01)
# bbb_config = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=16, prior_std=1, lr=0.001, likelihood_std=.01)
# hmc_config = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=20, tune_samples=10, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01)

# fcnn_config_bundle = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100, dimension_batch_size=16, lr=0.001)
# nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=10, dimension_batch_size=16, likelihood_std=.01)
# bbb_config_bundle = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=32, prior_std=1, lr=0.001, likelihood_std=.01)
# hmc_config_bundle = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=20, tune_samples=10, chains=1, dimension_batch_size=32, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01)

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
        return CCDeterministic(parameters_min, parameters_max, solution, H_quint, dataset_path)
    return CCBayesian(parameters_min, parameters_max, solution, H_quint, dataset_path)

inverse_configs = {
    "fcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "nlm": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "bbb": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "hmc": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
}


# inverse_configs = {
#     "fcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator),
#     "nlm": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
#     "bbb": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
#     "hmc": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
#     "clfcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator),
# }

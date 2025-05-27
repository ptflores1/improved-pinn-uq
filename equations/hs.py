import torch
import numpy as np
from neurodiffeq import diff
from inverse.cc import CCBayesian, CCDeterministic
from models.reparameterizations import CustomCondition, HS_reparams
from utils import CLFCNNConfig, FCNNConfig, InverseConfig, NLMConfig, BBBConfig, HMCConfig, Equation

## EQUATION RELATED INFORMATION ##

# Set the range of the independent variable:
z_prime_0 = .0
# z_prime_f = 1.0
# z_prime_f_test = 1.0

# Set the range of the parameters of the bundle:
b_prime = 0.5
Om_m_0 = 0.25

b_prime_min = 0.000001
b_prime_max = 1.0
b_min = 0
b_max = 5

Om_m_0_min = 0.1
Om_m_0_max = 0.4

H_0_min = 50
H_0_max = 80

# Define the differential equation:
z_0 = 10.0
b_max = 5.0

coords_train_min = (.5,)
coords_train_max = (1.0,)
coords_test_min = (.0,)
coords_test_max = (1.0,)
bundle_parameters_min = (b_prime_min, Om_m_0_min)
bundle_parameters_max = (b_prime_max, Om_m_0_max)
bundle_parameters_min_plot = (b_prime_min, Om_m_0_min)
bundle_parameters_max_plot = (b_prime_max, Om_m_0_max)
non_bundle_parameters_min = (H_0_min,)
non_bundle_parameters_max = (H_0_max,)
inverse_params_min = (b_min, Om_m_0_min, H_0_min)
inverse_params_max = (b_max, Om_m_0_max, H_0_max)
bundle_plot_dimension_sizes = (100, 100, 4)

HS = HS_reparams(z_0=z_0, alpha=1/6, b_prime_min=b_prime_min, b_prime=b_prime, Om_m_0=Om_m_0)
conditions = bundle_conditions = [CustomCondition(HS.x_reparam),
                                  CustomCondition(HS.y_reparam),
                                  CustomCondition(HS.v_reparam),
                                  CustomCondition(HS.Om_reparam),
                                  CustomCondition(HS.r_prime_reparam)]

def system_bundle(x, y, v, Om, r_prime, z_prime, b_prime, Om_m_0):
    r"""Function that defines the differential equations of the system, by defining the residuals of it. In this case:
    :math:`\displaystyle
    \mathcal{R}_1\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{1}{z_0}\dfrac{d\tilde{x}}{dz^{\prime}}
    + \dfrac{-\tilde{\Omega}-2\tilde{v}+\tilde{x}+4\tilde{y}+\tilde{x}\tilde{v}+\tilde{x}^{2}}
    {z_{0}\left(1-z^{\prime}\right)+1},`
    :math:`\displaystyle
    \mathcal{R}_2\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{1}{z_0}\dfrac{d\tilde{y}}{dz^{\prime}}
    - \dfrac{\tilde{v}\tilde{x}\Gamma\left(\tilde{r}^{\prime}\right)-\tilde{x}\tilde{y}+4\tilde{y}-2\tilde{y}\tilde{v}}
    {z_{0}\left(1-z^{\prime}\right)+1},`
    :math:`\displaystyle
    \mathcal{R}_3\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{1}{z_0}\dfrac{d\tilde{v}}{dz^{\prime}}
    - \dfrac{\tilde{v}\left(\tilde{x}\Gamma\left(\tilde{r}^{\prime}\right)+4-2\tilde{v}\right)}
    {z_{0}\left(1-z^{\prime}\right)+1},`
    :math:`\displaystyle
    \mathcal{R}_4\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{1}{z_0}\dfrac{d\tilde{\Omega}}{dz^{\prime}}
    + \dfrac{\tilde{\Omega}\left(-1+2\tilde{v}+\tilde{x}\right)}{z_{0}\left(1-z^{\prime}\right)+1},`
    :math:`\displaystyle
    \mathcal{R}_5\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)=
    \dfrac{1}{z_0}\dfrac{d\tilde{r}^{\prime}}{dz^{\prime}}
    - \dfrac{\Gamma\left(\tilde{r}^{\prime}\right)\tilde{x}}{z_{0}\left(1-z^{\prime}\right)+1},`
    where :math:`\Gamma` is:
    :math:`\displaystyle \Gamma\left(\tilde{r}^\prime\right)=
    \dfrac{\left(e^{\tilde{r}^\prime}+5b^{\prime}\right)\left[\left(e^{\tilde{r}^\prime}+5b^{\prime}\right)^{2}-10b^{\prime}\right]}{20b^{\prime}e^{\tilde{r}^\prime}}.`
    :param x: The reparametrized output of the network corresponding to the first dependent variable.
    :type x: `torch.Tensor`
    :param y: The reparametrized output of the network corresponding to the second dependent variable.
    :type y: `torch.Tensor`
    :param v: The reparametrized output of the network corresponding to the third dependent variable.
    :type v: `torch.Tensor`
    :param Om: The reparametrized output of the network corresponding to the fourth dependent variable.
    :type Om: `torch.Tensor`
    :param r_prime: The reparametrized output of the network corresponding to the fifth dependent variable.
    :type r_prime: `torch.Tensor`
    :param z_prime: The independent variable.
    :type z_prime: `torch.Tensor`
    :param theta: The parameters of the bundle.
    :type theta: list[`torch.Tensor`,`torch.Tensor`]
    :return: The residuals of the differential equations.
    :rtype: list[`torch.Tensor`, `torch.Tensor`,...]
    """

    b = b_max * b_prime
    z = z_0 * (1 - z_prime)
    r = torch.exp(r_prime)

    Gamma = (r + b)*(((r + b)**2) - 2*b)/(4*r*b)

    # Equation System:
    res_1 = (diff(x, z_prime)/z_0) + (-Om - 2*v + x + 4*y + x*v + x**2)/(z + 1)
    res_2 = (diff(y, z_prime)/z_0) - (v*x*Gamma - x*y + 4*y - 2*y*v)/(z + 1)
    res_3 = (diff(v, z_prime)/z_0) - v*(x*Gamma + 4 - 2*v)/(z + 1)
    res_4 = (diff(Om, z_prime)/z_0) + Om*(-1 + 2*v + x)/(z + 1)
    res_5 = (diff(r_prime, z_prime)/z_0) - (Gamma*x)/(z + 1)

    return [res_1, res_2, res_3, res_4, res_5]

def system(x, y, v, Om, r_prime, z_prime):
    b = b_max * b_prime
    z = z_0 * (1 - z_prime)
    r = torch.exp(r_prime)

    Gamma = (r + b)*(((r + b)**2) - 2*b)/(4*r*b)

    # Equation System:
    res_1 = (diff(x, z_prime)/z_0) + (-Om - 2*v + x + 4*y + x*v + x**2)/(z + 1)
    res_2 = (diff(y, z_prime)/z_0) - (v*x*Gamma - x*y + 4*y - 2*y*v)/(z + 1)
    res_3 = (diff(v, z_prime)/z_0) - v*(x*Gamma + 4 - 2*v)/(z + 1)
    res_4 = (diff(Om, z_prime)/z_0) + Om*(-1 + 2*v + x)/(z + 1)
    res_5 = (diff(r_prime, z_prime)/z_0) - (Gamma*x)/(z + 1)

    return [res_1, res_2, res_3, res_4, res_5]

# Define a custom loss function:

def custom_loss_HS(res, f, t):
    r"""A custom loss function.
    In this case a sum of two different loss functions :math:`L_{\mathcal{R}}` and :math:`L_{\mathcal{C}}`.
    The former is the part of the loss that concerns the resiudals.
    While the default would be a sum of the squares of the residuals, here a weighting function is added:
    :math:`\displaystyle
    L_{\mathcal{R}}\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)
    =\sum^5_{i=1}\mathcal{R}_i\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},\tilde{r}^{\prime},z^{\prime},b^{\prime}\right)^2e^{-2z^{\prime}b^{\prime}}.`
    On the other hand, :math:`L_{\mathcal{C}}` is constructed from the relative difference between some
    equations' right and left hand side that must hold true due to symetries that relate the variables of the system.
    In particular:
    :math:`\displaystyle \begin{split}
    L_\mathcal{C}\left(\tilde{x},\tilde{y},\tilde{v},\tilde{\Omega},
    \tilde{r}^{\prime}, z^{\prime}, b^{\prime}, \Omega^{\Lambda}_{m,0}\right)=
    &\left(\tilde{\Omega} +\tilde{v}-\tilde{x}-\tilde{y}-1\right)^2\\
    &+ \left\{\dfrac{2\tilde{y}\Omega^\Lambda_{m,0}\left[1+z_{0}\left(1 - z^{\prime}\right)\right]^3}
    {\tilde{\Omega}e^{\tilde{r}^\prime}\left(1-\Omega^\Lambda_{m,0}\right)}
    \left[\dfrac{e^{\tilde{r}^\prime}+5b^{\prime}}{e^{\tilde{r}^\prime}+5b^{\prime}-2}\right]-1\right \}^2\\
    &+ \left\{\dfrac{2\tilde{v}\Omega^\Lambda_{m,0}\left[1+z_{0}\left(1 - z^{\prime}\right)\right]^{3}}
    {\tilde{\Omega}e^{\tilde{r}^\prime}\left(1-\Omega^\Lambda_{m,0}\right )}
    \left[\dfrac{\left(e^{\tilde{r}^\prime}+5b^{\prime}\right)^{2}}
    {\left(e^{\tilde{r}^\prime}+5b^{\prime}\right)^{2}-10b^{\prime}}\right]-1\right\}^2. \end{split}`
    Thus, the final total loss is:
    :math:`\displaystyle L=L_{\mathcal{R}}+L_{\mathcal{C}}.`
    :param res: The residuals of the differential equation.
    :type res: `torch.Tensor`.
    :param f: The reparametrized outputs of the networks corresponding to the dependent variables.
    :type f: list[`torch.Tensor`, `torch.Tensor`, ...].
    :type t: The inputs of the neural network: i.e, the independent variable and the parameter of the bundle.
    :param t: list[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`].
    :return: The mean value of the loss across the training points.
    :rtype: `torch.Tensor`.
    """
    z_prime = t[0]

    x = f[0]
    y = f[1]
    v = f[2]
    Om = f[3]
    r_prime = f[4]

    b = b_max*b_prime
    z = z_0 * (1 - z_prime)
    r = torch.exp(r_prime)

    w = 2

    loss_R = torch.exp(-w * z_prime * (b_prime - b_prime_min)) * (res ** 2)
    loss_C_1 = (Om + v - x - y - 1)**2
    loss_C_2 = (2*y*Om_m_0*((1+z)**3)*(r+b)/(r*Om*(1-Om_m_0)*(r+b-2)) - 1)**2
    loss_C_3 = (2*v*Om_m_0*((1+z)**3)*((r+b)**2)/(r*Om*(1-Om_m_0)*(((r+b)**2)-2*b)) - 1)**2
    loss_C = loss_C_1 + loss_C_2 + loss_C_3

    loss = torch.mean(loss_R + loss_C, dim=0).sum()

    return loss

def H_HS(z, b, Om_m_0, H_0, solution):
    r"""The Hubble parameter, :math:`H`, as a function of the redshift :math:`z`, the parameters of the funcion,
    and the reprarametrized outputs of the neural networks:

    :math:`\displaystyle H=H^\Lambda_{0}\sqrt{\dfrac{e^{\tilde{r}^\prime}}
    {2\tilde{v}}\left(1-\Omega^\Lambda_{m,0}\right)}.`

    :param z: The redshift.
    :type z: float or `numpy.array`.
    :param b: The first parameter of the function.
    :type b: float.
    :param Om_m_0: The second parameter of the function.
    :type Om_m_0: float.
    :param H_0: The third parameter of the function.
    :type H_0: float.
    :param v:
        The reparametrized output of the network that represents the third dependent variable
        of the differential system of Hu-Sawicki.
    :type v function.
    :param r_prime:
        The reparametrized output of the network that represents the fifth dependent variable
        of the differential system of Hu-Sawicki.
    :type r_prime function.
    :return: The value of the Hubble parameter.
    :rtype: float or `numpy.array`.
    """

    shape = np.ones_like(z)

    zs_prime = 1 - (z/z_0)
    b_prime = b/b_max
    b_primes = b_prime*shape
    Om_m_0s = Om_m_0*shape

    _, _, vs, _, r_primes = solution(zs_prime, b_primes, Om_m_0s, to_numpy=True) if callable(solution) else solution
    rs = np.exp(r_primes)

    H = H_0*np.sqrt(rs*(1 - Om_m_0)/(2*vs))
    #H = H_0*np.sqrt(torch.tensor(rs*(1 - Om_m_0)/(2*vs), device="cpu").clamp(min=0)).numpy()
    #print("WARNING: H_LCDM is not correct!")
    return H

equation = Equation(coords_train_min=coords_train_min, coords_train_max=coords_train_max,
                    coords_test_min=coords_test_min, coords_test_max=coords_test_max,
                    bundle_parameters_min=bundle_parameters_min, bundle_parameters_max=bundle_parameters_max,
                    non_bundle_parameters_min=non_bundle_parameters_min, non_bundle_parameters_max=non_bundle_parameters_max,
                    system=system, system_bundle=system_bundle, loss_fn=custom_loss_HS, analytic=None,
                    system_size=5, conditions=conditions, bundle_conditions=bundle_conditions,
                    int_eP=None, eP=None)

## METHOD RELATED INFORMATION ##

fcnn_config = FCNNConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=600_000, dimension_batch_size=128, lr=0.001)
clfcnn_config = CLFCNNConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, step_iterations=[100_000]*5, dimension_batch_size=128, coords_step_proportions=(0.2, 0.4, 0.6, 0.8, 1))
nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=500, likelihood_std=.005, res_likelihood_std=.005)
bbb_config = BBBConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=100, prior_std=1, lr=0.001, likelihood_std=.005, res_likelihood_std=.005)
hmc_config = HMCConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.005, res_likelihood_std=.005)

fcnn_config_bundle = FCNNConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=(128, 128, 64), lr=0.001)
clfcnn_config_bundle = CLFCNNConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, step_iterations=[100_000]*5, dimension_batch_size=(32, 32, 32), coords_step_proportions=(0.2, 0.4, 0.6, 0.8, 1))
nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=(32, 32, 32), likelihood_std=.005, res_likelihood_std=.005)
bbb_config_bundle = BBBConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=(32, 32, 32), prior_std=1, lr=0.001, likelihood_std=.005, res_likelihood_std=.005)
hmc_config_bundle = HMCConfig(input_features=3, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=(32, 32, 32), target_acceptance_rate=0.95, prior_std=1, likelihood_std=.005, res_likelihood_std=.005)

# fcnn_config = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=1000, dimension_batch_size=16, lr=0.001)
# clfcnn_config = CLFCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, step_iterations=[1_000]*5, dimension_batch_size=64, coords_step_proportions=(0.2, 0.4, 0.6, 0.8, 1))
# nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=16, dimension_batch_size=500, likelihood_std=.01)
# bbb_config = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=16, prior_std=1, lr=0.001, likelihood_std=.01)
# hmc_config = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=20, tune_samples=10, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01)

# fcnn_config_bundle = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100, dimension_batch_size=16, lr=0.001)
# clfcnn_config_bundle = CLFCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, step_iterations=[1_00]*5, dimension_batch_size=32, coords_step_proportions=(0.2, 0.4, 0.6, 0.8, 1))
# nlm_config_bundle = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=10, dimension_batch_size=16, likelihood_std=.01)
# bbb_config_bundle = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=10, dimension_batch_size=32, prior_std=1, lr=0.001, likelihood_std=.01)
# hmc_config_bundle = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10, tune_samples=10, chains=1, dimension_batch_size=32, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01)

methods_configs = {
    "fcnn": fcnn_config,
    "clfcnn": clfcnn_config,
    "nlm": nlm_config,
    "bbb": bbb_config,
    "hmc": hmc_config,
}

methods_configs_bundle = {
    "fcnn": fcnn_config_bundle,
    "clfcnn": clfcnn_config_bundle,
    "nlm": nlm_config_bundle,
    "bbb": bbb_config_bundle,
    "hmc": hmc_config_bundle,
}

## INVERSE RELATED INFORMATION ##

def get_inverse_posterior_evaluator(solution, type, dataset_path):
    parameters_min = equation.bundle_parameters_min + equation.non_bundle_parameters_min
    parameters_max = equation.bundle_parameters_max + equation.non_bundle_parameters_max
    if type == "deterministic":
        return CCDeterministic(parameters_min, parameters_max, solution, H_HS, dataset_path)
    return CCBayesian(parameters_min, parameters_max, solution, H_HS, dataset_path)

inverse_configs = {
    "fcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "nlm": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "bbb": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "hmc": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
    "clfcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator),
}

# inverse_configs = {
#     "fcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator),
#     "nlm": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
#     "bbb": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
#     "hmc": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=100, log_posterior_evaluator=get_inverse_posterior_evaluator),
#     "clfcnn": InverseConfig(inverse_params_min=inverse_params_min, inverse_params_max=inverse_params_max, chains=32, samples=10, burn_in=10, solution_samples=None, log_posterior_evaluator=get_inverse_posterior_evaluator),
# }

import torch
from neurodiffeq import diff
from neurodiffeq.conditions import IVP

from utils import BBBConfig, CLFCNNConfig, Equation, FCNNConfig, HMCConfig, NLMConfig

mu_Y_Glc_max = 0.253
mu_Y_Fru_max = 0.359
mu_LAB_max = 0.358
mu_AAB_EtOH_max = 0.380
mu_AAB_LA_max = 0.008
K_Y_Glc = 35.322
K_Y_Fru = 35.492
K_LAB_Glc = 37.966
K_AAB_EtOH = 16.056
K_AAB_LA = 2509.622
k_Y = 0.0333
k_LAB = 0.0054
k_AAB = 0.0069
Y_Glc_Y = 33.4
Y_Glc_LAB = 29.259
Y_Fru_Y = 41.105
Y_Glc_EtOH_Y = 7.436
Y_Fru_EtOH_Y = 5.927
Y_EtOH_AAB = 1298.07
Y_LA_LAB = 10.617
Y_LA_AAB = 1928.619
Y_Ac_LAB = 5.612
Y_EtOH_Ac_AAB = 103.056
Y_LA_Ac_AAB = 1427.225

coords_train_min = (0,)
coords_train_max = (72,)
coords_test_min = (0,)
coords_test_max = (144,)
bundle_parameters_min = ()
bundle_parameters_max = ()
non_bundle_parameters_min = ()
non_bundle_parameters_max = ()
inverse_params_min = ()
inverse_params_max = ()

initial_conditions = [51.963, 57.741, 0, 0, 0, 0.029180401, 0.007868827, 3.36634e-6]
conditions = [IVP(0, initial_conditions[0]), IVP(0, initial_conditions[1]), IVP(0, initial_conditions[2]),
              IVP(0, initial_conditions[3]), IVP(0, initial_conditions[4]), IVP(0, initial_conditions[5]),
              IVP(0, initial_conditions[6]), IVP(0, initial_conditions[7])]

def system(x, y, z, u, v, w, r, s, t):
    return [
        diff(x, t) - (- Y_Glc_Y * mu_Y_Glc_max * x * w / (x + K_Y_Glc) - Y_Glc_LAB * mu_LAB_max * x * r / (x + K_LAB_Glc)),
        diff(y, t) - (- Y_Fru_Y * mu_Y_Fru_max * y * w / (y + K_Y_Fru)),
        diff(z, t) - (Y_Glc_EtOH_Y * mu_Y_Glc_max * x * w / (x + K_Y_Glc) + Y_Fru_EtOH_Y * mu_Y_Fru_max *y * w / (y + K_Y_Fru) - Y_EtOH_AAB * mu_AAB_EtOH_max * z * s / (z + K_AAB_EtOH)),
        diff(u, t) - (Y_LA_LAB * mu_LAB_max * x * r / (x + K_LAB_Glc) - Y_LA_AAB * mu_AAB_LA_max * u * s / (u + K_AAB_LA * s)),
        diff(v, t) - (Y_Ac_LAB * mu_LAB_max * x * r / (x + K_LAB_Glc) + Y_EtOH_Ac_AAB * mu_AAB_EtOH_max * z * s / (z + K_AAB_EtOH) + Y_LA_Ac_AAB * mu_AAB_LA_max * u * s / (u + K_AAB_LA * s)),
        diff(w, t) - (mu_Y_Glc_max * x * w / (x + K_Y_Glc) + mu_Y_Fru_max * y * w / (y + K_Y_Fru) - k_Y * w * z),
        diff(r, t) - (mu_LAB_max * x * r / (x + K_LAB_Glc) - k_LAB * r * u),
        diff(s, t) - (mu_AAB_EtOH_max * z * s / (z + K_AAB_EtOH) + mu_AAB_LA_max * u * s / (u + K_AAB_LA * s) - k_AAB * s * v**2)
    ]

equation = Equation(coords_train_min=coords_train_min, coords_train_max=coords_train_max,
                    coords_test_min=coords_test_min, coords_test_max=coords_test_max,
                    bundle_parameters_min=bundle_parameters_min, bundle_parameters_max=bundle_parameters_max,
                    non_bundle_parameters_min=non_bundle_parameters_min, non_bundle_parameters_max=non_bundle_parameters_max,
                    system=system, system_bundle=None, loss_fn=None, analytic=None,
                    system_size=8, conditions=conditions, bundle_conditions=None,
                    int_eP=None, eP=None)

fcnn_config = FCNNConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=100_000, dimension_batch_size=64, lr=0.001)
nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=500, likelihood_std=.01, res_likelihood_std=.01)
bbb_config = BBBConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=20_000, dimension_batch_size=100, prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01)
hmc_config = HMCConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=10_000, tune_samples=1000, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01)
clfcnn_config = CLFCNNConfig(input_features=1, output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, step_iterations=[150_000, 100_000, 100_000, 150_000, 100_000], dimension_batch_size=64, coords_step_proportions=(0.2, 0.4, 0.6, 0.8, 1))

# fcnn_config = FCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=1_000, dimension_batch_size=64, lr=0.001)
# nlm_config = NLMConfig(prior_calibration_range=(.1, 2), prior_calibration_points=100, dimension_batch_size=500, likelihood_std=.01, res_likelihood_std=.01)
# bbb_config = BBBConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, iterations=500, dimension_batch_size=100, prior_std=1, lr=0.001, likelihood_std=.01, res_likelihood_std=.01)
# hmc_config = HMCConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, samples=20, tune_samples=10, chains=1, dimension_batch_size=100, target_acceptance_rate=0.95, prior_std=1, likelihood_std=.01, res_likelihood_std=.01)
# clfcnn_config = CLFCNNConfig(output_features=1, hidden_units=(32, 32), activation=torch.nn.Tanh, step_iterations=[150, 100, 100, 150, 100], dimension_batch_size=64, coords_step_proportions=(0.2, 0.4, 0.6, 0.8, 1))

methods_configs = {
    "fcnn": fcnn_config,
    "nlm": nlm_config,
    "bbb": bbb_config,
    "hmc": hmc_config,
    "clfcnn": clfcnn_config,
}

methods_configs_bundle = {}

inverse_configs = {
    "fcnn": None,
    "clfcnn": None,
    "nlm": None,
    "bbb": None,
    "hmc": None,
}
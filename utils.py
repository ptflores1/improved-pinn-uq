from dataclasses import dataclass
import os
import torch

@dataclass
class FCNNConfig:
    dimension_batch_size: int
    input_features: int
    output_features: int
    hidden_units: tuple
    activation: torch.nn.Module
    iterations: int
    lr: float
    device: str = "cuda"

    def __post_init__(self):
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", False) else "cpu"

@dataclass
class NLMConfig:
    dimension_batch_size: int
    prior_calibration_range: tuple
    likelihood_std: float
    res_likelihood_std: float
    prior_calibration_points: int
    eb_dimension_batch_size: int = 100
    device: str = "cpu"

    def __post_init__(self):
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", False) else "cpu"

@dataclass
class BBBConfig:
    dimension_batch_size: int
    input_features: int
    output_features: int
    hidden_units: tuple
    activation: torch.nn.Module
    iterations: int
    prior_std: float
    lr: float
    likelihood_std: float
    res_likelihood_std: float
    eb_dimension_batch_size: int = 100
    device: str = "cuda"

    def __post_init__(self):
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", False) else "cpu"

@dataclass
class HMCConfig:
    dimension_batch_size: int
    input_features: int
    output_features: int
    hidden_units: tuple
    activation: torch.nn.Module
    samples: int
    tune_samples: int
    chains: int
    target_acceptance_rate: float
    prior_std: float
    likelihood_std: float
    res_likelihood_std: float
    eb_dimension_batch_size: int = 100
    device: str = "cuda"
    step_size: float = None

    def __post_init__(self):
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", False) else "cpu"

@dataclass
class CLFCNNConfig:
    dimension_batch_size: int
    input_features: int
    output_features: int
    hidden_units: tuple
    activation: torch.nn.Module
    step_iterations: int
    coords_step_proportions: tuple
    device: str = "cuda"

    def __post_init__(self):
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", False) else "cpu"

@dataclass
class Equation:
    coords_train_min: tuple
    coords_train_max: tuple
    coords_test_min: tuple
    coords_test_max: tuple
    bundle_parameters_min: tuple
    bundle_parameters_max: tuple
    non_bundle_parameters_min: tuple
    non_bundle_parameters_max: tuple
    system: callable
    system_bundle: callable
    loss_fn: callable
    analytic: callable
    system_size: int
    conditions: list
    bundle_conditions: list
    int_eP: callable
    eP: callable
    eb_parameters_min: tuple = None
    eb_parameters_max: tuple = None

@dataclass
class InverseConfig:
    chains: int
    solution_samples: int
    log_posterior_evaluator: callable
    inverse_params_min: tuple
    inverse_params_max: tuple
    burn_in: int = 1000
    samples: int = 10_000
    device: str = "cuda"

    def __post_init__(self):
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", False) else "cpu"

def build_experiment_name(args):
    experiment_name = args.equation + "_" + args.method
    experiment_name += "_res" if args.res_loss else ""
    experiment_name += "_ov" if args.output_variance else ""
    experiment_name = "bundle_" + experiment_name if args.bundle else experiment_name
    experiment_name += "_eb" if args.errorbounds else ""
    return experiment_name
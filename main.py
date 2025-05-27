import os
import importlib
import argparse
import torch
from neurodiffeq.utils import set_tensor_type
#from services.wandb import init_wandb_run

from utils import build_experiment_name

equation_datasets = {
    "lcdm": ["cc", "cc_lcdm_syn"],
    "cpl": ["cc", "cc_cpl_syn"],
    "cpl_params": ["cc", "cc_cpl_syn"],
    "quintessence": ["cc", "cc_quint_syn"],
    "hs": ["cc", "cc_hs_syn"],
    "lab": ["lab", "lab_syn"],
    "cocoa": []
}

parser = argparse.ArgumentParser(description='Run a method on a given equation.')

parser.add_argument("equation", metavar="equation", type=str, help='The equation to run the method on. Options are ["lcdm", "cpl", "quintessence", "hs", "lab", "cocoa"]', choices=["lcdm", "cpl", "cpl_params", "quintessence", "hs", "lab", "cocoa"])
parser.add_argument("method", metavar="method", type=str, help='The method to run on the equation. Options are ["fcnn", "nlm", "bbb", "hmc"]', choices=["fcnn", "nlm", "bbb", "hmc", "clfcnn"])
parser.add_argument("-eb", "--errorbounds", action="store_true", help="Whether to use error bounds.")
parser.add_argument("-b", "--bundle", action="store_true", help="Whether to run the bundle solutions version.")
parser.add_argument("-i", "--inverse", type=int, choices=[0, 1, 2], help="Whether to run inverse sampling. 0 for not doing inverse, 1 for doing forward and inverse, 2 just for inverse.", default=0, metavar="")
parser.add_argument("-bm", "--basemethod", type=str, help="The base method to use for a bayesian method. Options are [fcnn, clfcnn]", choices=["fcnn", "clfcnn"], default="fcnn")
parser.add_argument("-gpu", "--gpu", action="store_true", help="Wether to use gpus.")
parser.add_argument("-ov", "--output_variance", action="store_true", help="Wether to output variances.")
parser.add_argument("-ids", "--inverse_dataset", default="", help="Dataset for inverse estimation.", choices=["cc", "cc_lcdm_syn", "cc_cpl_syn", "cc_quint_syn", "cc_hs_syn", "lab", "lab_syn"])
parser.add_argument("-rl", "--res_loss", action="store_true", help="Whether to use residual loss instead of deterministic solution as targets.")

args = parser.parse_args()
if args.errorbounds and args.equation in ["quintessence", "hs", "lab", "cocoa"]:
    parser.error("Error bounds are only supported for lcdm and cpl equations.")
if args.method == "fcnn" and args.errorbounds:
    parser.error("Error bounds are not supported for fcnn method.")
if not args.bundle and args.inverse in [1, 2]:
    parser.error("Inverse sampling is only supported for bundle solutions.")
if args.output_variance and args.method in ["fcnn", "clfcnn", "nlm"]:
    parser.error(f"Output variance is not supported for {args.method}.")
if args.inverse_dataset and args.inverse_dataset not in equation_datasets[args.equation]:
    parser.error(f"Dataset {args.inverse_dataset} is not available for the equation {args.equation}.")
if args.inverse in [1, 2] and args.inverse_dataset == "":
    parser.error("Inverse dataset is required for inverse sampling.")

if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", False):
    print("GPU Available")

equation_data = importlib.import_module("equations." + args.equation)
equation = equation_data.equation
method_config = equation_data.methods_configs[args.method]
method_config_bundle = equation_data.methods_configs_bundle.get(args.method, None)
inverse_config = equation_data.inverse_configs.get(args.method, None)
method = importlib.import_module("methods." + args.method)
experiment_name = build_experiment_name(args)
#init_wandb_run(equation, method_config, method_config_bundle, inverse_config, experiment_name, args)

print("Experiment:", experiment_name)
if args.bundle:
    print("METHOD DEVICE:", method_config_bundle.device)
    set_tensor_type(method_config_bundle.device, 64)
    method.run_bundle(equation, method_config_bundle, inverse_config, experiment_name, args=args)
else:
    print("METHOD DEVICE:", method_config.device)
    set_tensor_type(method_config.device, 64)
    method.run_forward(equation, method_config, experiment_name, args=args)
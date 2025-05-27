import importlib
import argparse
import torch

parser = argparse.ArgumentParser(description='Generate plot data.')

parser.add_argument("equation", metavar="equation", type=str, help='The equation to run the method on. Options are ["lcdm", "cpl", "quintessence", "hs", "lab", "cocoa"]', choices=["lcdm", "cpl", "cpl_params", "quintessence", "hs", "lab", "cocoa"])
parser.add_argument("-eb", "--errorbounds", action="store_true", help="Whether to use error bounds.")
parser.add_argument("-b", "--bundle", action="store_true", help="Whether to run the bundle solutions version.")
parser.add_argument("-bf", "--best_fit", action="store_true", help="Whether to run the best fit data.")
parser.add_argument("--domain_type", type=str)

args = parser.parse_args()
equation_module = importlib.import_module("plotters." + args.equation)

torch.set_default_device("cpu")
if args.bundle:
    if args.errorbounds:
        print("Getting bundle plot data with error bounds")
        equation_module.get_bundle_plot_data(force=True, eb=True, domain_type=args.domain_type)
    else:
        print("Getting bundle plot data")
        equation_module.get_bundle_plot_data(force=True, domain_type=args.domain_type)
elif args.best_fit:
    if args.errorbounds:
        print("Getting best fit plot data with error bounds")
        equation_module.get_best_fit_plot_data(force=True, eb=True)
    else:
        print("Getting best fit plot data")
        equation_module.get_best_fit_plot_data(force=True)
else:
    if args.errorbounds:
        print("Getting plot data with error bounds")
        equation_module.get_plot_data(force=True, eb=True, domain_type=args.domain_type)
    else:
        print("Getting plot data")
        equation_module.get_plot_data(force=True, domain_type=args.domain_type)
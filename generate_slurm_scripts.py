import os
import jinja2
from collections import defaultdict
from itertools import product

os.system("rm slurm/*.sh")

env = jinja2.Environment(loader=jinja2.FileSystemLoader('./slurm'))
template = env.get_template('template.sh.jinja2')

def build_experiment_name(args):
    experiment_name = args["equation"]
    experiment_name += "_inverse_" + args["inverse_ds"] if args.get("inverse", False) else ""
    experiment_name += "_" + args["method"]
    experiment_name += "_res" if args["res_loss"] else ""
    experiment_name += "_ov" if args["ov"] else ""
    experiment_name += "_bundle" if args["bundle"] else ""
    experiment_name += "_eb" if args["errorbounds"] else ""
    return experiment_name

methods = ["fcnn", "clfcnn", "nlm", "bbb", "hmc"]
equations = ["lcdm", "cpl", "quintessence", "hs", "lab", "cocoa"]#, "cpl_params"]
bundle_options = ["-b", ""]
errorbounds_options = ["-eb", ""]
output_variance_options = ["-ov", ""]
inverse_datasets = ["cc", "lab", "cc_lcdm_syn", "cc_cpl_syn", "cc_quint_syn", "cc_hs_syn", "lab_syn"]
loss_options = ["", "-rl"]
equation_datasets = {
    "lcdm": ["cc", "cc_lcdm_syn"],
    "cpl": ["cc", "cc_cpl_syn"],
    #"cpl_params": ["cc", "cc_cpl_syn"],
    "quintessence": ["cc", "cc_quint_syn"],
    "hs": ["cc", "cc_hs_syn"],
    "lab": ["lab", "lab_syn"],
    "cocoa": []
}

def valid_experiment(x):
    if x[0] in ["quintessence", "hs", "lab", "cocoa"] and x[3] == "-eb":
        return False
    if x[0] == "cocoa" and x[2] == "-b":
        return False
    if x[1] in ["fcnn", "clfcnn"] and x[3] == "-eb":
        return False
    if x[1] == "clfcnn" and x[0] not in ["hs", "cocoa"]:
        return False
    if x[1] in ["fcnn", "clfcnn", "nlm"] and x[4] == "-ov":
        return False
    if x[4] == "-ov" and x[3] == "-eb":
        return False
    if x[5] != "" and x[5] not in equation_datasets[x[0]]:
        return False
    if x[6] == "-rl" and x[1] not in ["hmc", "bbb"]:
        return False
    if x[6] == "-rl" and x[4] == "-ov":
        return False
    if x[6] == "-rl" and x[3] == "-eb":
        return False
    return True

args = list(map(list, filter(valid_experiment, product(equations, methods, bundle_options, errorbounds_options, output_variance_options, inverse_datasets, loss_options))))
groups = defaultdict(list)
experiment_names = []
experiments = []

for i, command_args in enumerate(args):
    gpus = 0 if command_args[1] in ["hmc", "nlm"] else (1 if "-b" in command_args else 0)
    cpus = 16 if command_args[1] == "nlm" else 4
    
    exp_name = build_experiment_name({"equation": command_args[0], "method": command_args[1], "bundle": "-b" in command_args, "errorbounds": "-eb" in command_args, "ov": "-ov" in command_args, "res_loss": "-rl" in command_args})
    if exp_name not in experiment_names:
        experiment_names.append(exp_name)
        experiments.append((exp_name, {"equation": command_args[0], "method": command_args[1], "bundle": "-b" in command_args, "errorbounds": "-eb" in command_args, "ov": "-ov" in command_args, "inverse": False, "res_loss": "-rl" in command_args}))
    key = command_args[0] + ("_bundle" if "-b" in command_args else "")
    values = {
        "job_name": exp_name,
        "group_name": key,
        "cpus": cpus,
        "time": "8-00:00:00",
        "gpus": 0,
        "command": " ".join(command_args).replace(command_args[5], "") + (" --gpu" if gpus > 0 else "") + (" -bm=clfcnn" if command_args[0] in ["cocoa", "hs"] else ""),
        "email": True, #i == len(args)-1 or args[i+1][0] != args[i][0],
        "exclude": False # command_args[1] == "hmc"
    }

    groups[key].append(values)

    with open(f"slurm/{exp_name}.sh", "w") as f:
        f.write(template.render(values))

    values["job_name"] += "_gpu"
    values["gpus"] = 1
    with open(f"slurm/{exp_name}_gpu.sh", "w") as f:
        f.write(template.render(values))

    # Inverse only
    if "-b" in command_args:
        gpus = 0
        command_args += ["-i=2"]
        # exp_name += "_inverse" + "_" + command_args[5]
        exp_name = build_experiment_name({"equation": command_args[0], "method": command_args[1], "bundle": "-b" in command_args, "errorbounds": "-eb" in command_args, "ov": "-ov" in command_args, "inverse": True, "res_loss": "-rl" in command_args, "inverse_ds": command_args[5]})
        if exp_name not in experiment_names:
            experiment_names.append(exp_name)
            experiments.append((exp_name, {"equation": command_args[0], "method": command_args[1], "bundle": "-b" in command_args, "errorbounds": "-eb" in command_args, "ov": "-ov" in command_args, "inverse": True, "res_loss": "-rl" in command_args}))
        command_args[5] = f"-ids={command_args[5]}"
        values = {
            "job_name": exp_name,
            "group_name": key + "_inverse",
            "cpus": cpus,
            "time": "8-00:00:00",
            "gpus": gpus,
            "command": " ".join(command_args) + (" --gpu" if gpus > 0 else "") + (" -bm=clfcnn" if command_args[0] in ["cocoa", "hs"] else ""),
            "email": True, #i == len(args)-1 or args[i+1][0] != args[i][0],
            "exclude": False
        }

        with open(f"slurm/{exp_name}.sh", "w") as f:
            f.write(template.render(values))
            


det_experiments = list(filter(lambda ex: not ex[1]["res_loss"], experiments))

forward_exp = list(filter(lambda ex: not ex[1]["bundle"] and not ex[1]["inverse"] and not ex[1]["ov"], det_experiments))
bundle_exp = list(filter(lambda ex: ex[1]["bundle"] and not ex[1]["ov"] and not ex[1]["inverse"], det_experiments))
inverse_exp = list(filter(lambda ex: ex[1]["inverse"] and not ex[1]["ov"] and "syn" not in ex[0], det_experiments))
inverse_syn_exp = list(filter(lambda ex: ex[1]["inverse"] and not ex[1]["ov"] and "syn" in ex[0], det_experiments))
forward_ov_exp = list(filter(lambda ex: not ex[1]["bundle"] and not ex[1]["inverse"] and ex[1]["ov"], det_experiments))
bundle_ov_exp = list(filter(lambda ex: ex[1]["bundle"] and ex[1]["ov"] and not ex[1]["inverse"], det_experiments))
inverse_ov_exp = list(filter(lambda ex: ex[1]["inverse"] and ex[1]["ov"], det_experiments))

res_experiments = list(filter(lambda ex: ex[1]["res_loss"], experiments))
res_forward_exp = list(filter(lambda ex: not ex[1]["bundle"] and not ex[1]["inverse"] and not ex[1]["ov"], res_experiments))
res_bundle_exp = list(filter(lambda ex: ex[1]["bundle"] and not ex[1]["ov"] and not ex[1]["inverse"], res_experiments))
res_inverse_exp = list(filter(lambda ex: ex[1]["inverse"] and not ex[1]["ov"], res_experiments))
res_forward_ov_exp = list(filter(lambda ex: not ex[1]["bundle"] and not ex[1]["inverse"] and ex[1]["ov"], res_experiments))
res_bundle_ov_exp = list(filter(lambda ex: ex[1]["bundle"] and ex[1]["ov"] and not ex[1]["inverse"], res_experiments))
res_inverse_ov_exp = list(filter(lambda ex: ex[1]["inverse"] and ex[1]["ov"], res_experiments))

def group_by_eq(array):
    groups = {}
    for eq in equations:
        groups[eq] = list(filter(lambda ex: ex[1]["equation"] == eq, array))
    return groups


forward_eq = group_by_eq(forward_exp)
bundle_eq = group_by_eq(bundle_exp)
inverse_eq = group_by_eq(inverse_exp)
inverse_syn_eq = group_by_eq(inverse_syn_exp)
forward_ov_eq = group_by_eq(forward_ov_exp)
bundle_ov_eq = group_by_eq(bundle_ov_exp)
inverse_ov_eq = group_by_eq(inverse_ov_exp)


for name, exps in  zip(["forward", "bundle", "inverse", "inverse_syn", "forward_ov", "bundle_ov", "inverse_ov", "forward_res", "bundle_res", "inverse_res", "forward_ov_res", "bundle_ov_res", "inverse_ov_res"], [forward_exp, bundle_exp, inverse_exp, inverse_syn_exp, forward_ov_exp, bundle_ov_exp, inverse_ov_exp, res_forward_exp, res_bundle_exp, res_inverse_exp, res_forward_ov_exp, res_bundle_ov_exp, res_inverse_ov_exp]):
    with open(f"slurm/run_{name}.sh", "w") as f:
        for item in exps:
            f.write(f"sbatch slurm/{item[0]}.sh\n")

for method_name, grouped_exps in  zip(["forward", "bundle", "inverse", "inverse_syn", "forward_ov", "bundle_ov", "inverse_ov"], [forward_eq, bundle_eq, inverse_eq, inverse_syn_eq, forward_ov_eq, bundle_ov_eq, inverse_ov_eq]):
    for eq in equations:
        name = f"{eq}_{method_name}"
        exps = grouped_exps[eq]
        with open(f"slurm/run_{name}.sh", "w") as f:
            for item in exps:
                f.write(f"sbatch slurm/{item[0]}.sh\n")


template_plot_data = env.get_template('template_plot_data.sh.jinja2')
with open("slurm/run_all_plot_data.sh", "w") as f2:
    with open("slurm/run_all_plot_data_best_fit.sh", "w") as f3:
        for eq in equations:
            for eb in ["", "-eb"]:
                if eq in ["quintessence", "hs", "lab", "cocoa"] and eb == "-eb":
                    continue
                for b in ["", "-b"]:
                    if eq == "cocoa" and b == "-b":
                        continue
                    for dt in ["test", "train", "ood"]:
                        file_name = f"plot_data_{eq}"
                        file_name += f"_{eb.replace('-', '')}" if eb != "" else ""
                        file_name += f"_{b.replace('-', '')}" if b != "" else ""
                        file_name += f"_{dt}"
                        with open(f"slurm/{file_name}.sh", "w") as f:
                            f.write(template_plot_data.render({"job_name": f"plot_data_{eq}_{eb}_{b}", "logs_name": f"plot_data_{eq}_{eb}_{b}_{dt}", "equation": eq, "bundle": b, "eb": eb, "dt": dt}))
                        f2.write(f"sbatch slurm/{file_name}.sh\n")

                file_name = f"plot_data_{eq}_best_fit"
                file_name += f"_{eb.replace('-', '')}" if eb != "" else ""
                with open(f"slurm/{file_name}.sh", "w") as f:
                    f.write(template_plot_data.render({"job_name": file_name, "logs_name": file_name, "equation": eq, "bundle": "", "eb": eb, "bf": "-bf"}))
                f3.write(f"sbatch slurm/{file_name}.sh\n")

template_plot_data = env.get_template('template_plot.sh.jinja2')
with open("slurm/run_all_plots.sh", "w") as f2:
    for eq in equations + ["mixed_equations", "inverse"]:
        file_name = f"plot_{eq}"
        with open(f"slurm/{file_name}.sh", "w") as f:
            f.write(template_plot_data.render({"job_name": f"plot_eq_{eq}", "equation": eq}))
        f2.write(f"sbatch slurm/{file_name}.sh\n")


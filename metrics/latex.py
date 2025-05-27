import os
import dill
import jinja2
import numpy as np

env = jinja2.Environment(loader=jinja2.FileSystemLoader('./metrics_texs/templates'))

def recursive_dict_update(original, param):
    for key in param.keys():
        if type(param[key]) == dict and key in original:
            recursive_dict_update(original[key], param[key])
        else:
            original[key] = param[key]

def recursive_round(d, round_digits):
    for key in d.keys():
        if type(d[key]) == dict:
            recursive_round(d[key], round_digits)
        else:
            d[key] = round(d[key], round_digits) if isinstance(d[key], float) else d[key]
            d[key] = f'{d[key]:.2e}' if isinstance(d[key], float) and abs(d[key]) >= 100000 else d[key]

def flatten_dict(d, key_prefix="", new_dict=None):
    if new_dict is None:
        new_dict = {}
    for key in d:
        new_prefix = key_prefix + f"_{key}" if key_prefix != "" else key
        if isinstance(d[key], dict):
            flatten_dict(d[key], new_prefix, new_dict)
        else:
            new_dict[new_prefix] = d[key]
    return new_dict

def make_bold(d):
    for metric in ["mre", "mae", "rpd", "std_mae", "std_mre", "std_rpd", "residual", "calibration", "total_nan_proportion", "agg_nan_proportion"]:
        for bundle in ["forward", "bundle"]:
            best_keys = None
            best_value = float("inf")
            for method in ["FCNN", "NLM", "BBB", "HMC"]:
                for eb in ["eb", "no_eb"]:
                    key = f"{method}_{bundle}_{eb}_{metric}"
                    if key not in d: continue
                    if not isinstance(d[key], float): continue
                    if d[key] < best_value:
                        best_value = d[key]
                        best_keys = key
            if best_keys is not None:
                d[best_keys] = "$\mathbf{" + str(d[best_keys]) + "}$"

def safe_mean(arr):
    try:
        return np.mean(arr)
    except np.core._exceptions.UFuncTypeError:
        return "-"

def parse_eq_metrics(data, name):
    eb_key = "eb" if "eb" in name else "no_eb"
    bundle_key = "bundle" if "bundle" in name else "forward"

    new_data = {}
    for method in ["FCNN", "NLM", "BBB", "HMC"]:
        new_data[method] = {
            bundle_key: {
                eb_key: {
                    "mre": np.mean([data[method][i]["mre"] for i in range(len(data[method]))]),
                    "mae": np.mean([data[method][i]["mae"] for i in range(len(data[method]))]),
                    "rpd": np.mean([data[method][i]["rpd"] for i in range(len(data[method]))]),
                    "std_mae": safe_mean([data[method][i].get("std_mae", "-") for i in range(len(data[method]))]),
                    "std_mre": safe_mean([data[method][i].get("std_mre", "-") for i in range(len(data[method]))]),
                    "std_rpd": safe_mean([data[method][i].get("std_rpd", "-") for i in range(len(data[method]))]),
                    "residual": safe_mean([data[method][i].get("mean_residual", "-") for i in range(len(data[method]))]),

                    "rms_cal": safe_mean([data[method][i].get("avg_calibration", {}).get("rms_cal", "-") for i in range(len(data[method]))]),
                    "ma_cal": safe_mean([data[method][i].get("avg_calibration", {}).get("ma_cal", "-") for i in range(len(data[method]))]),
                    "miscal_area": safe_mean([data[method][i].get("avg_calibration", {}).get("miscal_area", "-") for i in range(len(data[method]))]),

                    "sharpness": safe_mean([data[method][i].get("sharpness", {}).get("sharp", "-") for i in range(len(data[method]))]),

                    "nll": safe_mean([data[method][i].get("scoring_rule", {}).get("nll", "-") for i in range(len(data[method]))]),
                    "crps": safe_mean([data[method][i].get("scoring_rule", {}).get("crps", "-") for i in range(len(data[method]))]),
                    "check": safe_mean([data[method][i].get("scoring_rule", {}).get("check", "-") for i in range(len(data[method]))]),
                    "interval": safe_mean([data[method][i].get("scoring_rule", {}).get("interval", "-") for i in range(len(data[method]))]),

                    "acc_mae": safe_mean([data[method][i].get("accuracy", {}).get("mae", "-") for i in range(len(data[method]))]),
                    "acc_rmse": safe_mean([data[method][i].get("accuracy", {}).get("rmse", "-") for i in range(len(data[method]))]),
                    "acc_mdae": safe_mean([data[method][i].get("accuracy", {}).get("mdae", "-") for i in range(len(data[method]))]),
                    "acc_marpd": safe_mean([data[method][i].get("accuracy", {}).get("marpd", "-") for i in range(len(data[method]))]),
                    "acc_r2": safe_mean([data[method][i].get("accuracy", {}).get("r2", "-") for i in range(len(data[method]))]),
                    "acc_corr": safe_mean([data[method][i].get("accuracy", {}).get("corr", "-") for i in range(len(data[method]))]),

                    "re_q10": safe_mean([data[method][i].get("re_q10", "-") for i in range(len(data[method]))]),
                    "re_q20": safe_mean([data[method][i].get("re_q20", "-") for i in range(len(data[method]))]),
                    "re_q30": safe_mean([data[method][i].get("re_q30", "-") for i in range(len(data[method]))]),
                    "re_q40": safe_mean([data[method][i].get("re_q40", "-") for i in range(len(data[method]))]),
                    "re_q50": safe_mean([data[method][i].get("re_q50", "-") for i in range(len(data[method]))]),
                    "re_q60": safe_mean([data[method][i].get("re_q60", "-") for i in range(len(data[method]))]),
                    "re_q70": safe_mean([data[method][i].get("re_q70", "-") for i in range(len(data[method]))]),
                    "re_q80": safe_mean([data[method][i].get("re_q80", "-") for i in range(len(data[method]))]),
                    "re_q90": safe_mean([data[method][i].get("re_q90", "-") for i in range(len(data[method]))]),
                    "re_q100": safe_mean([data[method][i].get("re_q100", "-") for i in range(len(data[method]))]),

                    "total_nan_proportion": safe_mean([data[method][i].get("total_nan_proportion", "-") for i in range(len(data[method]))]),
                    "agg_nan_proportion": safe_mean([data[method][i].get("agg_nan_proportion", "-") for i in range(len(data[method]))]),
                }
            }
        }
    return new_data

def load_metric_data(metricquee, equation, dt="", eb=False):
    data = {}
    dt_str = f"_{dt}" if dt in ["train", "ood"] else ""
    with open(f'metric_results/{metricquee}_{equation}_forward{dt_str}.dill', 'rb') as f:
        recursive_dict_update(data, parse_eq_metrics(dill.load(f), f"{metricquee}_{equation}_forward{dt_str}"))
    with open(f'metric_results/{metricquee}_{equation}_bundle{dt_str}.dill', 'rb') as f:
        recursive_dict_update(data, parse_eq_metrics(dill.load(f), f"{metricquee}_{equation}_bundle{dt_str}"))
    if eb:
        with open(f'metric_results/{metricquee}_{equation}_eb_forward{dt_str}.dill', 'rb') as f:
            recursive_dict_update(data, parse_eq_metrics(dill.load(f), f"{metricquee}_{equation}_eb_forward{dt_str}"))
        with open(f'metric_results/{metricquee}_{equation}_eb_bundle{dt_str}.dill', 'rb') as f:
            recursive_dict_update(data, parse_eq_metrics(dill.load(f), f"{metricquee}_{equation}_eb_bundle{dt_str}"))
    return data

def make_metrics_table(equation, metricquee, dt, eb=False):
    tex_template = f'metrics_{metricquee}_lcdm_cpl.tex.jinja2' if eb else f'metrics_{metricquee}_quint_hs.tex.jinja2'
    template = env.get_template(tex_template)

    data = load_metric_data(metricquee, equation, dt=dt, eb=eb)

    recursive_round(data, 3)
    data = flatten_dict(data)
    make_bold(data)
    with open(f"metrics_texs/metrics_{metricquee}_{equation}_{dt}.tex", "w") as f:
        f.write(template.render(**data))
    os.system(f"/home/ptflores1/storage/tex-installation/bin/x86_64-linux/pdflatex --output-directory=./metric_results ./metrics_texs/metrics_{metricquee}_{equation}_{dt}.tex")
    os.system("rm metric_results/*.aux metric_results/*.log")

def make_metrics_table_dts_bundle(equation, metricquee, eb=False):
    tex_template = f'metrics_{metricquee}_lcdm_cpl_dts_bundle.tex.jinja2' if eb else f'metrics_{metricquee}_quint_hs_dts_bundle.tex.jinja2'
    template = env.get_template(tex_template)
    data = {}
    data["test"] = load_metric_data(metricquee, equation, dt="", eb=eb)
    data["train"] = load_metric_data(metricquee, equation, dt="train", eb=eb)
    data["ood"] = load_metric_data(metricquee, equation, dt="ood", eb=eb)

    recursive_round(data, 3)
    data = flatten_dict(data)
    make_bold(data)
    with open(f"metrics_texs/metrics_{metricquee}_{equation}_bundle.tex", "w") as f:
        f.write(template.render(**data))
    os.system(f"/home/ptflores1/storage/tex-installation/bin/x86_64-linux/pdflatex --output-directory=./metric_results ./metrics_texs/metrics_{metricquee}_{equation}_bundle.tex")
    os.system("rm metric_results/*.aux metric_results/*.log")

def make_metrics_table_dts_forward(equation, metricquee, eb=False):
    tex_template = f'metrics_{metricquee}_lcdm_cpl_dts_forward.tex.jinja2' if eb else f'metrics_{metricquee}_quint_hs_dts_forward.tex.jinja2'
    template = env.get_template(tex_template)

    data = {}
    data["test"] = load_metric_data(metricquee, equation, dt="", eb=eb)
    data["train"] = load_metric_data(metricquee, equation, dt="train", eb=eb)
    data["ood"] = load_metric_data(metricquee, equation, dt="ood", eb=eb)

    recursive_round(data, 3)
    data = flatten_dict(data)
    make_bold(data)
    with open(f"metrics_texs/metrics_{metricquee}_{equation}_forward.tex", "w") as f:
        f.write(template.render(**data))
    os.system(f"/home/ptflores1/storage/tex-installation/bin/x86_64-linux/pdflatex --output-directory=./metric_results ./metrics_texs/metrics_{metricquee}_{equation}_forward.tex")
    os.system("rm metric_results/*.aux metric_results/*.log")

def make_metrics_table_all_eqs(metricquee):
    tex_template = f'metrics_{metricquee}.tex.jinja2'
    template = env.get_template(tex_template)

    data = {}
    for eq in ["lcdm", "cpl", "hs", "quintessence"]:
        data[eq] = load_metric_data(metricquee, eq, dt="", eb=(eq in ["lcdm", "cpl"]))

    recursive_round(data, 3)
    data = flatten_dict(data)
    make_bold(data)
    with open(f"metrics_texs/metrics_{metricquee}.tex", "w") as f:
        f.write(template.render(**data))
    os.system(f"/home/ptflores1/storage/tex-installation/bin/x86_64-linux/pdflatex --output-directory=./metric_results ./metrics_texs/metrics_{metricquee}.tex")
    os.system("rm metric_results/*.aux metric_results/*.log")

def make_all_metrics_table_all_eqs(metricquee, dt=""):
    tex_template = f'metrics_all_{metricquee}.tex.jinja2'
    template = env.get_template(tex_template)

    data = {}
    for eq in ["lcdm", "cpl", "hs", "quintessence"]:
        data[eq] = load_metric_data(metricquee, eq, dt=dt, eb=(eq in ["lcdm", "cpl"]))

    recursive_round(data, 3)
    data = flatten_dict(data)
    make_bold(data)
    with open(f"metrics_texs/metrics_all_{metricquee}_{dt}.tex", "w") as f:
        f.write(template.render(**data))
    os.system(f"/home/ptflores1/storage/tex-installation/bin/x86_64-linux/pdflatex --output-directory=./metric_results ./metrics_texs/metrics_all_{metricquee}_{dt}.tex")
    os.system("rm metric_results/*.aux metric_results/*.log")

def make_nan_proportion_table(equation, eb=False):
    tex_template = 'nan_table_1.tex.jinja2' if eb else 'nan_table_2.tex.jinja2'
    template = env.get_template(tex_template)

    data = load_metric_data("hubble", equation, eb)

    recursive_round(data, 3)
    data = flatten_dict(data)
    make_bold(data)
    with open(f"metrics_texs/nan_proportion_{equation}.tex", "w") as f:
        f.write(template.render(**data))
    os.system(f"/home/ptflores1/storage/tex-installation/bin/x86_64-linux/pdflatex  --output-directory=./metric_results ./metrics_texs/nan_proportion_{equation}.tex")
    os.system("rm metric_results/*.aux metric_results/*.log")

def load_inverse_data(equation, param_names, eb=False):
    fcnn_means = np.load(f"checkpoints/inverse_samples_bundle_{equation}_fcnn_cc.npy").mean(axis=0)
    bbb_means = np.load(f"checkpoints/inverse_samples_bundle_{equation}_bbb_cc.npy").mean(axis=0)
    nlm_means = np.load(f"checkpoints/inverse_samples_bundle_{equation}_nlm_cc.npy").mean(axis=0)
    hmc_means = np.load(f"checkpoints/inverse_samples_bundle_{equation}_hmc_cc.npy").mean(axis=0)

    fcnn_stds = np.load(f"checkpoints/inverse_samples_bundle_{equation}_fcnn_cc.npy").std(axis=0)
    bbb_stds = np.load(f"checkpoints/inverse_samples_bundle_{equation}_bbb_cc.npy").std(axis=0)
    nlm_stds = np.load(f"checkpoints/inverse_samples_bundle_{equation}_nlm_cc.npy").std(axis=0)
    hmc_stds = np.load(f"checkpoints/inverse_samples_bundle_{equation}_hmc_cc.npy").std(axis=0)

    means = [fcnn_means, bbb_means, nlm_means, hmc_means]
    stds = [fcnn_stds, bbb_stds, nlm_stds, hmc_stds]
    methods = ["FCNN", "BBB", "NLM", "HMC"]
    data = {}
    for mi, m in enumerate(methods):
        data[m] = {}
        data[m]["no_eb"] = {}
        for pmi, pm in enumerate(param_names):
            data[m]["no_eb"][pm] = {"mean": means[mi][pmi], "std": stds[mi][pmi]}
    if eb:
        bbb_means_eb = np.load(f"checkpoints/inverse_samples_bundle_{equation}_bbb_eb_cc.npy").mean(axis=0)
        nlm_means_eb = np.load(f"checkpoints/inverse_samples_bundle_{equation}_nlm_eb_cc.npy").mean(axis=0)
        hmc_means_eb = np.load(f"checkpoints/inverse_samples_bundle_{equation}_hmc_eb_cc.npy").mean(axis=0)

        bbb_stds_eb = np.load(f"checkpoints/inverse_samples_bundle_{equation}_bbb_eb_cc.npy").std(axis=0)
        nlm_stds_eb = np.load(f"checkpoints/inverse_samples_bundle_{equation}_nlm_eb_cc.npy").std(axis=0)
        hmc_stds_eb = np.load(f"checkpoints/inverse_samples_bundle_{equation}_hmc_eb_cc.npy").std(axis=0)

        means_eb = [bbb_means_eb, nlm_means_eb, hmc_means_eb]
        stds_eb = [bbb_stds_eb, nlm_stds_eb, hmc_stds_eb]
        for mi, m in enumerate(methods[1:]):
            data[m]["eb"] = {}
            for pmi, pm in enumerate(param_names):
                data[m]["eb"][pm] = {"mean": means_eb[mi][pmi], "std": stds_eb[mi][pmi]}

    return flatten_dict(data)

def make_inverse_table(equation):
    eb = "cpl" in equation or "lcdm" in equation
    param_names = {
        "lcdm": ["omega", "h0"],
        "cpl": ["w0", "w1", "omega", "h0"],
        "cpl_params": ["w0", "w1", "omega", "h0"],
        "quintessence": ["lambda", "omega", "h0"],
        "hs": ["b", "omega", "h0"],
    }
    data = load_inverse_data(equation, param_names=param_names[equation], eb=eb)
    template_name = equation if equation != "cpl_params" else "cpl"
    template = env.get_template(f"inverse_{template_name}.tex.jinja2")

    recursive_round(data, 2)
    with open(f"metrics_texs/inverse_{equation}.tex", "w") as f:
        f.write(template.render(**data))
    os.system(f"/home/ptflores1/storage/tex-installation/bin/x86_64-linux/pdflatex  --output-directory=./metric_results ./metrics_texs/inverse_{equation}.tex")
    os.system("rm metric_results/*.aux metric_results/*.log")

def make_inverse_table_all():
    template = env.get_template(f"inverse_all.tex.jinja2")
    param_names = {
        "lcdm": ["omega", "h0"],
        "cpl": ["w0", "w1", "omega", "h0"],
        "cpl_params": ["w0", "w1", "omega", "h0"],
        "quintessence": ["lambda", "omega", "h0"],
        "hs": ["b", "omega", "h0"],
    }
    data = {}
    for equation in ["lcdm", "cpl", "quintessence", "hs"]:
        data[equation] = load_inverse_data(equation, param_names=param_names[equation], eb=("cpl" in equation or "lcdm" in equation))
    data = flatten_dict(data)

    recursive_round(data, 2)
    with open(f"metrics_texs/inverse_all.tex", "w") as f:
        f.write(template.render(**data))
    os.system(f"/home/ptflores1/storage/tex-installation/bin/x86_64-linux/pdflatex  --output-directory=./metric_results ./metrics_texs/inverse_all.tex")
    os.system("rm metric_results/*.aux metric_results/*.log")

def make_re_quantiles_table(metricquee, dt):
    tex_template = f're_quantiles.tex.jinja2'
    template = env.get_template(tex_template)

    data = {}
    for eq in ["lcdm", "cpl", "hs", "quintessence"]:
        data[eq] = load_metric_data(metricquee, eq, dt=dt, eb=(eq in ["lcdm", "cpl"]))

    data = {key.replace("_train", "").replace("_test", "").replace("_ood", ""): val for key, val in data.items()}

    recursive_round(data, 3)
    data = flatten_dict(data)
    make_bold(data)
    with open(f"metrics_texs/re_quantiles_{dt}.tex", "w") as f:
        f.write(template.render(**data))
    os.system(f"/home/ptflores1/storage/tex-installation/bin/x86_64-linux/pdflatex --output-directory=./metric_results ./metrics_texs/re_quantiles_{dt}.tex")
    os.system("rm metric_results/*.aux metric_results/*.log")
        

if __name__ == "__main__":
    equations = ["lcdm", "cpl", "quintessence", "hs"]
    #make_metrics_table("lcdm", "eq", eb=True)
    #make_metrics_table("hs", "eq", eb=False)
    #make_metrics_table("quint", "eq", eb=False)
    # for eq in equations:
    #     for dt in ["", "train", "ood"]:
    #         make_metrics_table(eq, "hubble", eb=(eq in ["lcdm", "cpl", "cpl_params"]), dt=dt)
    #         make_metrics_table(eq, "eq", eb=(eq in ["lcdm", "cpl", "cpl_params"]), dt=dt)
    #         make_nan_proportion_table(eq, eb=(eq in ["lcdm", "cpl", "cpl_params"]))
    # for eq in ["lcdm", "cpl", "hs", "quintessence"]:
    #     make_inverse_table(eq)

    # for eq in ["lcdm", "cpl", "hs", "quintessence"]:
    #     make_metrics_table_dts_forward(eq, "eq", eb=(eq in ["lcdm", "cpl", "cpl_params"]))
    #     make_metrics_table_dts_bundle(eq, "eq", eb=(eq in ["lcdm", "cpl", "cpl_params"]))
    # make_metrics_table_all_eqs("eq")
    # make_inverse_table_all()
    for dt in ["", "train", "ood"]:
        make_re_quantiles_table("eq", dt=dt)
        make_all_metrics_table_all_eqs("eq", dt=dt)
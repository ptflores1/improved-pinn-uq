import numpy as np
import matplotlib.pyplot as plt
import os
import dill
#from .config import *

def build_exp_name(equation, eb=False, bundle=False, domain_type="test"):
    eb_str = "_eb" if eb else ""
    bundle_str = "_bundle" if bundle else "_forward"
    domain_str = "" if domain_type == "test" else ("_train" if domain_type == "train" else "_ood")
    return f"{equation}{eb_str}{bundle_str}{domain_str}"

def plot_loss(losses, title, save_path=None, log_scale=False):
    plt.title(title)
    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    if log_scale:
        plt.yscale("log")
    plt.tight_layout()
    if save_path:
        if log_scale:
            save_path = save_path.replace(".png", "_log.png")
        plt.savefig(save_path)

def dill_dec_old(file_path, file_path_eb=None):
    def wrapper(func):
        def wrapped_f(*args, **kwargs):
            force = kwargs.get("force", False)
            if "force" in kwargs:
                del kwargs["force"]
            final_file_path = file_path_eb if file_path_eb and kwargs.get("eb", False) else file_path
            if os.path.exists(final_file_path) and not force:
                print("Loading plot data from file", final_file_path)
                with open(final_file_path, "rb") as f:
                    return dill.load(f)
            data = func(*args, **kwargs)
            print("Saving plot data to file", final_file_path)
            with open(final_file_path, "wb") as f:
                dill.dump(data, f)
            return data
        return wrapped_f
    return wrapper

def dill_dec(equation, bundle=False):
    def wrapper(func):
        def f(eb=False, domain_type="test", force=False):
            exp_name = build_exp_name(equation, eb=eb, bundle=bundle, domain_type=domain_type)
            final_file_path = f"plot_data/{exp_name}.dill"
            if not force and os.path.exists(final_file_path):
                with open(final_file_path, "rb") as f:
                    return dill.load(f)
            data = func(eb=eb, domain_type=domain_type)
            print("Saving plot data to file", final_file_path)
            with open(final_file_path, "wb") as f:
                dill.dump(data, f)
            return data
        return f
    return wrapper


def ae(approximation, stds, real_values):
    """Mean Absolute Error"""
    error = np.abs(approximation - real_values)
    return error

def re(approximation, stds, real_values):
    """Mean Relative Error"""
    error = np.abs((approximation - real_values) / real_values)
    return error

def rpd(approximation, stds, real_values):
    zero_num = np.abs(2*(approximation - real_values)) == 0.
    rpd_ = np.abs(2*(approximation - real_values)) / (np.abs(approximation) + np.abs(real_values))
    rpd_[zero_num] = 0
    return rpd_

def std_ae(approximation, stds, real_values):
    abs_error = np.abs(approximation.ravel() - real_values.ravel())
    return ae(stds.ravel(), None, abs_error)

def std_re(approximation, stds, real_values):
    abs_error = np.abs(approximation.ravel() - real_values.ravel())
    return re(stds.ravel(), None, abs_error)

def std_rpd(approximation, stds, real_values):
    abs_error = np.abs(approximation.ravel() - real_values.ravel())
    return rpd(stds.ravel(), None, abs_error)

error_metrics = {
    "ae": ae,
    "re": re,
    "rpd": rpd,
    "std_ae": std_ae,
    "std_re": std_re,
    "std_rpd": std_rpd,
}
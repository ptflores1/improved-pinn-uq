import torch
import numpy as np
from tqdm import tqdm

eps = torch.sqrt(torch.tensor(torch.finfo().eps)).item()


def inf_norm(f, left, right, points=10, params=None):
    t = torch.linspace(left, right, points).reshape(-1, 1)
    params = [torch.ones(10).reshape(-1, 1) * p for p in params] if params is not None else None
    if params is not None:
        return float(abs(f(t, *params)).max())
    return float(abs(f(t)).max())


def integral_bound(f_inf, left, right, int_eP, params=None):
    if params is not None:
        b = f_inf * (int_eP(right, *params) - int_eP(left, *params))
    else:
        b = f_inf * (int_eP(right) - int_eP(left))
    assert b >= 0
    return b


def get_bound_until(f, t0, tn, n, int_eP, eP, n_per_interval=10, param=None):
    nodes = torch.linspace(t0, tn, n + 1)
    assert len(nodes) >= 2
    params = [torch.ones(10).reshape(-1, 1) * p for p in param] if param is not None else None
    params = param

    ts_list = []
    bounds = []
    base_bound = 0
    for i, n in enumerate(nodes[:-1]):
        left, right = n, nodes[i + 1]
        f_inf = inf_norm(f, left, right, params=params)
        ts = torch.linspace(left, right, n_per_interval)
        for t in ts:
            b = base_bound + integral_bound(f_inf, left, t, int_eP=int_eP, params=params)
            bounds.append(b)
        base_bound += integral_bound(f_inf, left, right, int_eP=int_eP, params=params)
        ts_list.append(ts)

    ts = torch.cat(ts_list)
    if params is not None:
        bounds = torch.tensor(bounds) / eP(ts, *params)
    else:
        bounds = torch.tensor(bounds) / eP(ts)
    return ts, bounds

def get_bounds_for_params(params, f, t0, tn, n, int_eP, eP, n_per_interval=10):
    bounds = []
    for p in tqdm(params, "Error bounds"):
        ts, bs = get_bound_until(f, t0, tn, n, int_eP, eP, n_per_interval=n_per_interval, param=p)
        bounds.append(bs)
    bounds = torch.from_numpy(np.array([b.cpu() for b in bounds]))
    return ts, bounds
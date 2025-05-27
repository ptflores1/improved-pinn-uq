import dill
import numpy as np
import uncertainty_toolbox as uct
import equations.lcdm as lcdm
import equations.cpl as cpl
import equations.quintessence as quint
import equations.hs as hs
import plotters.lcdm as plcdm
import plotters.cpl as pcpl
import plotters.cpl_params as pcpl_params
import plotters.quintessence as pquint
import plotters.hs as phs
from plotters.utils import build_exp_name

def mae(approximation, real_values):
    """Mean Absolute Error"""
    error = np.abs(approximation.ravel() - real_values.ravel())
    return error.mean()

def mre(approximation, real_values):
    """Mean Relative Error"""
    error = np.abs((approximation.ravel() - real_values.ravel()) / real_values.ravel())
    mask = real_values.ravel() != 0
    return error[mask].mean()

def re_quantiles(approximation, real_values):
    error = np.abs((approximation.ravel() - real_values.ravel()) / real_values.ravel())
    mask = real_values.ravel() != 0
    error = error[mask]
    error = np.sort(error)

    N = error.shape[0] - 1
    quantiles_idx = [int(.1 * i * N) for i in range(1, 11)]
    return {f"re_q{(i+1)*10}": error[qi] for i, qi in enumerate(quantiles_idx)}

def uct_metrics(approximation, stds, real_values):
    metrics = uct.get_all_metrics(approximation.ravel(), stds.ravel(), real_values.ravel(), verbose=False)
    return metrics

def nan_proportion(samples):
    return np.count_nonzero(np.isnan(samples)) / samples.size

def rpd(approximation, real_values):
    zero_num = np.abs(approximation.ravel() - real_values.ravel()) == 0.
    rpd_ = (2*np.abs(approximation.ravel() - real_values.ravel()) / (np.abs(approximation.ravel()) + np.abs(real_values.ravel())))
    rpd_[zero_num] = 0.
    return rpd_.mean()

def std_mae(approximation, stds, real_values):
    abs_error = np.abs(approximation.ravel() - real_values.ravel())
    return mae(stds, abs_error).mean()

def std_mre(approximation, stds, real_values):
    abs_error = np.abs(approximation.ravel() - real_values.ravel())
    return mre(stds, abs_error)

def std_rpd(approximation, stds, real_values):
    abs_error = np.abs(approximation.ravel() - real_values.ravel())
    return rpd(abs_error, stds.ravel())

def compute_all_metrics(approximations, stds, real_values):
    metrics_list = []
    if stds is None: stds = [None]*len(approximations)
    for app, std, rv in zip(approximations, stds, real_values):
        metrics = {}
        if std is not None:
            metrics.update(uct_metrics(app, std, rv))
            metrics.update({ "std_mre": std_mre(app, std, rv) })
            metrics.update({ "std_mae": std_mae(app, std, rv) })
            metrics.update({ "std_rpd": std_rpd(app, std, rv) })
        metrics.update({ "mre": mre(app, rv) })
        metrics.update({ "mae": mae(app, rv) })
        metrics.update({ "rpd": rpd(app, rv) })
        metrics.update(re_quantiles(app, rv))
        metrics_list.append(metrics)
    return metrics_list

def _compute_hubble_lcdm_forward(data):
    hubble_an = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["analytic"])
    hubble_fcnn = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["FCNN"])
    hubble_bbb = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["BBB_samples"][0])
    hubble_nlm = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["NLM_samples"][0])
    hubble_hmc = lcdm.H_LCDM(data["x"], lcdm.Om_m_0, 65, data["HMC_samples"][0])
    return hubble_an, hubble_fcnn, hubble_bbb, hubble_nlm, hubble_hmc

def _compute_hubble_lcdm_bundle(data):
    z, Om_m_0 = data["x"][0].reshape(lcdm.bundle_plot_dimension_sizes), data["x"][1].reshape(lcdm.bundle_plot_dimension_sizes)

    hubble_an = lcdm.H_LCDM(z, Om_m_0, 65, data["analytic"].reshape(lcdm.bundle_plot_dimension_sizes))
    hubble_fcnn = lcdm.H_LCDM(z, Om_m_0, 65, data["FCNN"].reshape(lcdm.bundle_plot_dimension_sizes))
    hubble_bbb = lcdm.H_LCDM(z, Om_m_0, 65, data["BBB_samples"][0].reshape(-1, *lcdm.bundle_plot_dimension_sizes))
    hubble_nlm = lcdm.H_LCDM(z, Om_m_0, 65, data["NLM_samples"][0].reshape(-1, *lcdm.bundle_plot_dimension_sizes))
    hubble_hmc = lcdm.H_LCDM(z, Om_m_0, 65, data["HMC_samples"][0].reshape(-1, *lcdm.bundle_plot_dimension_sizes))
    return hubble_an, hubble_fcnn, hubble_bbb, hubble_nlm, hubble_hmc

def _compute_hubble_cpl_forward(data):
    hubble_an = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["analytic"])
    hubble_fcnn = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["FCNN"])
    hubble_bbb = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["BBB_samples"][0])
    hubble_nlm = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["NLM_samples"][0])
    hubble_hmc = cpl.H_CPL(data["x"], cpl.w_0, cpl.w_1, .3, 65, data["HMC_samples"][0])
    return hubble_an, hubble_fcnn, hubble_bbb, hubble_nlm, hubble_hmc

def _compute_hubble_cpl_bundle(data):
    z, w_0, w_1 = data["x"][0].reshape(cpl.bundle_plot_dimension_sizes), data["x"][1].reshape(cpl.bundle_plot_dimension_sizes), data["x"][2].reshape(cpl.bundle_plot_dimension_sizes)

    hubble_an = cpl.H_CPL(z, w_0, w_1, .3, 65, data["analytic"].reshape(cpl.bundle_plot_dimension_sizes))
    hubble_fcnn = cpl.H_CPL(z, w_0, w_1, .3, 65, data["FCNN"].reshape(cpl.bundle_plot_dimension_sizes))
    hubble_bbb = cpl.H_CPL(z, w_0, w_1, .3, 65, data["BBB_samples"][0].reshape(-1, *cpl.bundle_plot_dimension_sizes))
    hubble_nlm = cpl.H_CPL(z, w_0, w_1, .3, 65, data["NLM_samples"][0].reshape(-1, *cpl.bundle_plot_dimension_sizes))
    hubble_hmc = cpl.H_CPL(z, w_0, w_1, .3, 65, data["HMC_samples"][0].reshape(-1, *cpl.bundle_plot_dimension_sizes))
    return hubble_an, hubble_fcnn, hubble_bbb, hubble_nlm, hubble_hmc

def _compute_hubble_quint_forward(data):
    hubble_an = quint.H_quint(data["x"], 1, .3, 65, data["numerical"])
    hubble_fcnn = quint.H_quint(data["x"], 1, .3, 65, data["FCNN"])
    hubble_bbb = quint.H_quint(data["x"], 1, .3, 65, data["BBB_samples"])
    hubble_nlm = quint.H_quint(data["x"], 1, .3, 65, data["NLM_samples"])
    hubble_hmc = quint.H_quint(data["x"], 1, .3, 65, data["HMC_samples"])
    return hubble_an, hubble_fcnn, hubble_bbb, hubble_nlm, hubble_hmc

def _compute_hubble_quint_bundle(data):
    z, lam, Om = data["x"][0].reshape(quint.bundle_plot_dimension_sizes), data["x"][1].reshape(quint.bundle_plot_dimension_sizes), data["x"][2].reshape(quint.bundle_plot_dimension_sizes)
    hubble_an = quint.H_quint(z, lam, Om, 65, [data["numerical"][i].reshape(quint.bundle_plot_dimension_sizes) for i in range(2)])
    hubble_fcnn = quint.H_quint(z, lam, Om, 65, [data["FCNN"][i].reshape(quint.bundle_plot_dimension_sizes) for i in range(2)])
    hubble_bbb = quint.H_quint(z, lam, Om, 65, [data["BBB_samples"][i].reshape(-1, *quint.bundle_plot_dimension_sizes) for i in range(2)])
    hubble_nlm = quint.H_quint(z, lam, Om, 65, [data["NLM_samples"][i].reshape(-1, *quint.bundle_plot_dimension_sizes) for i in range(2)])
    hubble_hmc = quint.H_quint(z, lam, Om, 65, [data["HMC_samples"][i].reshape(-1, *quint.bundle_plot_dimension_sizes) for i in range(2)])
    return hubble_an, hubble_fcnn, hubble_bbb, hubble_nlm, hubble_hmc

def _compute_hubble_hs_forward(data):
    hubble_an = hs.H_HS(data["x"], 1, .3, 65, data["numerical"])
    hubble_fcnn = hs.H_HS(data["x"], 1, .3, 65, data["FCNN"])
    hubble_bbb = hs.H_HS(data["x"], 1, .3, 65, data["BBB_samples"])
    hubble_nlm = hs.H_HS(data["x"], 1, .3, 65, data["NLM_samples"])
    hubble_hmc = hs.H_HS(data["x"], 1, .3, 65, data["HMC_samples"])
    return hubble_an, hubble_fcnn, hubble_bbb, hubble_nlm, hubble_hmc

def _compute_hubble_hs_bundle(data):
    z, b, Om_m_0 = data["x"][0].reshape(hs.bundle_plot_dimension_sizes), data["x"][1].reshape(hs.bundle_plot_dimension_sizes), data["x"][2].reshape(hs.bundle_plot_dimension_sizes)

    hubble_an = hs.H_HS(z, b, Om_m_0, 65, [data["numerical"][i].reshape(hs.bundle_plot_dimension_sizes) for i in range(5)])
    hubble_fcnn = hs.H_HS(z, b, Om_m_0, 65, [data["FCNN"][i].reshape(hs.bundle_plot_dimension_sizes) for i in range(5)])
    hubble_bbb = hs.H_HS(z, b, Om_m_0, 65, [data["BBB_samples"][i].reshape(-1, *hs.bundle_plot_dimension_sizes) for i in range(5)])
    hubble_nlm = hs.H_HS(z, b, Om_m_0, 65, [data["NLM_samples"][i].reshape(-1, *hs.bundle_plot_dimension_sizes) for i in range(5)])
    hubble_hmc = hs.H_HS(z, b, Om_m_0, 65, [data["HMC_samples"][i].reshape(-1, *hs.bundle_plot_dimension_sizes) for i in range(5)])
    return hubble_an, hubble_fcnn, hubble_bbb, hubble_nlm, hubble_hmc

def compute_equation_metrics(data):
    if isinstance(data["FCNN"], list):
        gt_sols = data["numerical"]
        fcnn_sols = data["FCNN"]
        bbb_sols = [data["BBB"][i][0] for i in range(len(data["FCNN"]))]
        bbb_stds = [data["BBB"][i][1] for i in range(len(data["FCNN"]))]
        nlm_sols = [data["NLM"][i][0].numpy() for i in range(len(data["FCNN"]))]
        nlm_stds = [data["NLM"][i][1].numpy() for i in range(len(data["FCNN"]))]
        hmc_sols = [data["HMC"][i][0] for i in range(len(data["FCNN"]))]
        hmc_stds = [data["HMC"][i][1] for i in range(len(data["FCNN"]))]
    else:
        gt_sols = [data["analytic"]]
        fcnn_sols = [data["FCNN"]]
        bbb_sols = [data["BBB"][0]]
        bbb_stds = [data["BBB"][1]]
        nlm_sols = [data["NLM"][0].numpy()]
        nlm_stds = [data["NLM"][1].numpy()]
        hmc_sols = [data["HMC"][0]]
        hmc_stds = [data["HMC"][1]]

    fcnn_metrics = compute_all_metrics(fcnn_sols, None, gt_sols)
    bbb_metrics = compute_all_metrics(bbb_sols, bbb_stds, gt_sols)
    nlm_metrics = compute_all_metrics(nlm_sols, nlm_stds, gt_sols)
    hmc_metrics = compute_all_metrics(hmc_sols, hmc_stds, gt_sols)

    for i in range(len(fcnn_metrics)):
        fcnn_metrics[i].update({ "mean_residual": np.abs(data["FCNN_residuals"][i]).mean() })
        bbb_metrics[i].update({ "mean_residual": np.abs(data["BBB_residuals"][i]).mean() })
        nlm_metrics[i].update({ "mean_residual": np.abs(data["NLM_residuals"][i]).mean() })
        hmc_metrics[i].update({ "mean_residual": np.abs(data["HMC_residuals"][i]).mean() })

    metrics = {
            "FCNN": fcnn_metrics,
            "BBB": bbb_metrics,
            "NLM": nlm_metrics,
            "HMC": hmc_metrics
    }
    return metrics

def compute_hubble_metrics(data, hubble_getter):
    hubble_an, hubble_fcnn, hubble_bbb, hubble_nlm, hubble_hmc = hubble_getter(data)

    hubble_bbb_mean, hubble_bbb_std = np.nanmean(hubble_bbb, axis=0), np.nanstd(hubble_bbb, axis=0)
    hubble_nlm_mean, hubble_nlm_std = np.nanmean(hubble_nlm, axis=0), np.nanstd(hubble_nlm, axis=0)
    hubble_hmc_mean, hubble_hmc_std = np.nanmean(hubble_hmc, axis=0), np.nanstd(hubble_hmc, axis=0)

    bbb_mask = ~(np.isnan(hubble_bbb_mean) | (hubble_bbb_std == 0))
    nlm_mask = ~(np.isnan(hubble_nlm_mean) | (hubble_nlm_std == 0))
    hmc_mask = ~(np.isnan(hubble_hmc_mean) | (hubble_hmc_std == 0))

    hubble_fcnn_metrics = compute_all_metrics([hubble_fcnn], None, [hubble_an])
    hubble_bbb_metrics = compute_all_metrics([hubble_bbb_mean[bbb_mask]], [hubble_bbb_std[bbb_mask]], [hubble_an[bbb_mask]])
    hubble_nlm_metrics = compute_all_metrics([hubble_nlm_mean[nlm_mask]], [hubble_nlm_std[nlm_mask]], [hubble_an[nlm_mask]])
    hubble_hmc_metrics = compute_all_metrics([hubble_hmc_mean[hmc_mask]], [hubble_hmc_std[hmc_mask]], [hubble_an[hmc_mask]])

    assert len(hubble_bbb_metrics) == 1
    assert len(hubble_nlm_metrics) == 1
    assert len(hubble_hmc_metrics) == 1

    hubble_fcnn_metrics[0].update({ "total_nan_proportion": nan_proportion(hubble_fcnn), "agg_nan_proportion": 0 })
    hubble_bbb_metrics[0].update({ "total_nan_proportion": nan_proportion(hubble_bbb), "agg_nan_proportion": nan_proportion(hubble_bbb_mean) })
    hubble_nlm_metrics[0].update({ "total_nan_proportion": nan_proportion(hubble_nlm), "agg_nan_proportion": nan_proportion(hubble_nlm_mean) })
    hubble_hmc_metrics[0].update({ "total_nan_proportion": nan_proportion(hubble_hmc), "agg_nan_proportion": nan_proportion(hubble_hmc_mean) })

    metrics = {
            "FCNN": hubble_fcnn_metrics,
            "BBB": hubble_bbb_metrics,
            "NLM": hubble_nlm_metrics,
            "HMC": hubble_hmc_metrics
    }
    return metrics

if __name__ == "__main__":
    from gc import collect
    eq_modules = {
        "lcdm": plcdm,
        "cpl": pcpl,
        "quintessence": pquint,
        "hs": phs
    }

    def get_data_getter(equation, bundle):
        module = eq_modules[equation]
        if bundle:
            return module.get_bundle_plot_data
        return module.get_plot_data
    
    def get_hubble_getter(equation, bundle):
        if equation == "lcdm" and bundle:
            return _compute_hubble_lcdm_bundle
        if equation == "lcdm" and not bundle:
            return _compute_hubble_lcdm_forward
        
        if equation == "cpl" and bundle:
            return _compute_hubble_cpl_bundle
        if equation == "cpl" and not bundle:
            return _compute_hubble_cpl_forward
        
        if equation == "quintessence" and bundle:
            return _compute_hubble_quint_bundle
        if equation == "quintessence" and not bundle:
            return _compute_hubble_quint_forward
        
        if equation == "hs" and bundle:
            return _compute_hubble_hs_bundle
        if equation == "hs" and not bundle:
            return _compute_hubble_hs_forward
        

    equations = ["lcdm", "cpl", "quintessence", "hs"]
    bundles = [False, True]
    domain_types = ["test", "train", "ood"]
    for eq in equations:
        for b in bundles:
            for dt in domain_types:
                for eb in [False, True]:
                    if eb and eq in ["quintessence", "hs"]:
                        continue
                    exp_name = build_exp_name(eq, eb=eb, bundle=b, domain_type=dt)
                    print(exp_name)
                    data = get_data_getter(eq, b)(eb=eb, domain_type=dt)
                    eq_metrics = compute_equation_metrics(data)
                    hubble_metrics = compute_hubble_metrics(data, get_hubble_getter(eq, b))
                    
                    with open(f"metric_results/eq_{exp_name}.dill", "wb") as f:
                        dill.dump(eq_metrics, f)
                    with open(f"metric_results/hubble_{exp_name}.dill", "wb") as f:
                        dill.dump(hubble_metrics, f)
                    collect()


    # data_getters_eb = [False, True, False, True, False, True, False, True, False, False, False, False] # , False, True, False, True
    # data_getters = [plcdm.get_plot_data, plcdm.get_plot_data, plcdm.get_bundle_plot_data, plcdm.get_bundle_plot_data, pcpl.get_plot_data, pcpl.get_plot_data, pcpl.get_bundle_plot_data, pcpl.get_bundle_plot_data, pquint.get_plot_data, pquint.get_bundle_plot_data, phs.get_plot_data, phs.get_bundle_plot_data] # , pcpl_params.get_plot_data, pcpl_params.get_plot_data, pcpl_params.get_bundle_plot_data, pcpl_params.get_bundle_plot_data
    # hubble_getters = [_compute_hubble_lcdm_forward, _compute_hubble_lcdm_forward, _compute_hubble_lcdm_bundle, _compute_hubble_lcdm_bundle, _compute_hubble_cpl_forward, _compute_hubble_cpl_forward, _compute_hubble_cpl_bundle, _compute_hubble_cpl_bundle, _compute_hubble_quint_forward, _compute_hubble_quint_bundle, _compute_hubble_hs_forward, _compute_hubble_hs_bundle] # , _compute_hubble_cpl_forward, _compute_hubble_cpl_forward, _compute_hubble_cpl_bundle, _compute_hubble_cpl_bundle
    # names = ["lcdm_forward", "lcdm_forward_eb", "lcdm_bundle", "lcdm_bundle_eb", "cpl_forward", "cpl_forward_eb", "cpl_bundle", "cpl_bundle_eb", "quint_forward", "quint_bundle", "hs_forward", "hs_bundle"] # , "cpl_params_forward", "cpl_params_forward_eb", "cpl_params_bundle", "cpl_params_bundle_eb"
    # for i in range(len(data_getters)):
    #     # if "quint" in names[i] or "hs" in names[i]: continue
    #     # if "lcdm_bundle" not in names[i]: continue
    #     if "cpl_forward" not in names[i]: continue
    #     # if "hs" not in names[i]: continue
    #     # if "hs" not in names[i] and "quint" not in names[i]: continue
    #     # if data_getters_eb[i]:
    #     #     data = data_getters[i](eb=True)
    #     #     eq_metrics = compute_equation_metrics(data)
    #     #     hubble_metrics = compute_hubble_metrics(data, hubble_getters[i])
    #     # else:
    #     #     data = data_getters[i]()
    #     #     eq_metrics = compute_equation_metrics(data)
    #     #     hubble_metrics = compute_hubble_metrics(data, hubble_getters[i])
    #     try:
    #         if data_getters_eb[i]:
    #             data = data_getters[i](eb=True)
    #             eq_metrics = compute_equation_metrics(data)
    #             hubble_metrics = compute_hubble_metrics(data, hubble_getters[i])
    #         else:
    #             data = data_getters[i]()
    #             eq_metrics = compute_equation_metrics(data)
    #             hubble_metrics = compute_hubble_metrics(data, hubble_getters[i])
    #     except Exception as e:
    #         print(f"Error in {names[i]}: {e}")
    #         continue

    #     with open(f"metric_results/eq_{names[i]}.dill", "wb") as f:
    #         dill.dump(eq_metrics, f)
    #     with open(f"metric_results/hubble_{names[i]}.dill", "wb") as f:
    #         dill.dump(hubble_metrics, f)
    #     collect()


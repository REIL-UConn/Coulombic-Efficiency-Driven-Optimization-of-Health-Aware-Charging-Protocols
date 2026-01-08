import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.optimize import minimize

from utils.soc_sweep_analysis import get_coulombic_efficiency, shift_ce_to_100


def collect_ce_surrogate_data(processed_root, groups, ce_min=97.0, ce_max=100.0):
    rows = []

    for grp in groups:
        soh = grp["SOH"]
        legend_labels = grp["legend_labels"]
        channels_cells = grp["channels_cells"]
        cell_to_legend_label = grp["cell_to_legend_label"]

        dir_soh = processed_root / f"SOH{soh}"

        for ch, cells in channels_cells.items():
            for cell in cells:
                key = cell_to_legend_label.get((ch, cell))
                if key is None:
                    continue
                c_str = legend_labels.get(key)
                if c_str is None:
                    continue
                c_rate = float(str(c_str).replace("C", ""))

                ret = get_coulombic_efficiency(
                    dir_processed=dir_soh, channel_id=int(ch), cell_id=int(cell), soc=None
                )

                ce = ret["CE"] * 100.0
                u = ret["u_CE"] * 100.0
                ce, _ = shift_ce_to_100(ce, u)

                for soc, ce_val in zip(ret["SOC"], ce):
                    if ce_min <= ce_val <= ce_max:
                        rows.append([float(soc), float(c_rate), float(soh), float(ce_val)])

    arr = np.asarray(rows, dtype=float)
    X = arr[:, :3]
    y = arr[:, 3]
    df = pd.DataFrame(arr, columns=["SOC", "C-rate", "SOH", "CE"])
    return X, y, df


def fit_poly2_linear(X, y):
    poly = PolynomialFeatures(degree=2)
    Xp = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(Xp, y)
    return poly, model


def predict_poly2_linear(poly, model, X):
    return model.predict(poly.transform(X))


def fit_gpr_surrogate(X, y, n_restarts_optimizer=10):
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=True,
    )
    gpr.fit(X, y)
    return gpr


def predict_gpr(gpr, X, return_std=False):
    if return_std:
        mean, std = gpr.predict(X, return_std=True)
        return mean, std
    return gpr.predict(X)


mpl.rcParams["font.family"] = "DejaVu Sans"


def plot_surrogate_fits_1x4(
    X,
    y,
    *,
    model=None,
    poly=None,
    gpr=None,
    soh_levels=(100, 90, 80, 70),
    unique_colors=("C0", "C1", "C2", "C3", "C4"),
    outpath=None,
):
    data = np.column_stack((X, y))
    unique_crates = np.unique(data[:, 1])

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    soc_fine = np.linspace(10, 90, 200)

    for ax, soh in zip(axes, soh_levels):
        data_soh = data[data[:, 2] == soh]

        for i, cr in enumerate(unique_crates):
            idx = data_soh[:, 1] == cr
            soc_exp = data_soh[idx, 0]
            ce_exp = data_soh[idx, 3]

            valid_ce = ce_exp[ce_exp > 0]
            shift = 100 - valid_ce[0] if len(valid_ce) > 0 else 0
            ce_exp_shifted = ce_exp + shift

            ax.scatter(
                soc_exp,
                ce_exp_shifted,
                color=unique_colors[i],
                s=85,
                alpha=0.8,
                label=(f"Experimental ({cr:.1f}C)" if soh == soh_levels[0] else None),
            )

            X_fine = np.column_stack(
                (soc_fine, cr * np.ones_like(soc_fine), soh * np.ones_like(soc_fine))
            )

            if gpr is not None:
                ce_pred, sigma = gpr.predict(X_fine, return_std=True)
                ce_pred_shifted = ce_pred + shift
                ax.plot(
                    soc_fine,
                    ce_pred_shifted,
                    color=unique_colors[i],
                    linewidth=4,
                    label=(f"GPR Fit ({cr:.1f}C)" if soh == soh_levels[0] else None),
                )
                ax.fill_between(
                    soc_fine,
                    ce_pred_shifted - 1.96 * sigma,
                    ce_pred_shifted + 1.96 * sigma,
                    color=unique_colors[i],
                    alpha=0.2,
                )
            else:
                ce_pred = model.predict(poly.transform(X_fine))
                ce_pred_shifted = ce_pred + shift
                ax.plot(
                    soc_fine,
                    ce_pred_shifted,
                    color=unique_colors[i],
                    linewidth=4,
                    label=(f"Linear Fit ({cr:.1f}C)" if soh == soh_levels[0] else None),
                )

        ax.set_title(f"SOH = {soh}%", fontsize=27)
        ax.set_xlabel("SOC (%)", fontsize=25)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlim(10, 90)
        ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        ax.set_ylim(97, 100.2)
        ax.tick_params(labelsize=20)

    axes[0].set_ylabel("Coulombic Efficiency (%)", fontsize=25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(unique_crates),
        frameon=False,
        fontsize=20,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if outpath is not None:
        plt.savefig(outpath, format="svg", dpi=600, bbox_inches="tight")

    plt.show()


def plot_surrogate_contours_1x4(
    *,
    model=None,
    poly=None,
    gpr=None,
    soh_values=(100, 90, 80, 70),
    outpath=None,
):
    soc_range = np.linspace(0, 100, 100)
    c_rate_range = np.linspace(1.5, 3.5, 100)
    SOC_grid, C_rate_grid = np.meshgrid(soc_range, c_rate_range)

    fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharey=True, dpi=600)

    levels = np.linspace(95, 100, 10)
    cmap = "YlGn"

    for ax, soh in zip(axs, soh_values):
        X_test = np.column_stack(
            (SOC_grid.ravel(), C_rate_grid.ravel(), soh * np.ones(SOC_grid.size))
        )

        if gpr is not None:
            y_pred = gpr.predict(X_test)
        else:
            y_pred = model.predict(poly.transform(X_test))

        CE_pred = np.clip(y_pred.reshape(SOC_grid.shape), 95, 100)

        cf = ax.contourf(
            SOC_grid,
            C_rate_grid,
            CE_pred,
            levels=levels,
            cmap=cmap,
            vmin=95,
            vmax=100,
        )

        ax.set_title(f"SOH = {soh}%", fontsize=27)
        ax.set_xlabel("SOC (%)", fontsize=25)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(labelsize=18)
        ax.grid(True, linestyle="--", alpha=0.5)

    axs[0].set_ylabel("C-Rate (C)", fontsize=25)

    divider = make_axes_locatable(axs[-1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_ticks(np.linspace(95, 100, 6))
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Predicted CE (%)", fontsize=25)

    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, format="svg", dpi=600, bbox_inches="tight")

    plt.show()


def _predict_ce_percent(*, soc, c_rate, soh, model=None, poly=None, gpr=None):
    X = np.array([[soc, c_rate, soh]], dtype=float)
    if gpr is not None:
        return float(gpr.predict(X)[0])
    return float(model.predict(poly.transform(X))[0])


def optimize_profile_given_loss(
    *,
    SOH_design=100,
    M=20,
    L_allowed=0.20,
    soc_points=None,
    bounds=(1.5, 3.5),
    x0=None,
    model=None,
    poly=None,
    gpr=None,
    maxiter=5000,
    disp=True,
):
    # ---- discretization ----
    if soc_points is None:
        soc_points = np.arange(0, 101, 5)
    soc_points = np.asarray(soc_points, dtype=float)

    N = len(soc_points) - 1
    delta_soc = (soc_points[1] - soc_points[0]) / 100.0  # 5% -> 0.05

    bnds = [tuple(bounds)] * N
    if x0 is None:
        x0 = np.full(N, 2.5, dtype=float)


    def objective(c_rate_array):
        total_time = 0.0
        for i in range(N):
            soc = soc_points[i]
            c_rate = c_rate_array[i]
            ce_pred = _predict_ce_percent(
                soc=soc, c_rate=c_rate, soh=SOH_design, model=model, poly=poly, gpr=gpr
            )
            ce_fraction = ce_pred / 100.0
            total_time += delta_soc / (c_rate * ce_fraction)
        return total_time


    def capacity_loss(c_rate_array):
        loss = 0.0
        for i in range(N):
            soc = soc_points[i]
            c_rate = c_rate_array[i]
            ce_pred = _predict_ce_percent(
                soc=soc, c_rate=c_rate, soh=SOH_design, model=model, poly=poly, gpr=gpr
            )
            loss += delta_soc * (100.0 / ce_pred - 1.0)
        return M * loss

    def constraint_fun(c_rate_array):
        return L_allowed - capacity_loss(c_rate_array)

    nl_con = {"type": "ineq", "fun": constraint_fun}

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=[nl_con],
        options={"disp": disp, "maxiter": maxiter},
    )

    return {
        "result": res,
        "soc_points": soc_points,
        "SOH_design": float(SOH_design),
        "M": int(M),
        "L_allowed": float(L_allowed),
        "profile_5pct": res.x,
        "objective": objective,          # to match your workflow
        "capacity_loss": capacity_loss,  # to match your workflow
        "total_time": float(objective(res.x)),
        "achieved_loss": float(capacity_loss(res.x)),
    }


def aggregate_profile(profile, group_size):
    profile = np.asarray(profile, dtype=float)
    n = len(profile)
    return np.array([np.mean(profile[i : i + group_size]) for i in range(0, n, group_size)], dtype=float)


def compute_aggregated_totals(
    *,
    profile_agg,
    group_size,
    soc_points,
    SOH_design,
    M,
    model=None,
    poly=None,
    gpr=None,
):
    # ---- compute_aggregated_totals ----
    soc_points = np.asarray(soc_points, dtype=float)
    total_time = 0.0
    total_loss = 0.0

    for j in range(len(profile_agg)):
        start = soc_points[j * group_size]
        end = soc_points[j * group_size + group_size]
        midpoint = (start + end) / 2.0
        delta_agg = (end - start) / 100.0
        c_rate_agg = float(profile_agg[j])

        ce_pred = _predict_ce_percent(
            soc=midpoint, c_rate=c_rate_agg, soh=SOH_design, model=model, poly=poly, gpr=gpr
        )
        ce_fraction = ce_pred / 100.0

        total_time += delta_agg / (c_rate_agg * ce_fraction)
        total_loss += delta_agg * (100.0 / ce_pred - 1.0)

    return float(total_time), float(M * total_loss)


def run_loss_sweep(
    *,
    SOH_design=100,
    M=20,
    allowed_losses=(0.10, 0.20, 0.30),
    soc_points=None,
    bounds=(1.5, 3.5),
    x0=None,
    model=None,
    poly=None,
    gpr=None,
    maxiter=5000,
    disp=True,
):
    sols = {}
    for L in allowed_losses:
        sols[float(L)] = optimize_profile_given_loss(
            SOH_design=SOH_design,
            M=M,
            L_allowed=float(L),
            soc_points=soc_points,
            bounds=bounds,
            x0=x0,
            model=model,
            poly=poly,
            gpr=gpr,
            maxiter=maxiter,
            disp=disp,
        )
    return sols


def plot_profiles_and_barcharts_loss_sweep(
    *,
    solutions_by_loss,
    model=None,
    poly=None,
    gpr=None,
    outpath=None,
):
    # ---- plotting: 5% vs 20% only ----
    allowed_losses = sorted(solutions_by_loss.keys())
    soc_points = solutions_by_loss[allowed_losses[0]]["soc_points"]
    SOH_design = solutions_by_loss[allowed_losses[0]]["SOH_design"]
    M = solutions_by_loss[allowed_losses[0]]["M"]

    # collect profiles
    optimal_profiles = {L: solutions_by_loss[L]["profile_5pct"] for L in allowed_losses}
    agg_profiles_20 = {L: aggregate_profile(optimal_profiles[L], 4) for L in allowed_losses}

    # time/loss for 5% intervals 
    time_5 = [solutions_by_loss[L]["total_time"] for L in allowed_losses]
    loss_5 = [solutions_by_loss[L]["achieved_loss"] for L in allowed_losses]

    # time/loss for 20% intervals
    time_20, loss_20 = [], []
    for L in allowed_losses:
        t, l = compute_aggregated_totals(
            profile_agg=agg_profiles_20[L],
            group_size=4,
            soc_points=soc_points,
            SOH_design=SOH_design,
            M=M,
            model=model,
            poly=poly,
            gpr=gpr,
        )
        time_20.append(t)
        loss_20.append(l)

    # ---- Plotting ----
    plt.rcParams.update({"xtick.labelsize": 18, "ytick.labelsize": 18})
    colors_allowed = ["#5D4037", "#8D6E63", "#D7CCC8"]

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(
        nrows=2, ncols=3,
        height_ratios=[1, 1],
        width_ratios=[1.5, 1.5, 1.0],
        hspace=0.3, wspace=0.3
    )

    # Subplot 1: Optimal profile (5% intervals)
    ax1 = fig.add_subplot(gs[:, 0])
    for idx, L in enumerate(allowed_losses):
        profile = optimal_profiles[L]
        profile_step = np.append(profile, profile[-1])
        ax1.step(
            soc_points, profile_step, where="post",
            color=colors_allowed[idx], linewidth=5,
            label=f"{int(L*100)}% Loss"
        )
    ax1.set_xlabel("SOC (%)", fontsize=22)
    ax1.set_ylabel("C-Rate (C)", fontsize=22)
    ax1.set_title("Optimal Profile (5% Intervals)", fontsize=22)
    ax1.grid(True, linestyle="--", linewidth=1, alpha=0.7)
    ax1.legend(fontsize=20, loc="best")
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_xlim(0, 100)

    # Subplot 2: Aggregated profile (20% intervals)
    ax2 = fig.add_subplot(gs[:, 1])
    agg_soc_20 = np.arange(0, 101, 20)
    for idx, L in enumerate(allowed_losses):
        profile = agg_profiles_20[L]
        profile_step = np.append(profile, profile[-1])
        ax2.step(
            agg_soc_20, profile_step, where="post",
            color=colors_allowed[idx], linewidth=6,
            label=f"{int(L*100)}% Loss"
        )
    ax2.set_xlabel("SOC (%)", fontsize=22)
    ax2.set_title("Aggregated Profile (20% Intervals)", fontsize=22)
    ax2.grid(True, linestyle="--", linewidth=1, alpha=0.7)
    ax2.legend(fontsize=20, loc="best")
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.set_xlim(0, 100)

    # Subplot 3: Bar plots
    ax3_top = fig.add_subplot(gs[0, 2])
    ax3_bot = fig.add_subplot(gs[1, 2])

    bar_width = 0.2
    x_pos = np.arange(len(allowed_losses))
    x_labels = [f"{int(L*100)}%" for L in allowed_losses]

    method_labels = ["5% Intervals", "20% Intervals"]
    bar_colors_time = [mcolors.to_rgba("#008000", 1.0), mcolors.to_rgba("#008000", 0.4)]
    bar_colors_loss = [mcolors.to_rgba("#DC143C", 1.0), mcolors.to_rgba("#DC143C", 0.4)]
    offsets = [-bar_width / 2, bar_width / 2]

    time_data = [time_5, time_20]
    loss_data = [loss_5, loss_20]

    for j in range(2):
        ax3_top.bar(
            x_pos + offsets[j], time_data[j],
            width=bar_width, color=bar_colors_time[j],
            label=method_labels[j]
        )
        for i, t in enumerate(time_data[j]):
            ax3_top.text(
                x_pos[i] + offsets[j], t * 0.92, f"{t:.2f}",
                ha="center", va="top", color="white", fontsize=12, rotation=90
            )
    ax3_top.set_ylabel("Total Time (hrs)", fontsize=20)
    ax3_top.set_title("Total Charging Time", fontsize=22)
    ax3_top.set_xticks(x_pos)
    ax3_top.set_xticklabels(x_labels, fontsize=20)

    for j in range(2):
        loss_percent = 100 * np.array(loss_data[j], dtype=float)
        ax3_bot.bar(
            x_pos + offsets[j], loss_percent,
            width=bar_width, color=bar_colors_loss[j],
            label=method_labels[j]
        )
        for i, l in enumerate(loss_percent):
            ax3_bot.text(
                x_pos[i] + offsets[j], l * 0.92, f"{l:.1f}",
                ha="center", va="top", color="white", fontsize=12, rotation=90
            )
    ax3_bot.set_ylabel("Achieved Loss (%)", fontsize=20)
    ax3_bot.set_xlabel("Allowed Loss (%)", fontsize=20)
    ax3_bot.set_title("Capacity Loss", fontsize=22)
    ax3_bot.set_xticks(x_pos)
    ax3_bot.set_xticklabels(x_labels, fontsize=20)

    ax3_top.legend(fontsize=12, loc="upper right")
    ax3_bot.legend(fontsize=12, loc="upper left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if outpath is not None:
        plt.savefig(outpath, format="svg", dpi=600, bbox_inches="tight")

    plt.show()


def plot_profiles_vs_soh_fixed_loss(
    *,
    SOH_levels=(100, 90, 80, 70),
    L_allowed=0.20,
    M=20,
    soc_points=None,
    bounds=(1.5, 3.5),
    x0=None,
    model=None,
    poly=None,
    gpr=None,
    title="Surrogate",
    outpath=None,
):
    if soc_points is None:
        soc_points = np.arange(0, 101, 5)

    fig = plt.figure(figsize=(5, 3), dpi=300)
    gs = fig.add_gridspec(1, 20)
    ax = fig.add_subplot(gs[0, :17])
    cax = fig.add_subplot(gs[0, 17])

    cmap = plt.cm.copper_r
    norm = plt.Normalize(vmin=70, vmax=100)

    for soh in SOH_levels:
        sol = optimize_profile_given_loss(
            SOH_design=soh,
            M=M,
            L_allowed=L_allowed,
            soc_points=soc_points,
            bounds=bounds,
            x0=x0,
            model=model,
            poly=poly,
            gpr=gpr,
            maxiter=5000,
            disp=False,
        )
        prof = sol["profile_5pct"]

        # ---- 20% aggregation from 0 to 80 ----
        soc_steps, c_rate_steps = [], []
        for i in range(4):  # 0-20, 20-40, 40-60, 60-80
            soc_start = i * 20
            soc_end = (i + 1) * 20
            c_rate = float(np.mean(prof[i * 4 : (i + 1) * 4]))
            soc_steps.extend([soc_start, soc_end])
            c_rate_steps.extend([c_rate, c_rate])

        ax.plot(soc_steps, c_rate_steps, "-", color=cmap(norm(soh)), linewidth=3)

    ax.set_xlabel("SOC (%)", fontsize=18)
    ax.set_ylabel("C-Rate (C)", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 80)
    ax.set_ylim(1.0, 4.0)
    ax.set_xticks(np.arange(0, 81, 20))
    ax.tick_params(labelsize=16)
    ax.set_title(title, fontsize=18, pad=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("SOH (%)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, format="svg", dpi=600, bbox_inches="tight")

    plt.show()

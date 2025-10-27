# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "click",
#     "fastcan",
#     "matplotlib",
# ]
# ///

"""Generate phase portraits results for dual stable equilibria data"""

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sklearn.metrics import r2_score
from fastcan.narx import make_narx

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def _duffing_equation(y, t):
    """Non-autonomous system"""
    # y1 is displacement and y2 is velocity
    y1, y2 = y
    # u is sinusoidal input
    u = 2.5 * np.cos(2 * np.pi * t)
    # dydt is derivative of y1 and y2
    dydt = [y2, -0.1 * y2 + y1 - 0.25 * y1**3 + u]
    return dydt


def _auto_duffing_equation(y, t):
    """Autonomous system"""
    y1, y2 = y
    dydt = [y2, -0.1 * y2 + y1 - 0.25 * y1**3]
    return dydt

def _plot_prediction(ax, t, y_true, y_pred, title, show_legend=False):
    ax.plot(t, y_true, label="true")
    ax.plot(t, y_pred, label="predicted", linestyle='--')
    if show_legend:
        ax.legend()
    ax.set_title(f"{title} (R2: {r2_score(y_true, y_pred):.3f})")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("y(t)")
    ax.grid(True, alpha=0.3)

def plot_fitting_performance():
    figure_name_0 = "eq1.pdf"
    figure_name_1 = "eq2.pdf"
    # 10 s duration with 0.01 Hz sampling time,
    # so 1000 samples in total for each measurement
    dur = 10
    n_samples = 1000
    t = np.linspace(0, dur, n_samples)
    # External excitation is the same for each measurement
    u = 2.5 * np.cos(2 * np.pi * t).reshape(-1, 1)

    # Small additional white noise
    rng = np.random.default_rng(12345)
    e_train_0 = rng.normal(0, 0.0004, n_samples)
    e_test = rng.normal(0, 0.0004, n_samples)

    # Solve differential equation to get displacement as y
    # Initial condition at displacement 0.6 and velocity 0.8
    sol = odeint(_duffing_equation, [0.6, 0.8], t)
    y_train_0 = sol[:, 0] + e_train_0

    # Initial condition at displacement 0.6 and velocity -0.8
    sol = odeint(_duffing_equation, [0.6, -0.8], t)
    y_test = sol[:, 0] + e_test

    max_delay = 3

    narx_model = make_narx(
        X=u,
        y=y_train_0,
        n_terms_to_select=5,
        max_delay=max_delay,
        poly_degree=3,
        verbose=0,
    )

    # OSA NARX
    narx_model.fit(u, y_train_0)
    y_train_0_osa_pred = narx_model.predict(u, y_init=y_train_0[:max_delay])
    y_test_osa_pred = narx_model.predict(u, y_init=y_test[:max_delay])

    # MSA NARX
    narx_model.fit(u, y_train_0, coef_init="one_step_ahead")
    y_train_0_msa_pred = narx_model.predict(u, y_init=y_train_0[:max_delay])
    y_test_msa_pred = narx_model.predict(u, y_init=y_test[:max_delay])

    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    _plot_prediction(ax[0, 0], t, y_train_0, y_train_0_osa_pred, "OSA NARX on Train 0", show_legend=True)
    _plot_prediction(ax[0, 1], t, y_train_0, y_train_0_msa_pred, "MSA NARX on Train 0")
    _plot_prediction(ax[1, 0], t, y_test, y_test_osa_pred, "OSA NARX on Test")
    _plot_prediction(ax[1, 1], t, y_test, y_test_msa_pred, "MSA NARX on Test")
    fig.tight_layout()
    plt.savefig(figure_name_0, bbox_inches="tight")
    print("Image " + figure_name_0 + " has been generated.")

    e_train_1 = rng.normal(0, 0.0004, n_samples)

    # Solve differential equation to get displacement as y
    # Initial condition at displacement 0.5 and velocity -1
    sol = odeint(_duffing_equation, [0.5, -1], t)
    y_train_1 = sol[:, 0] + e_train_1

    u_all = np.r_[u, [[np.nan]] * max_delay, u]
    y_all = np.r_[y_train_0, [np.nan] * max_delay, y_train_1]
    narx_model = make_narx(
        X=u_all,
        y=y_all,
        n_terms_to_select=5,
        max_delay=max_delay,
        poly_degree=3,
        verbose=0,
    )

    narx_model.fit(u_all, y_all)
    y_train_0_osa_pred = narx_model.predict(u, y_init=y_train_0[:max_delay])
    y_train_1_osa_pred = narx_model.predict(u, y_init=y_train_1[:max_delay])
    y_test_osa_pred = narx_model.predict(u, y_init=y_test[:max_delay])

    narx_model.fit(u_all, y_all, coef_init="one_step_ahead")
    y_train_0_msa_pred = narx_model.predict(u, y_init=y_train_0[:max_delay])
    y_train_1_msa_pred = narx_model.predict(u, y_init=y_train_1[:max_delay])
    y_test_msa_pred = narx_model.predict(u, y_init=y_test[:max_delay])

    fig, ax = plt.subplots(3, 2, figsize=(8, 9))
    _plot_prediction(ax[0, 0], t, y_train_0, y_train_0_osa_pred, "OSA NARX on Train 0", show_legend=True)
    _plot_prediction(ax[0, 1], t, y_train_0, y_train_0_msa_pred, "MSA NARX on Train 0")
    _plot_prediction(ax[1, 0], t, y_train_1, y_train_1_osa_pred, "OSA NARX on Train 1")
    _plot_prediction(ax[1, 1], t, y_train_1, y_train_1_msa_pred, "MSA NARX on Train 1")
    _plot_prediction(ax[2, 0], t, y_test, y_test_osa_pred, "OSA NARX on Test")
    _plot_prediction(ax[2, 1], t, y_test, y_test_msa_pred, "MSA NARX on Test")
    fig.tight_layout()
    plt.savefig(figure_name_1, bbox_inches="tight")
    print("Image " + figure_name_1 + " has been generated.")


def plot_phase_portraits():
    figure_name = "pp.pdf"
    dur = 10
    n_samples = 1000
    n_init = 10

    x0 = np.linspace(0, 2, n_init)
    y0_y = np.cos(np.pi * x0)
    y0_x = np.sin(np.pi * x0)
    y0 = np.c_[y0_x, y0_y]

    t = np.linspace(0, dur, n_samples)
    sol = np.zeros((n_init, n_samples, 2))
    for i in range(n_init):
        sol[i] = odeint(_auto_duffing_equation, y0[i], t)

    plt.figure()
    for i in range(n_init):
        plt.plot(sol[i, :, 0], sol[i, :, 1], c="tab:blue")

    y_min = np.nanmin(sol[:, :, 0])
    y_max = np.nanmax(sol[:, :, 0])
    dot_y_min = np.nanmin(sol[:, :, 1])
    dot_y_max = np.nanmax(sol[:, :, 1])
    y, dot_y = np.meshgrid(
        np.linspace(y_min, y_max, 30), np.linspace(dot_y_min, dot_y_max, 30)
    )
    ddot_y = _auto_duffing_equation([y, dot_y], 0)[1]
    plt.streamplot(
        y,
        dot_y,
        dot_y,
        ddot_y,
        color=(0.5, 0.5, 0.5, 0.3),
        density=1.2,
        minlength=0.02,
        maxlength=0.1,
        linewidth=0.5,
        arrowsize=0.5,
    )
    plt.xlabel("y(t)")
    plt.ylabel("dy/dt(t)")
    offset = 0.02
    plt.xlim(y_min - offset, y_max + offset)
    plt.ylim(dot_y_min - offset, dot_y_max + offset)
    plt.tick_params(axis="both")
    plt.savefig(figure_name, bbox_inches="tight")
    print("Image " + figure_name + " has been generated.")

@click.command()
@click.option("--plot-pp", is_flag=True, help="Plot phase portraits")
@click.option("--plot-fitting", is_flag=True, help="Plot fitting performance")
def main(plot_pp, plot_fitting):
    if plot_fitting:
        plot_fitting_performance()
    if plot_pp:
        plot_phase_portraits()

if __name__ == "__main__":
    main()

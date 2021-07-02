"""
Functions used for plotting in the examples.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt

from math_utilities.config import UNITS_IN_ONE_DAY
from math_utilities.discrete_distributions_utils import DiscreteDistributionOnNonNegatives
from math_utilities.general_utilities import effectiveness


def plot_discrete_distributions(
    ds: Sequence[DiscreteDistributionOnNonNegatives],
    custom_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    tau_max: Optional[float] = None,
    plot_cdfs: bool = False,
):
    """
    Function that plots the PMFs or CDFs of a collection of discrete distributions on [0,+∞).
    """
    if tau_max is None:
        tau_max = max(d.tau_max for d in ds)
    tau_range = range(0, tau_max + 1)
    if custom_labels is None:
        labels = [str(i) for i in range(len(ds))]
    else:
        labels = custom_labels
    for i, d in enumerate(ds):
        values = (
            [d.cdf(tau) for tau in tau_range] if plot_cdfs else [d.pmf(tau) for tau in tau_range]
        )
        plt.plot(
            [x / UNITS_IN_ONE_DAY for x in tau_range], values, label=labels[i],
        )
    if len(ds) > 1 or custom_labels is not None:
        plt.legend()
    plt.title(title)
    plt.show()


def plot_homogeneous_time_evolution(
    t_in_days_sequence: Sequence[float],
    R_ts: Sequence[float],
    R0: float,
    FT_infty_sequence: Sequence[float],
    nu_ts: Sequence[float],
    nu0_ts: Sequence[float],
) -> None:
    """
    Plots the time evolution of R_t, FT_t(∞), nu_t in the homogeneous scenario.
    """
    fig = plt.figure(figsize=(10, 15))

    t_max = t_in_days_sequence[-1]

    R_last = R_ts[-1]

    # R
    R_tplot = fig.add_subplot(311)
    R_tplot.set_xlabel("t (days)")
    R_tplot.set_ylabel("R_t")
    R_tplot.grid(True)
    R_tplot.set_xlim(0, t_max)
    R_tplot.set_ylim(0, R0)
    R_tplot.plot(
        t_in_days_sequence,
        R_ts,
        color="black",
        label=f"R_t → {round(R_last, 2)}, Eff_t → {round(effectiveness(R_last, R0), 2)}",
    ),
    R_tplot.legend()

    # FT
    Pplot = fig.add_subplot(312)
    Pplot.set_xlabel("t (days)")
    Pplot.set_ylabel("Probability")
    Pplot.grid(True)
    Pplot.set_xlim(0, t_max)
    Pplot.set_ylim(0, 1)
    Pplot.plot(
        t_in_days_sequence,
        FT_infty_sequence,
        color="black",
        label="Prob. that infected tests positive",
    )

    Pplot.legend()

    # nu
    nuplot = fig.add_subplot(313)
    nuplot.set_xlabel("t (days)")
    nuplot.set_ylabel("Number of people")
    nuplot.grid(True)
    nuplot.set_xlim(0, t_max)
    nuplot.plot(
        t_in_days_sequence, nu_ts, color="black", label="Number of infected",
    )
    nuplot.plot(
        t_in_days_sequence, nu0_ts, color="gray", label="Number of infected without measures",
    )

    nuplot.legend()

    plt.show()


def plot_time_evolution_with_app(
    t_in_days_sequence: Sequence[float],
    epsilon_app: Callable[[int], float],
    Fsigmaapp_infty: Sequence[float],
    R_ts: Sequence[float],
    Rapp_ts: Sequence[float],
    Rnoapp_ts: Sequence[float],
    R0: float,
    FT_ts_infty: Sequence[float],
    FT_ts_app_infty: Sequence[float],
    FT_ts_noapp_infty: Sequence[float],
) -> None:
    """
    Plots the time evolution of R_t, FT_t(∞) in the scenario with app.
    """
    fig = plt.figure(figsize=(10, 15))

    t_max = t_in_days_sequence[-1]

    R_last = R_ts[-1]
    Rapp_last = Rapp_ts[-1]
    Rnoapp_last = Rnoapp_ts[-1]

    # R
    R_tplot = fig.add_subplot(211)
    R_tplot.set_xlabel("t (days)")
    R_tplot.set_ylabel("R_t")
    R_tplot.grid(True)
    R_tplot.set_xlim(0, t_max)
    R_tplot.set_ylim(0, R0)
    R_tplot.plot(
        t_in_days_sequence,
        R_ts,
        color="black",
        label=f"R_t → {round(R_last, 2)}, Eff_t → {round(effectiveness(R_last, R0), 2)}",
    ),

    R_tplot.plot(
        t_in_days_sequence,
        Rapp_ts,
        color="green",
        label=f"R_t app → {round(Rapp_last, 2)}, Eff_t app → {round(effectiveness(Rapp_last, R0), 2)}",
    ),

    R_tplot.plot(
        t_in_days_sequence,
        Rnoapp_ts,
        color="blue",
        label=f"R_t no app → {round(Rnoapp_last, 2)}, Eff_t no app → {round(effectiveness(Rnoapp_last, R0), 2)}",
    ),
    R_tplot.legend()

    # FT, papp
    Pplot = fig.add_subplot(212)
    Pplot.set_xlabel("t (days)")
    Pplot.set_ylabel("Probability")
    Pplot.grid(True)
    Pplot.set_xlim(0, t_max)
    Pplot.set_ylim(0, 1)
    Pplot.plot(
        t_in_days_sequence,
        [epsilon_app(t) for t in range(len(t_in_days_sequence))],
        color="yellow",
        label="Prob. infected has the app",
    )
    Pplot.plot(
        t_in_days_sequence, Fsigmaapp_infty, color="red", label="Prob. infector has the app",
    )
    Pplot.plot(
        t_in_days_sequence, FT_ts_infty, color="black", label="Prob. that infected tests positive",
    )
    Pplot.plot(
        t_in_days_sequence,
        FT_ts_app_infty,
        color="green",
        label="Prob. that infected with app tests positive",
    )
    Pplot.plot(
        t_in_days_sequence,
        FT_ts_noapp_infty,
        color="blue",
        label="Prob. that infected without app tests positive",
    )
    Pplot.legend()

    plt.show()

"""
Some examples running the full algorithm in the homogeneous scenario (no app usage).
The examples print and plot the time evolution of the main KPIs.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from epidemic_suppression_algorithms.homogeneous_evolution_algorithm import (
    compute_time_evolution_homogeneous_case,
)
from examples.plotting_utils import plot_homogeneous_time_evolution
from math_utilities.discrete_distributions_utils import delta_distribution
from model_utilities.epidemic_data import (
    R0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    tauS,
)
from model_utilities.scenarios import HomogeneousScenario

p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()


def time_evolution_homogeneous_model_example():
    """
    Example running the full algorithm in the homogeneous scenario (no app usage).
    """
    t_max_in_days = 20

    DeltaAT_in_days = 2

    scenario = HomogeneousScenario(
        p_gs=p_gs,
        b0_gs=b0_gs,
        tauS=tauS,
        t_0=0,
        ss=(lambda t: 0, lambda t: 0.5),
        sc=lambda t: 0.7,
        xi=lambda t: 0.9,
        DeltaAT=delta_distribution(peak_tau_in_days=DeltaAT_in_days),
    )

    (
        t_in_days_list,
        nu,
        nu0,
        R,
        R_by_severity,
        FT_infty,
    ) = compute_time_evolution_homogeneous_case(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=tuple(d.normalize() for d in b0_gs),
        threshold_to_stop=0.001,
    )

    plot_homogeneous_time_evolution(
        t_in_days_sequence=t_in_days_list,
        R_ts=R,
        R0=R0,
        FT_infty_sequence=FT_infty,
        nu_ts=nu,
        nu0_ts=nu0,
    )


def time_evolution_homogeneous_model_example_no_negative_times():
    """
    Example running the full algorithm in the homogeneous scenario (no app usage).
    In this case we assume that there are no infected people at t<0.
    """
    t_max_in_days = 20

    DeltaAT_in_days = 2

    t_0 = 0

    scenario = HomogeneousScenario(
        p_gs=p_gs,
        b0_gs=b0_gs,
        tauS=tauS,
        t_0=0,
        ss=(lambda t: 0, lambda t: 0.5 if t >= t_0 else 0),
        sc=lambda t: 0.7 if t >= t_0 else 0,
        xi=lambda t: 0.9 if t >= t_0 else 0,
        DeltaAT=delta_distribution(peak_tau_in_days=DeltaAT_in_days),
    )

    (
        t_in_days_list,
        nu,
        nu0,
        R,
        R_by_severity,
        FT_infty,
    ) = compute_time_evolution_homogeneous_case(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=None,
        threshold_to_stop=0.001,
    )

    plot_homogeneous_time_evolution(
        t_in_days_sequence=t_in_days_list,
        R_ts=R,
        R0=R0,
        FT_infty_sequence=FT_infty,
        nu_ts=nu,
        nu0_ts=nu0,
    )

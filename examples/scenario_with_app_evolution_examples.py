"""
Some examples running the full algorithm in the scenario with app usage.
The examples print and plot the time evolution of the main KPIs.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from epidemic_suppression_algorithms.evolution_with_app_algorithm import (
    compute_time_evolution_with_app,
)
from examples.plotting_utils import plot_time_evolution_with_app
from math_utilities.config import TAU_UNIT_IN_DAYS
from math_utilities.discrete_distributions_utils import delta_distribution
from model_utilities.epidemic_data import (
    R0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    tauS,
)
from model_utilities.scenarios import ScenarioWithApp

p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()


def time_evolution_with_app_optimistic_scenario_example():
    """
    Example running the full algorithm in the scenario with app usage.
    We take high values for the sensitivities s^s, s^c for people using the app.
    """
    t_max_in_days = 20

    DeltaAT_app_in_days = 2
    DeltaAT_noapp_in_days = 4

    scenario = ScenarioWithApp(
        p_gs=p_gs,
        b0_gs=b0_gs,
        tauS=tauS,
        t_0=0,
        ssapp=(lambda t: 0, lambda t: 0.8),
        ssnoapp=(lambda t: 0, lambda t: 0.2),
        scapp=lambda t: 0.8,
        scnoapp=lambda t: 0.2,
        xi=lambda t: 0.9,
        DeltaATapp=delta_distribution(peak_tau_in_days=DeltaAT_app_in_days),
        DeltaATnoapp=delta_distribution(peak_tau_in_days=DeltaAT_noapp_in_days),
        epsilon_app=lambda t: 0.6,
    )

    (
        t_in_days_list,
        nu,
        nu0,
        Fsigmaapp_infty,
        R,
        R_app,
        R_noapp,
        FT_infty,
        FT_app_infty,
        FT_noapp_infty,
    ) = compute_time_evolution_with_app(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=tuple(d.normalize() for d in b0_gs),
    )

    plot_time_evolution_with_app(
        t_in_days_sequence=t_in_days_list,
        epsilon_app=scenario.epsilon_app,
        Fsigmaapp_infty=Fsigmaapp_infty,
        R_ts=R,
        Rapp_ts=R_app,
        Rnoapp_ts=R_noapp,
        R0=R0,
        FT_ts_infty=FT_infty,
        FT_ts_app_infty=FT_app_infty,
        FT_ts_noapp_infty=FT_noapp_infty,
    )


def time_evolution_with_app_pessimistic_scenario_example():
    """
    Example running the full algorithm in the scenario with app usage.
    We take low values for the sensitivities s^s, s^c for people using the app.
    """
    t_max_in_days = 20

    DeltaAT_app_in_days = 2
    DeltaAT_noapp_in_days = 4

    scenario = ScenarioWithApp(
        p_gs=p_gs,
        b0_gs=b0_gs,
        tauS=tauS,
        t_0=0,
        ssapp=(lambda t: 0, lambda t: 0.2),
        ssnoapp=(lambda t: 0, lambda t: 0.2),
        scapp=lambda t: 0.5,
        scnoapp=lambda t: 0.2,
        xi=lambda t: 0.8,
        DeltaATapp=delta_distribution(peak_tau_in_days=DeltaAT_app_in_days),
        DeltaATnoapp=delta_distribution(peak_tau_in_days=DeltaAT_noapp_in_days),
        epsilon_app=lambda t: 0.6,
    )

    (
        t_in_days_list,
        nu,
        nu0,
        Fsigmaapp_infty,
        R,
        R_app,
        R_noapp,
        FT_infty,
        FT_app_infty,
        FT_noapp_infty,
    ) = compute_time_evolution_with_app(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=tuple(d.normalize() for d in b0_gs),
    )

    plot_time_evolution_with_app(
        t_in_days_sequence=t_in_days_list,
        epsilon_app=scenario.epsilon_app,
        Fsigmaapp_infty=Fsigmaapp_infty,
        R_ts=R,
        Rapp_ts=R_app,
        Rnoapp_ts=R_noapp,
        R0=R0,
        FT_ts_infty=FT_infty,
        FT_ts_app_infty=FT_app_infty,
        FT_ts_noapp_infty=FT_noapp_infty,
    )


def time_evolution_with_app_gradual_app_adoption_example():
    """
    Example running the full algorithm in the scenario with app usage.
    In this scenario, the app adoption increases linearly from 0 to 60% in 30 days.
    """
    t_max_in_days = 50

    DeltaAT_app_in_days = 2
    DeltaAT_noapp_in_days = 4

    scenario = ScenarioWithApp(
        p_gs=p_gs,
        b0_gs=b0_gs,
        tauS=tauS,
        t_0=0,
        ssapp=(lambda t: 0, lambda t: 0.8),
        ssnoapp=(lambda t: 0, lambda t: 0.2),
        scapp=lambda t: 0.8,
        scnoapp=lambda t: 0.2,
        xi=lambda t: 0.9,
        DeltaATapp=delta_distribution(peak_tau_in_days=DeltaAT_app_in_days),
        DeltaATnoapp=delta_distribution(peak_tau_in_days=DeltaAT_noapp_in_days),
        epsilon_app=lambda t: 0.6 * (t * TAU_UNIT_IN_DAYS) / 30
        if t * TAU_UNIT_IN_DAYS < 30
        else 0.6,
    )

    (
        t_in_days_list,
        nu,
        nu0,
        Fsigmaapp_infty,
        R,
        R_app,
        R_noapp,
        FT_infty,
        FT_app_infty,
        FT_noapp_infty,
    ) = compute_time_evolution_with_app(
        scenario=scenario,
        t_max_in_days=t_max_in_days,
        nu_start=1000,
        b_negative_times=tuple(d.normalize() for d in b0_gs),
    )

    plot_time_evolution_with_app(
        t_in_days_sequence=t_in_days_list,
        epsilon_app=scenario.epsilon_app,
        Fsigmaapp_infty=Fsigmaapp_infty,
        R_ts=R,
        Rapp_ts=R_app,
        Rnoapp_ts=R_noapp,
        R0=R0,
        FT_ts_infty=FT_infty,
        FT_ts_app_infty=FT_app_infty,
        FT_ts_noapp_infty=FT_noapp_infty,
    )

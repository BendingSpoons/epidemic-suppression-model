"""
Examples showing how the efficiency of the isolation measures depends on the parameters describing
the isolation measures.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

import numpy as np
from matplotlib import pyplot as plt

from epidemic_suppression_algorithms.evolution_with_app_algorithm import (
    compute_time_evolution_with_app,
)
from epidemic_suppression_algorithms.homogeneous_evolution_algorithm import (
    compute_time_evolution_homogeneous_case,
)
from math_utilities.discrete_distributions_utils import delta_distribution
from math_utilities.general_utilities import effectiveness
from model_utilities.epidemic_data import (
    R0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    tauS,
)
from model_utilities.scenarios import HomogeneousScenario, ScenarioWithApp


def dependency_on_testing_timeliness_homogeneous_model_example():
    """
    Example of several computations of the limit Eff_∞ in homogeneous scenarios
    in which the time interval Δ^{A → T} varies from 0 to 10 days.
    """
    # Severities: gs = [asymptomatic, symptomatic]
    p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()
    t_0 = 0

    DeltaAT_values_list = [i for i in range(10)]
    Effinfty_values_list = []

    for DeltaAT_in_days in DeltaAT_values_list:

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
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=b0_gs,
            verbose=False,
            threshold_to_stop=0.001,
        )

        Rinfty = R[-1]
        Effinfty = effectiveness(Rinfty, R0)
        Effinfty_values_list.append(Effinfty)

    fig = plt.figure(figsize=(10, 15))

    Rinfty_plot = fig.add_subplot(111)
    Rinfty_plot.set_xlabel("Δ^{A → T} (days)")
    Rinfty_plot.set_ylabel("Eff_∞")
    Rinfty_plot.grid(True)
    Rinfty_plot.set_xlim(0, DeltaAT_values_list[-1])
    Rinfty_plot.set_ylim(0, 0.4)
    Rinfty_plot.plot(DeltaAT_values_list, Effinfty_values_list, color="black",),

    plt.show()


def dependency_on_isolation_strength_homogeneous_model_example():
    """
    Example of several computations of the limit Eff_∞ in homogeneous scenarios
    in which the parameter ξ varies from 0 to 10 days.
    """
    # Severities: gs = [asymptomatic, symptomatic]
    p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    xi_values_list = [0.1 * i for i in range(11)]
    Effinfty_values_list = []

    for xi in xi_values_list:

        scenario = HomogeneousScenario(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ss=(lambda t: 0, lambda t: 0.5),
            sc=lambda t: 0.7,
            xi=lambda t: xi,
            DeltaAT=delta_distribution(peak_tau_in_days=2),
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
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=b0_gs,
            verbose=False,
            threshold_to_stop=0.001,
        )

        Rinfty = R[-1]
        Effinfty = effectiveness(Rinfty, R0)
        Effinfty_values_list.append(Effinfty)

    fig = plt.figure(figsize=(10, 15))

    Rinfty_plot = fig.add_subplot(111)
    Rinfty_plot.set_xlabel("ξ")
    Rinfty_plot.set_ylabel("Eff_∞")
    Rinfty_plot.grid(True)
    Rinfty_plot.set_xlim(0, xi_values_list[-1])
    Rinfty_plot.set_ylim(0, 0.4)
    Rinfty_plot.plot(xi_values_list, Effinfty_values_list, color="black",),

    plt.show()


def dependency_on_efficiencies_example():
    """
    Example of several computations of the limit Eff_∞ with app usage,
    where the parameters s^{s,app} and s^{c,app} vary.
    """

    # Severities: gs = (asymptomatic, symptomatic)
    p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()
    ssapp_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    scapp_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    Effinfty_values_list = []

    for scapp in scapp_list:
        Effinfty_values_list_for_scapp = []
        for ssapp in ssapp_list:
            scenario = ScenarioWithApp(
                p_gs=p_gs,
                b0_gs=b0_gs,
                tauS=tauS,
                t_0=0,
                ssapp=(lambda t: 0, lambda t: ssapp),
                ssnoapp=(lambda t: 0, lambda t: 0.2),
                scapp=lambda t: scapp,
                scnoapp=lambda t: 0.2,
                xi=lambda t: 0.9,
                DeltaATapp=delta_distribution(peak_tau_in_days=2),
                DeltaATnoapp=delta_distribution(peak_tau_in_days=4),
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
                t_max_in_days=20,
                nu_start=1000,
                b_negative_times=b0_gs,
                threshold_to_stop=0.001,
            )

            Rinfty = R[-1]
            Effinfty = effectiveness(Rinfty, R0)
            Effinfty_values_list_for_scapp.append(Effinfty)

            print(
                f"s^{{s,app}} = {ssapp}, s^{{c,app}} = {scapp}, Eff_∞ = {round(Effinfty, 2)}"
            )

        Effinfty_values_list.append(Effinfty_values_list_for_scapp)

    ssapp_values_array, scapp_values_array = np.meshgrid(ssapp_list, scapp_list)

    fig = plt.figure(figsize=(10, 15))
    ax = fig.gca(projection="3d")

    ax.plot_surface(
        ssapp_values_array, scapp_values_array, np.array(Effinfty_values_list),
    )

    ax.set_xlabel("s^{s,app}")
    ax.set_ylabel("s^{c,app}")
    ax.set_zlabel("Eff_∞")

    plt.show()


def dependency_on_app_adoption_example():
    """
    Example of several computations of the limit Eff_∞ with app usage, where the fraction p^app of app adopters varies.
    """
    # Severities: gs = (asymptomatic, symptomatic)
    p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

    epsilon_app_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Effinfty_values_list = []

    for epsilon_app in epsilon_app_list:

        scenario = ScenarioWithApp(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ssapp=(lambda t: 0, lambda t: 0.5),
            ssnoapp=(lambda t: 0, lambda t: 0.2),
            scapp=lambda t: 0.7,
            scnoapp=lambda t: 0.2,
            xi=lambda t: 0.9,
            DeltaATapp=delta_distribution(peak_tau_in_days=2),
            DeltaATnoapp=delta_distribution(peak_tau_in_days=4),
            epsilon_app=lambda t: epsilon_app,
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
            t_max_in_days=40,
            nu_start=1000,
            b_negative_times=b0_gs,
            threshold_to_stop=0.001,
        )

        Rinfty = R[-1]
        Effinfty = effectiveness(Rinfty, R0)
        Effinfty_values_list.append(Effinfty)

    fig = plt.figure(figsize=(10, 15))

    Rinfty_plot = fig.add_subplot(111)
    Rinfty_plot.set_xlabel("ϵ_app")
    Rinfty_plot.set_ylabel("Eff_∞")
    Rinfty_plot.grid(True)
    Rinfty_plot.set_xlim(0, 1)
    Rinfty_plot.set_ylim(0, 0.3)
    Rinfty_plot.plot(epsilon_app_list, Effinfty_values_list, color="black",),

    plt.show()

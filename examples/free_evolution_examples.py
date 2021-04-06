"""
Examples printing the evolution of the number of infected people, given the infectiousness
(no isolation measures are included here).

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""


from epidemic_suppression_algorithms.free_evolution_algorithm import (
    free_evolution_by_severity,
    free_evolution_global,
)
from epidemic_suppression_algorithms.model_blocks.nu_and_tausigma import (
    check_b_negative_times,
)
from math_utilities.config import UNITS_IN_ONE_DAY
from model_utilities.epidemic_data import (
    b0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
)

p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()


def free_evolution_without_severities_example():

    b0_scaled = b0.rescale_by_factor(scale_factor=0.9)

    t_max_in_days = 30
    nu_start = 1000

    free_evolution_global(
        b=[b0_scaled] * t_max_in_days * UNITS_IN_ONE_DAY,
        nu_start=nu_start,
        b_negative_times=b0,
    )


def free_evolution_by_severity_example():

    t_max_in_days = 30
    nu_start = 1000

    b0_gs_scaled = tuple(b0_g.rescale_by_factor(scale_factor=0.9) for b0_g in b0_gs)

    check_b_negative_times(p_gs=p_gs, b_negative_times=b0_gs)

    free_evolution_by_severity(
        b=[b0_gs_scaled] * t_max_in_days * UNITS_IN_ONE_DAY,
        nu_start=nu_start,
        p_gs=p_gs,
        b_negative_times=b0_gs,
    )

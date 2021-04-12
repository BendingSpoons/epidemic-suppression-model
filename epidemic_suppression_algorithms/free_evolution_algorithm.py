"""
This file contains functions computing the evolution of an epidemic, given the infectiousness
at each time step.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from typing import List, Optional, Sequence, Tuple

from epidemic_suppression_algorithms.model_blocks.nu_and_tausigma import (
    compute_tausigma_and_nu_at_time_t,
    compute_tausigma_and_nu_components_at_time_t,
)
from math_utilities.config import TAU_UNIT_IN_DAYS, UNITS_IN_ONE_DAY
from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)


def free_evolution_global(
    b: Sequence[DiscreteDistributionOnNonNegatives],
    nu_start: int,
    b_negative_times: Optional[DiscreteDistributionOnNonNegatives] = None,
) -> Tuple[List[int], List[float], List[DiscreteDistributionOnNonNegatives]]:
    nu = []
    R = []
    tausigma = []

    for t in range(0, len(b)):
        t_in_days = t * TAU_UNIT_IN_DAYS

        beta_t = b[t]
        R_t = beta_t.total_mass
        R.append(R_t)

        if t == 0 and b_negative_times is None:
            nu_t = nu_start
            tausigma_t = DiscreteDistributionOnNonNegatives(
                pmf_values=[], tau_min=0, improper=True
            )
        else:
            nu_t, tausigma_t = compute_tausigma_and_nu_at_time_t(
                t=t,
                b=b,
                nu=nu,
                b_negative_times=b_negative_times,
                nu_negative_times=nu_start,
            )

        nu.append(nu_t)
        tausigma.append(tausigma_t)

        if t % UNITS_IN_ONE_DAY == 0:
            print(
                f"""t = {t_in_days} days
                    nu_t = {nu[t]}
                    R_t = {round(R_t, 2)}
                    Fsigma_t(∞) = {tausigma_t.total_mass}
                    """
            )

    return nu, R, tausigma


def free_evolution_by_severity(
    b: Sequence[Tuple[DiscreteDistributionOnNonNegatives, ...]],
    nu_start: int,
    p_gs: Tuple[float, ...],
    b_negative_times: Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]] = None,
) -> Tuple[
    List[int],
    List[Tuple[float, ...]],
    List[Tuple[DiscreteDistributionOnNonNegatives, ...]],
]:
    nu = []
    R = []
    tausigma = []

    gs = range(len(p_gs))  # Values of severity G

    for t in range(0, len(b)):
        t_in_days = t * TAU_UNIT_IN_DAYS
        b_t_gs = b[t]
        R_t_gs = tuple(b_t_g.total_mass for b_t_g in b_t_gs)
        R.append(R_t_gs)

        if t == 0 and b_negative_times is None:
            nu_t = nu_start
            tausigmags_t = tuple(
                DiscreteDistributionOnNonNegatives(
                    pmf_values=[], tau_min=0, improper=True
                )
                for _ in gs
            )
            nu_t_gs = None
        else:
            nu_t_gs, tausigmags_t = compute_tausigma_and_nu_components_at_time_t(
                t=t,
                b=b,
                nu=nu,
                p_gs=p_gs,
                b_negative_times=b_negative_times,
                nu_negative_times=nu_start,
            )
            nu_t = sum(nu_t_gs)

        nu.append(nu_t)
        tausigma.append(tausigmags_t)

        R_t = sum(p_g * R_t_g for (p_g, R_t_g) in zip(p_gs, R_t_gs))

        if t % UNITS_IN_ONE_DAY == 0:
            print(
                f"""t = {t_in_days} days
            nu_t_gs = {tuple(nu_t_gs)},   nu_t = {nu[t]}
            R_t_gs = {R_t_gs},    R_t = {round(R_t, 2)}
            Fsigmags_t(∞) = {tuple(tausigmag_t.total_mass for tausigmag_t in tausigmags_t)}
            """
            )

    return nu, R, tausigma

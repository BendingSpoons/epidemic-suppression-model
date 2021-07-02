"""
Functions that compute the testing time distribution (in terms of the notification times)
and the discretized infectiousness b_t by degree of severity (given the default
infectiousness and the testing time distribution).

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""


from typing import Tuple

from math_utilities.config import TAU_MAX_IN_UNITS
from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
    generate_discrete_distribution_from_cdf_function,
)


def compute_tauT_t(
    tauAs_t_gs: Tuple[DiscreteDistributionOnNonNegatives, ...],
    tauAc_t: DiscreteDistributionOnNonNegatives,
    DeltaAT: DiscreteDistributionOnNonNegatives,
):
    """
    Computes the testing time distributions tauT_t_g from the notification time distributions:
        tauA_t_g = min(tauAs_t_g, tauAc_t),
        tauT_t_g = tauA_t_g + DeltaAT.
    """
    gs = range(len(tauAs_t_gs))
    tauT_t_gs = []
    for g in gs:
        FAs_g = tauAs_t_gs[g].cdf
        FAc_t = tauAc_t.cdf
        # The improper CDF and distribution of tauA_t_g = min(tauAs_g, tauAc_t):
        FA_t_g = lambda tau: FAs_g(tau) + FAc_t(tau) - FAs_g(tau) * FAc_t(tau)
        tauA_t_g = generate_discrete_distribution_from_cdf_function(
            cdf=FA_t_g, tau_min=0, tau_max=TAU_MAX_IN_UNITS,
        )

        tauT_t_g = tauA_t_g + DeltaAT

        tauT_t_gs.append(tauT_t_g)

    return tuple(tauT_t_gs)


def compute_suppressed_b_t(
    b0_t_gs: Tuple[DiscreteDistributionOnNonNegatives, ...],
    tauT_t_gs: Tuple[DiscreteDistributionOnNonNegatives, ...],
    xi_t: float,
) -> Tuple[DiscreteDistributionOnNonNegatives, ...]:
    """
    Computes b_t using the suppression formula:
      b_t_g = b0_t_g (1 - xi_t * FT_{t,g}).
    :param b0_t_gs: the tuple of default discrete infectiousnesses at t, one for each severity g.
    :param tauT_t_gs: the tuple of testing time distributions, one for each g.
    :param xi_t: the suppression factor at time t.
    :return: the tuple of discrete infectiousnesses at t, one for each severity g.
    """
    b_t = []
    for b0_t_g, tau_T_g in zip(b0_t_gs, tauT_t_gs):
        b_t_g = b0_t_g.rescale_by_function(scale_function=lambda tau: 1 - xi_t * tau_T_g.cdf(tau))
        b_t.append(b_t_g)

    return tuple(b_t)

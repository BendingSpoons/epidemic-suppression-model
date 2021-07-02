"""
This file contains the main function implementing the algorithm in the "homogeneous" scenario.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from typing import List, Optional, Tuple

from epidemic_suppression_algorithms.model_blocks.nu_and_tausigma import (
    compute_tausigma_and_nu_components_at_time_t,
)
from epidemic_suppression_algorithms.model_blocks.testing_time_and_b_t_suppression import (
    compute_suppressed_b_t,
    compute_tauT_t,
)
from epidemic_suppression_algorithms.model_blocks.time_evolution_block import compute_tauAc_t
from math_utilities.config import UNITS_IN_ONE_DAY
from math_utilities.discrete_distributions_utils import DiscreteDistributionOnNonNegatives
from model_utilities.scenarios import HomogeneousScenario


def compute_time_evolution_homogeneous_case(
    scenario: HomogeneousScenario,
    t_max_in_days: int,
    nu_start: int,
    b_negative_times: Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]] = None,
    verbose: bool = True,
    threshold_to_stop: Optional[float] = None,
) -> Tuple[
    List[int], List[float], List[float], List[float], List[Tuple[float, ...]], List[float],
]:
    """
    The main function that implements the algorithm, computing the time evolution in the
    "homogeneous" scenario, in which the same tracing and isolation measures apply to the whole
    population.
    Note that the time step Δ𝜏 is determined by the constant
        math_utilities.config.UNITS_IN_ONE_DAY.
    :param scenario: the object gathering the input parameters (epidemic data and
      suppression measures).
    :param t_max_in_days: the maximum number of days for which the algorithm runs
    :param nu_start: the initial number of infected people per time step.
    :param b_negative_times: the optional infectiousness (by degree of severity) at times t<0.
      These improper distribution must be jointly normalized to have total mass 1,
      to ensure that the number of infections per time step is constantly equal to nu_start at t<0.
      If None, the epidemic is assumed to start at t=0.
    :param verbose: if True, the main KPIs are printed for each day.
    :param threshold_to_stop: an optional float that makes the algorithm stop if the reproduction
      number R_t and total testing probability FT_infty_t have had relative variations below
      threshold_to_stop in the previous iteration.
    :return: Several lists of floats, one per time step:
    - t_in_days_list: the absolute times 0, Δ𝜏, 2Δ𝜏,...
    - nu: the number of infections at each time step.
    - nu0: the number of infections at each time step, if there were no isolation measures.
    - R: the effective reproduction numbers.
    - R_by_severity: the tuples of effective reproduction numbers by degree of severity,
    - FT_infty: the probabilities to be eventually tested positive.
    """
    #
    t_in_days_list = []
    nu = []
    nu0 = []
    b: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    R_by_severity: List[Tuple[float, ...]] = []
    R: List[float] = []
    tausigma: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    tauT: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    FT_infty: List[float, ...] = []

    gs = range(scenario.n_severities)  # Values of severity G

    t_max = t_max_in_days * UNITS_IN_ONE_DAY
    for t in range(0, t_max + 1):
        t_in_days = t / UNITS_IN_ONE_DAY

        # Compute tausigma_t and nu_t from nu_t' and b_t' for t' = 0,...,t-1
        if t == 0 and b_negative_times is None:
            nu_t = nu_start
            nugs_t = tuple(nu_t * p_g for p_g in scenario.p_gs)
            nu0_t = nu_start
            tausigmags_t = tuple(
                DiscreteDistributionOnNonNegatives(pmf_values=[], tau_min=0, improper=True)
                for _ in gs
            )
        else:
            nugs_t, tausigmags_t = compute_tausigma_and_nu_components_at_time_t(
                t=t,
                b=b,
                nu=nu,
                p_gs=scenario.p_gs,
                b_negative_times=b_negative_times,
                nu_negative_times=nu_start,
            )
            nugs0_t, _ = compute_tausigma_and_nu_components_at_time_t(
                t=t,
                b=[scenario.b0_gs] * t,
                nu=nu0,
                p_gs=scenario.p_gs,
                b_negative_times=b_negative_times,
                nu_negative_times=nu_start,
            )

            nu_t = sum(nugs_t)  # People infected at t
            nu0_t = sum(nugs0_t)  # People infected at t without isolation measures

            if nu_t < 0.5:  # Breaks the loop when nu_t = 0
                break

        # Compute tauAs_t components from tauS
        tauAs_t_gs = tuple(scenario.tauS.rescale_by_factor(scenario.ss[g](t)) for g in gs)

        # Time evolution step:
        # Compute tauAc_t from tausigma_t and tauT_t' (for t' = 0,...,t-1) components
        tauAc_t = compute_tauAc_t(
            t=t, tauT=tauT, tausigmags_t=tausigmags_t, xi=scenario.xi, sc_t=scenario.sc(t),
        )

        # Compute tauA_t and tauT_t components from tauAs_t, tauAc_t, and DeltaAT
        tauT_t_gs = compute_tauT_t(tauAs_t_gs=tauAs_t_gs, tauAc_t=tauAc_t, DeltaAT=scenario.DeltaAT)

        # Compute b and R
        b_t_gs = compute_suppressed_b_t(
            b0_t_gs=scenario.b0_gs, tauT_t_gs=tauT_t_gs, xi_t=scenario.xi(t)
        )
        R_t_gs = tuple(b_t_g.total_mass for b_t_g in b_t_gs)
        R_t = sum(p_g * R_t_g for (p_g, R_t_g) in zip(scenario.p_gs, R_t_gs))
        FT_t_infty = sum(
            p_g * tauT_t_g.total_mass for (p_g, tauT_t_g) in zip(scenario.p_gs, tauT_t_gs)
        )

        t_in_days_list.append(t_in_days)
        tausigma.append(tausigmags_t)
        nu.append(nu_t)
        nu0.append(nu0_t)
        b.append(b_t_gs)
        R.append(R_t)
        R_by_severity.append(R_t_gs)
        tauT.append(tauT_t_gs)
        FT_infty.append(FT_t_infty)

        if verbose and t % UNITS_IN_ONE_DAY == 0:
            EtauC_t_gs_in_days = [b_t_g.normalize().mean() * UNITS_IN_ONE_DAY for b_t_g in b_t_gs]

            print(
                f"""t = {t_in_days} days
                    nugs_t = {tuple(nugs_t)},   nu_t = {int(round(nu_t, 0))}
                    nu0_t = {int(round(nu0_t, 0))}
                    R_t_gs = {R_t_gs},    R_t = {round(R_t, 2)}
                    EtauC_t_gs = {tuple(EtauC_t_gs_in_days)} days
                    Fsigmags_t(∞) = {tuple(tausigmag_t.total_mass for tausigmag_t in tausigmags_t)}
                    FAs_t_gs(∞) = {tuple(tauAs_t_g.total_mass for tauAs_t_g in tauAs_t_gs)}
                    FAc_t(∞) = {tauAc_t.total_mass}
                    FT_t_gs(∞) = {tuple(tauT_t_g.total_mass for tauT_t_g in tauT_t_gs)},   FT_t(∞) = {round(FT_t_infty, 2)}
                    tauT_t_gs_mean = {tuple(tauT_t_g.normalize().mean() if tauT_t_g.total_mass > 0 else None for tauT_t_g in tauT_t_gs)},
                    """
            )

        if (
            threshold_to_stop is not None
            and t > 10
            and (
                abs((R[-2] - R[-1]) / R[-2]) < threshold_to_stop
                and FT_infty[-2] != 0
                and abs((FT_infty[-2] - FT_infty[-1]) / FT_infty[-2]) < threshold_to_stop
            )
        ):
            break

    return t_in_days_list, nu, nu0, R, R_by_severity, FT_infty

"""
This file contains the main function implementing the algorithm in the scenario with app usage.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from typing import List, Optional, Tuple

from epidemic_suppression_algorithms.model_blocks.nu_and_tausigma import (
    compute_tausigma_and_nu_components_at_time_t,
    compute_tausigma_and_nu_components_at_time_t_with_app,
)
from epidemic_suppression_algorithms.model_blocks.testing_time_and_b_t_suppression import (
    compute_suppressed_b_t,
    compute_tauT_t,
)
from epidemic_suppression_algorithms.model_blocks.time_evolution_block import (
    compute_tauAc_t_two_components,
)
from math_utilities.config import UNITS_IN_ONE_DAY
from math_utilities.discrete_distributions_utils import DiscreteDistributionOnNonNegatives
from model_utilities.scenarios import ScenarioWithApp


def compute_time_evolution_with_app(
    scenario: ScenarioWithApp,
    t_max_in_days: int,
    nu_start: int,
    b_negative_times: Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]] = None,
    verbose: bool = True,
    threshold_to_stop: Optional[float] = None,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    """
    The main function that implements the algorithm, computing the time evolution in the scenario
    in which an app for epidemic control is used.
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
    - Fsigmaapp_infty: the probabilities that one's infector was using the app.
    - R: the effective reproduction numbers.
    - R_app: the effective reproduction numbers for people with the app.
    - R_noapp: the effective reproduction numbers for people without the app.
    - FT_infty: the probabilities to be eventually tested positive.
    - FT_app_infty: the probabilities to be eventually tested positive, for people with the app.
    - FT_noapp_infty: the probabilities to be eventually tested positive, for people without
      the app.
    """
    #
    t_in_days_list: List[float] = []
    nu = []
    nu_app = []
    nu_noapp = []
    nu0 = []
    tausigma_app: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    tausigma_noapp: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    Fsigmaapp_infty: List[float] = []
    b_app: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    b_noapp: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    R_app: List[float] = []
    R_noapp: List[float] = []
    R: List[float] = []
    tauT_app: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    tauT_noapp: List[Tuple[DiscreteDistributionOnNonNegatives, ...]] = []
    FT_infty: List[float, ...] = []
    FT_app_infty: List[float, ...] = []
    FT_noapp_infty: List[float, ...] = []

    gs = range(scenario.n_severities)  # Values of severity G

    t_max = t_max_in_days * UNITS_IN_ONE_DAY
    for t in range(0, t_max + 1):
        t_in_days = t / UNITS_IN_ONE_DAY

        pgs_t_app = tuple(p_g * scenario.epsilon_app(t) for p_g in scenario.p_gs)
        pgs_t_noapp = tuple(p_g * (1 - scenario.epsilon_app(t)) for p_g in scenario.p_gs)

        # Compute tausigma_t and nu_t from nu_t' and b_t' for t' = 0,...,t-1
        if t == 0 and b_negative_times is None:

            nugsapp_t = tuple(nu_start * p_g for p_g in pgs_t_app)
            nugsnoapp_t = tuple(nu_start * p_g for p_g in pgs_t_noapp)
            nu0_t = nu_start
            tausigmagsapp_t = tausigmagsnoapp_t = tuple(
                DiscreteDistributionOnNonNegatives(pmf_values=[], tau_min=0, improper=True)
                for _ in gs
            )
        else:
            (
                nugsapp_t,
                tausigmagsapp_t,
                nugsnoapp_t,
                tausigmagsnoapp_t,
            ) = compute_tausigma_and_nu_components_at_time_t_with_app(
                t=t,
                b_app=b_app,
                b_noapp=b_noapp,
                nu=nu,
                p_gs=scenario.p_gs,
                epsilon_app=scenario.epsilon_app,
                b_negative_times=b_negative_times,
                nu_negative_times=nu_start,
            )

            nu0_t_gs, _ = compute_tausigma_and_nu_components_at_time_t(
                t=t,
                b=[scenario.b0_gs] * t,
                nu=nu0,
                p_gs=scenario.p_gs,
                b_negative_times=b_negative_times,
                nu_negative_times=nu_start,
            )
            nu0_t = sum(nu0_t_gs)  # People infected at t without isolation measures
        # Prob. that infector had the app
        Fsigmaapp_t_infty = sum(tausigmag_t.total_mass for tausigmag_t in tausigmagsapp_t)
        nuapp_t = sum(nugsapp_t)
        nunoapp_t = sum(nugsnoapp_t)

        nu_t = nuapp_t + nunoapp_t  # People infected at t

        if nu_t < 0.5:  # Breaks the loop when nu_t = 0
            break

        # Compute tauAs_t components from tauS
        tauAs_t_gs_app = tuple(scenario.tauS.rescale_by_factor(scenario.ssapp[g](t)) for g in gs)
        tauAs_t_gs_noapp = tuple(
            scenario.tauS.rescale_by_factor(scenario.ssnoapp[g](t)) for g in gs
        )

        # Time evolution step:
        # Compute tauAc_t from tausigma_t and tauT_t' (for t' = 0,...,t-1) components
        tauAc_t_app, tauAc_t_noapp = compute_tauAc_t_two_components(
            t=t,
            tauT_app=tauT_app,
            tauT_noapp=tauT_noapp,
            tausigmagsapp_t=tausigmagsapp_t,
            tausigmagsnoapp_t=tausigmagsnoapp_t,
            xi=scenario.xi,
            scapp_t=scenario.scapp(t),
            scnoapp_t=scenario.scnoapp(t),
        )

        # Compute tauA_t and tauT_t components from tauAs_t, tauAc_t, and DeltaAT
        tauT_t_gs_app = compute_tauT_t(
            tauAs_t_gs=tauAs_t_gs_app, tauAc_t=tauAc_t_app, DeltaAT=scenario.DeltaATapp
        )
        tauT_t_gs_noapp = compute_tauT_t(
            tauAs_t_gs=tauAs_t_gs_noapp, tauAc_t=tauAc_t_noapp, DeltaAT=scenario.DeltaATnoapp,
        )

        # Compute b and R
        b_t_gs_app = compute_suppressed_b_t(
            b0_t_gs=scenario.b0_gs, tauT_t_gs=tauT_t_gs_app, xi_t=scenario.xi(t)
        )
        b_t_gs_noapp = compute_suppressed_b_t(
            b0_t_gs=scenario.b0_gs, tauT_t_gs=tauT_t_gs_noapp, xi_t=scenario.xi(t)
        )
        R_t_gs_app = tuple(b_t_g.total_mass for b_t_g in b_t_gs_app)
        R_t_gs_noapp = tuple(b_t_g.total_mass for b_t_g in b_t_gs_noapp)
        R_t_app = sum(p_g * R_t_g for (p_g, R_t_g) in zip(scenario.p_gs, R_t_gs_app))
        R_t_noapp = sum(p_g * R_t_g for (p_g, R_t_g) in zip(scenario.p_gs, R_t_gs_noapp))
        R_t = scenario.epsilon_app(t) * R_t_app + (1 - scenario.epsilon_app(t)) * R_t_noapp
        FT_t_app_infty = sum(
            p_g * tauT_t_g_app.total_mass
            for (p_g, tauT_t_g_app, tauT_t_g_noapp) in zip(
                scenario.p_gs, tauT_t_gs_app, tauT_t_gs_noapp
            )
        )
        FT_t_noapp_infty = sum(
            p_g * tauT_t_g_noapp.total_mass
            for (p_g, tauT_t_g_app, tauT_t_g_noapp) in zip(
                scenario.p_gs, tauT_t_gs_app, tauT_t_gs_noapp
            )
        )
        FT_t_infty = (
            scenario.epsilon_app(t) * FT_t_app_infty
            + (1 - scenario.epsilon_app(t)) * FT_t_noapp_infty
        )

        t_in_days_list.append(t_in_days)
        tausigma_app.append(tausigmagsapp_t)
        tausigma_noapp.append(tausigmagsnoapp_t)
        Fsigmaapp_infty.append(Fsigmaapp_t_infty)
        nu.append(nu_t)
        nu_app.append(nuapp_t)
        nu_noapp.append(nunoapp_t)
        nu0.append(nu0_t)
        b_app.append(b_t_gs_app)
        b_noapp.append(b_t_gs_noapp)
        R_app.append(R_t_app)
        R_noapp.append(R_t_noapp)
        R.append(R_t)
        tauT_app.append(tauT_t_gs_app)
        tauT_noapp.append(tauT_t_gs_noapp)
        FT_infty.append(FT_t_infty)
        FT_app_infty.append(FT_t_app_infty)
        FT_noapp_infty.append(FT_t_noapp_infty)

        if verbose and t % UNITS_IN_ONE_DAY == 0:
            EtauC_t_gs_app_in_days = [
                b_t_g.normalize().mean() * UNITS_IN_ONE_DAY for b_t_g in b_t_gs_app
            ]
            EtauC_t_gs_noapp_in_days = [
                b_t_g.normalize().mean() * UNITS_IN_ONE_DAY for b_t_g in b_t_gs_noapp
            ]

            print(
                f"""t = {t_in_days} days
                    nugsapp_t = {tuple(nugsapp_t)},   nugsnoapp_t = {tuple(nugsnoapp_t)},   nu_t = {int(round(nu_t, 0))}
                    nu0_t = {int(round(nu0_t, 0))}
                    R_t_gs_app = {R_t_gs_app},    R_t_app = {R_t_app},    
                    R_t_gs_noapp = {R_t_gs_noapp},    R_t_noapp = {R_t_noapp},        
                    R_t = {round(R_t, 2)}
                    EtauC_t_gs_app = {tuple(EtauC_t_gs_app_in_days)} days
                    EtauC_t_gs_noapp = {tuple(EtauC_t_gs_noapp_in_days)} days
                    Fsigmagsapp_t(∞) = {tuple(tausigmag_t.total_mass for tausigmag_t in tausigmagsapp_t)}
                    Fsigmagsnoapp_t(∞) = {tuple(tausigmag_t.total_mass for tausigmag_t in tausigmagsnoapp_t)}
                    Fsigmaapp_t(∞) = {Fsigmaapp_t_infty}
                    FAs_t_gs_app(∞) = {tuple(tauAs_t_g_app.total_mass for tauAs_t_g_app in tauAs_t_gs_app)}
                    FAs_t_gs_noapp(∞) = {tuple(tauAs_t_g_noapp.total_mass for tauAs_t_g_noapp in tauAs_t_gs_noapp)}
                    FAc_t_app(∞) = {tauAc_t_app.total_mass},    FAc_t_noapp(∞) = {tauAc_t_noapp.total_mass}
                    FT_t_gs_app(∞) = {tuple(tauT_t_g.total_mass for tauT_t_g in tauT_t_gs_app)},
                    FT_t_app(∞) = {FT_t_app_infty},
                    FT_t_gs_noapp(∞) = {tuple(tauT_t_g.total_mass for tauT_t_g in tauT_t_gs_noapp)},
                    FT_t_noapp(∞) = {FT_t_noapp_infty}
                    FT_t(∞) = {round(FT_t_infty, 2)}
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

    return (
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
    )

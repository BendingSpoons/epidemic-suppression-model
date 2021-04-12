"""
Functions that compute the number nu of infected people per time step (for each degree of severity),
and the probability distribution of the infection time and degree of severity of the infector.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from typing import Callable, List, Optional, Sequence, Tuple

from math_utilities.config import DISTRIBUTION_NORMALIZATION_TOLERANCE
from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)


def check_b_negative_times(
    p_gs: Tuple[float, ...],
    b_negative_times: Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]],
):
    """
    Checks the normalization of the distributions b_t_g at t<0.
    """
    gs = range(len(p_gs))
    R_neg_gs = tuple(b_negative_times[g].total_mass for g in gs)
    R_neg = sum(pg * R_neg_g for (pg, R_neg_g) in zip(p_gs, R_neg_gs))
    discrepancy = abs(R_neg - 1)
    assert (
        discrepancy < DISTRIBUTION_NORMALIZATION_TOLERANCE
    ), "The negative times infectiousnesses should be normalized to give R_t = 1"


def compute_tausigma_and_nu_components_at_time_t(
    t: int,
    b: Sequence[Tuple[DiscreteDistributionOnNonNegatives, ...]],
    nu: List[int],
    p_gs: Tuple[float, ...],
    b_negative_times: Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]] = None,
    nu_negative_times: Optional[int] = None,
) -> Tuple[Tuple[float, ...], Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]]]:
    """
    Computes, for each g:
    - The number nug_t of people infected at t by someone with severity g
    - The improper distribution tausigmag_t of probabilities to be infected by someone with
      severity g that was infected at a given time prior to t

    :param t: the absolute time at which we want to compute tausigma.
    :param b: the sequence of discretized infectiousnesses from time 0 to at least t-1.
    :param nu: the number of infected people per time step, from time 0 to t-1.
    :param p_gs: the tuple of fractions of infected people with given severity.
    :param b_negative_times: the (optional) tuple of discretized infectiousness distributions (one
     for each severity) at times t<0. They should be normalized in such a way that R
     at negative times is 1, to give meaningful results (as we assume that nu at negative times is
     constantly equal to nu_0).
    :param nu_negative_times: the (constant) number of people infected at a time t<0.
    :return: The tuples (nug_t) and (tausigmag_t).
    """
    if t == 0 and b_negative_times is None:
        raise ValueError("No computation can be done at t=0 without past data.")
    if nu_negative_times and t > 0:
        assert abs(nu[0] - nu_negative_times) < DISTRIBUTION_NORMALIZATION_TOLERANCE
    assert t <= len(b) and t == len(nu)

    gs = range(len(p_gs))

    mgs_t = []
    nugs_t = []

    for g in gs:
        # Let mg_t(tau) be the number of people infected at t by someone infected on t - tau
        # with severity g. We create a list m_t_g_pmf_values of these numbers for tau = 1,2,...
        mg_t_pmf_values = [
            p_gs[g] * b[t - tau][g].pmf(tau) * nu[t - tau] for tau in range(1, t + 1)
        ]
        if b_negative_times is not None and nu_negative_times is not None:
            tau_max_beta_negative = b_negative_times[g].tau_max
            mg_t_pmf_values += [
                p_gs[g] * b_negative_times[g].pmf(tau) * nu_negative_times
                for tau in range(t + 1, tau_max_beta_negative + 1)
            ]
        mg_t = DiscreteDistributionOnNonNegatives(
            pmf_values=mg_t_pmf_values, tau_min=1, improper=True
        )
        # Now nug_t (number of people infected at t by someone with severity g) is the sum of
        # mg_t(tau) for all tau:
        nug_t = mg_t.total_mass
        nugs_t.append(nug_t)
        mgs_t.append(mg_t)

    nu_t = sum(nugs_t)  # People infected at t
    if nu_t == 0:
        return tuple(nugs_t), None
    tausigmags_t = [mg_t.rescale_by_factor(1 / nu_t) for mg_t in mgs_t]

    # Check that tausigma_t is correctly normalized (for t = 0, this also checks that nu_t is
    # nu_negative_times):
    discrepancy = abs(sum(tausigmag_t.total_mass for tausigmag_t in tausigmags_t) - 1)
    assert discrepancy < DISTRIBUTION_NORMALIZATION_TOLERANCE

    return tuple(nugs_t), tuple(tausigmags_t)


def compute_tausigma_and_nu_at_time_t(
    t: int,
    b: Sequence[DiscreteDistributionOnNonNegatives],
    nu: List[int],
    b_negative_times: Optional[DiscreteDistributionOnNonNegatives] = None,
    nu_negative_times: Optional[int] = None,
) -> Tuple[float, DiscreteDistributionOnNonNegatives]:
    """
    Computes, for each g:
    - The number nu_t of people infected at t
    - The distribution tausigma_t of probabilities to be infected by someone
      that was infected at a given time prior to t
    """
    nu_t_gs, tausigmags_t = compute_tausigma_and_nu_components_at_time_t(
        t=t,
        b=[(b_t,) for b_t in b],
        nu=nu,
        p_gs=(1,),
        b_negative_times=(b_negative_times,),
        nu_negative_times=nu_negative_times,
    )
    return nu_t_gs[0], tausigmags_t[0]


def compute_tausigma_and_nu_components_at_time_t_with_app(
    t: int,
    b_app: Sequence[Tuple[DiscreteDistributionOnNonNegatives, ...]],
    b_noapp: Sequence[Tuple[DiscreteDistributionOnNonNegatives, ...]],
    nu: List[int],
    p_gs: Tuple[float, ...],
    epsilon_app: Callable[[int], float],
    b_negative_times: Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]] = None,
    nu_negative_times: Optional[int] = None,
) -> Tuple[
    Tuple[float, ...],
    Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]],
    Tuple[float, ...],
    Optional[Tuple[DiscreteDistributionOnNonNegatives, ...]],
]:
    """
    Computes, for each g, a:
    - The number nuga_t of people infected at t by someone with severity g and app/no app status a
    - The improper distribution tausigmaga_t of probabilities to be infected by someone with
      severity g app/no app status a,that was infected at a given time prior to t

    :param t: the absolute time at which we want to compute tausigma.
    :param b_app: the sequence of discretized infectiousnesses from time 0 to at least t-1,
      for individuals with the app.
    :param b_noapp: the sequence of discretized infectiousnesses from time 0 to at least t-1,
      for individuals without the app.
    :param nu: the number of infected people per time step, from time 0 to t-1.
    :param p_gs: the tuple of fractions of infected people with given severity.
    :param epsilon_app: the probability that an individual infected at t has the app,
    as a function of t.
    :param b_negative_times: the (optional) tuple of discretized infectiousness distributions (one
     for each severity) at times t<0. They should be normalized in such a way that R
     at negative times is 1, to give meaningful results (as we assume that nu at negative times is
     constantly equal to nu_0).
    :param nu_negative_times: the (constant) number of people infected at a time t<0.
    :return: The tuples (nuga_t) and (tausigmaga_t), for a = app, no app.
    """
    if t == 0 and b_negative_times is None:
        raise ValueError("No computation can be done at t=0 without past data.")
    if nu_negative_times and t > 0:
        assert abs(nu[0] - nu_negative_times) < DISTRIBUTION_NORMALIZATION_TOLERANCE
    assert t <= len(b_app) and t <= len(b_noapp) and t == len(nu)

    gs = range(len(p_gs))

    # app/no app status: a = [0, 1] = [app, noapp]
    p_as = [epsilon_app, lambda t: 1 - epsilon_app(t)]
    b = [b_app, b_noapp]

    mgsas_t = [[], []]
    nugsas_t = [[], []]
    for a in [0, 1]:
        for g in gs:
            # Let mga_t(tau) be the number of people infected at t by someone infected on t - tau
            # with severity g and app usage a. We create a list mga_t_pmf_values of
            # these numbers for tau = 1,2,...
            mga_t_pmf_values = [
                p_as[a](t - tau) * p_gs[g] * nu[t - tau] * b[a][t - tau][g].pmf(tau)
                for tau in range(1, t + 1)
            ]
            if b_negative_times is not None and nu_negative_times is not None:
                tau_max_beta_negative = b_negative_times[g].tau_max
                p_g_a_negative_time = p_gs[g] if a == 1 else 0
                mga_t_pmf_values += [
                    p_g_a_negative_time
                    * nu_negative_times
                    * b_negative_times[g].pmf(tau)
                    for tau in range(t + 1, tau_max_beta_negative + 1)
                ]
            mga_t = DiscreteDistributionOnNonNegatives(
                pmf_values=mga_t_pmf_values, tau_min=1, improper=True
            )
            # Now nuga_t (number of people infected at t by someone with severity g and app usage a)
            # is the sum of mga_t(tau) for all tau:
            nuga_t = mga_t.total_mass
            nugsas_t[a].append(nuga_t)
            mgsas_t[a].append(mga_t)

    nu_t = sum(nugsas_t[0]) + sum(nugsas_t[1])  # People infected at t
    if nu_t == 0:
        return tuple(nugsas_t[0]), None, tuple(nugsas_t[1]), None
    tausigmagsapp_t = [mg_t.rescale_by_factor(1 / nu_t) for mg_t in mgsas_t[0]]
    tausigmagsnoapp_t = [mg_t.rescale_by_factor(1 / nu_t) for mg_t in mgsas_t[1]]

    # Check that tausigma_t is correctly normalized (for t = 0, this also checks that nu_t is
    # nu_negative_times):
    discrepancy = abs(
        sum(tausigmag_t.total_mass for tausigmag_t in tausigmagsapp_t)
        + sum(tausigmag_t.total_mass for tausigmag_t in tausigmagsnoapp_t)
        - 1
    )
    assert discrepancy < DISTRIBUTION_NORMALIZATION_TOLERANCE

    return (
        tuple(nugsas_t[0]),
        tuple(tausigmagsapp_t),
        tuple(nugsas_t[1]),
        tuple(tausigmagsnoapp_t),
    )

"""
Functions that implement the time evolution step of the algorithm, computing the distribution
tauAc_t of the notification time in terms of the distribution of the testing time at
previous time steps.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from typing import List, Tuple

from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
    FunctionOfTimeUnit,
    generate_discrete_distribution_from_cdf_function,
    linear_combination_discrete_distributions_by_values,
)


def compute_cut_checkF_t(
    t: int,
    tauT: List[Tuple[DiscreteDistributionOnNonNegatives, ...]],
    xi: FunctionOfTimeUnit,
    measures: Tuple[DiscreteDistributionOnNonNegatives, ...],
) -> FunctionOfTimeUnit:
    """
    Returns the function
        cut_checkFT_t(rho) := checkF_t(rho) - checkF_t(0),
    computed as
        cut_checkFT_t(rho) =
            sum_g
                int_(0,infty)
                    ( FT_{t - tau,g}(rho + tau) - FT_{t - tau,g}(tau)) /
                        (1 - xi(t - tau) * FT_{t - tau,g}(tau))
                dmeasure_g(tau)
    """
    assert t == len(tauT)

    gs = range(len(measures))

    def cut_checkFT_t(rho: int) -> float:
        if rho < 0:
            return 0
        result = 0
        for g in gs:

            def integrand(tau: int, g=g) -> float:
                if t - tau < 0:
                    return 0
                num = tauT[t - tau][g].cdf(rho + tau) - tauT[t - tau][g].cdf(tau)
                den = 1 - xi(t - tau) * tauT[t - tau][g].cdf(tau)
                return num / den

            result += measures[g].integrate(integrand=integrand)
        return result

    return cut_checkFT_t


def compute_tauAc_t(
    t: int,
    tauT: List[Tuple[DiscreteDistributionOnNonNegatives, ...]],
    tausigmags_t: Tuple[DiscreteDistributionOnNonNegatives, ...],
    xi: FunctionOfTimeUnit,
    sc_t: float,
) -> DiscreteDistributionOnNonNegatives:
    """
    Implements the time evolution equation, by computing the distribution of the relative time
    tauAc_t at which someone infected at t receives a risk notification. This is computed from the
    testing times tauT_t'_g for t'<t, that are averaged with weights given by the distributions
    tausigmag_t.

    :param t: the absolute time at which tauAc_t is computed
    :param tauT: the list of distributions tauT_t'_g (one tuple for each t'=0,...,t-1)
    :param tausigmags_t: the tuple for distributions of the infection time and severity
     of the source.
    :param xi: the suppression factor, as a function of absolute time t.
    :param sc_t: the contact-tracing sensitivity at absolute time t.
    :return: the distribution of tauAc_t.
    """
    if t == 0:
        return DiscreteDistributionOnNonNegatives(pmf_values=[], tau_min=0, improper=True)

    gs = range(len(tausigmags_t))

    rho_max = max(max(tauT_t[g].tau_max for g in gs) for tauT_t in tauT)

    # The function rho -> checkFT_t(rho) - checkFT_t(0)
    cut_checkFT_t = compute_cut_checkF_t(t=t, tauT=tauT, xi=xi, measures=tausigmags_t)

    def FAc_t(rho: int) -> float:
        """
        The function
        FAc_t(rho) = sc_t(checkFT_t(rho) - checkFT_t(0))
        """
        return sc_t * cut_checkFT_t(rho)

    tauAc_t = generate_discrete_distribution_from_cdf_function(
        cdf=FAc_t, tau_min=1, tau_max=rho_max,
    )
    return tauAc_t


def compute_tauAc_t_two_components(
    t: int,
    tauT_app: List[Tuple[DiscreteDistributionOnNonNegatives, ...]],
    tauT_noapp: List[Tuple[DiscreteDistributionOnNonNegatives, ...]],
    tausigmagsapp_t: Tuple[DiscreteDistributionOnNonNegatives, ...],
    tausigmagsnoapp_t: Tuple[DiscreteDistributionOnNonNegatives, ...],
    xi: FunctionOfTimeUnit,
    scapp_t: float,
    scnoapp_t: float,
) -> Tuple[DiscreteDistributionOnNonNegatives, DiscreteDistributionOnNonNegatives]:
    """
    Implements the time evolution equation in the two-components scenario,
    by computing the distribution of the relative time tauAc_t at which someone
    infected at t receives a risk notification. This is computed from the
    testing times tauT_t'_g for t'<t, that are averaged with weights given by the distributions
    tausigmag_t.

    :param t: the absolute time at which tauAc_t is computed
    :param tauT: the list of distributions tauT_t'_g (one tuple for each t'=0,...,t-1)
    :param tausigmags_t: the tuple for distributions of the infection time and severity
     of the source.
    :param xi: the suppression factor, as a function of absolute time t.
    :param sc: the contact-tracing sensitivity, as a function of absolute time t.
    :return: the distribution of tauAc_t.
    """
    if t == 0:
        return (
            DiscreteDistributionOnNonNegatives(pmf_values=[], tau_min=0, improper=True),
            DiscreteDistributionOnNonNegatives(pmf_values=[], tau_min=0, improper=True),
        )
    gs = range(len(tausigmagsapp_t))

    # The improper distribution with CDF rho -> checkFTapp_t(rho) - checkFTapp_t(0)
    cut_checkFTapp_t = compute_cut_checkF_t(t=t, tauT=tauT_app, xi=xi, measures=tausigmagsapp_t)
    rho_max_app = max(max(tauT_t[g].tau_max for g in gs) for tauT_t in tauT_app)
    cut_check_tauTapp_t = generate_discrete_distribution_from_cdf_function(
        cdf=cut_checkFTapp_t, tau_min=1, tau_max=rho_max_app
    )

    # The improper distribution with CDF rho -> checkFTnoapp_t(rho) - checkFTnoapp_t(0)
    cut_checkFTnoapp_t = compute_cut_checkF_t(
        t=t, tauT=tauT_noapp, xi=xi, measures=tausigmagsnoapp_t
    )
    rho_max_noapp = max(max(tauT_t[g].tau_max for g in gs) for tauT_t in tauT_noapp)
    cut_check_tauTnoapp_t = generate_discrete_distribution_from_cdf_function(
        cdf=cut_checkFTnoapp_t, tau_min=1, tau_max=rho_max_noapp
    )

    tauAc_t_app = linear_combination_discrete_distributions_by_values(
        scalars=[scapp_t, scnoapp_t],
        seq=[cut_check_tauTapp_t, cut_check_tauTnoapp_t],
        use_cdfs=True,
    )
    tauAc_t_noapp = linear_combination_discrete_distributions_by_values(
        scalars=[scnoapp_t, scnoapp_t],
        seq=[cut_check_tauTapp_t, cut_check_tauTnoapp_t],
        use_cdfs=True,
    )

    return tauAc_t_app, tauAc_t_noapp

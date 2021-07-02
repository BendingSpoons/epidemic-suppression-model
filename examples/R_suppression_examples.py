"""
Some examples of computations of the suppressed effective reproduction number,
given the default infectiousness and the distribution of the time of
positive testing for an infected individual.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from epidemic_suppression_algorithms.model_blocks.testing_time_and_b_t_suppression import (
    compute_suppressed_b_t,
)
from examples.plotting_utils import plot_discrete_distributions
from math_utilities.config import TAU_MAX_IN_UNITS, UNITS_IN_ONE_DAY
from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
    generate_discrete_distribution_from_cdf_function,
)
from model_utilities.epidemic_data import b0, tauS

tau_max = 30
step = 0.05


def R_suppression_with_fixed_testing_time():
    """
    Example computing the suppressed infectiousness and R, assuming that all individuals are tested
    at a given infectious age tau_s.
    """
    tau_s_in_days = 7

    tauT = DiscreteDistributionOnNonNegatives(
        pmf_values=[1], tau_min=tau_s_in_days * UNITS_IN_ONE_DAY, improper=True,
    )

    xi = 1.0  # Probability of (immediate) isolation given positive test

    suppressed_b0 = compute_suppressed_b_t(b0_t_gs=(b0,), tauT_t_gs=(tauT,), xi_t=xi,)[0]
    suppressed_R_0 = suppressed_b0.total_mass

    print("suppressed R_0 =", suppressed_R_0)
    plot_discrete_distributions(ds=[b0, suppressed_b0], custom_labels=["β^0", "β"])


def R_suppression_due_to_symptoms_only():
    """
    Example computing the suppressed infectiousness and R, given that the testing time CDF F^T
    is obtained by translating and rescaling the symptoms onset CDF F^S.
    """
    Deltat_test = 4
    ss = 0.2

    FT = lambda tau: ss * tauS.cdf(tau - Deltat_test)  # CDF of testing time
    xi = 1.0  # Probability of (immediate) isolation given positive test

    tauT = generate_discrete_distribution_from_cdf_function(
        cdf=FT, tau_min=0, tau_max=TAU_MAX_IN_UNITS,
    )

    suppressed_b0 = compute_suppressed_b_t(b0_t_gs=(b0,), tauT_t_gs=(tauT,), xi_t=xi,)[0]
    suppressed_R_0 = suppressed_b0.total_mass

    print("suppressed R_0 =", suppressed_R_0)
    plot_discrete_distributions(ds=[b0, suppressed_b0], custom_labels=["β^0", "β"])

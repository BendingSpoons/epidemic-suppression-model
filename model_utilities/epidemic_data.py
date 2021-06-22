"""
This file contains epidemic data specific of COVID-19 used in the calculations as inputs.
Note that we normalize the "default" effective reproduction number R0 to 1,
as we are only interested in the relative suppression. On the other hand, the
breakdown of R0 in components and the shape of the infectiousness beta0 are taken
from the literature, as well as the incubation period distribution.
Below we consider a "two components model", in which the infected individuals
are only divided, by illness severity, into symptomatic and asymptomatic.
The source for all the numeric values is Ferretti et al. "Quantifying SARS-CoV-2 transmission
suggests epidemic control with digital contact tracing",
https://science.sciencemag.org/content/368/6491/eabb6936

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""
from typing import Tuple

from math_utilities.config import (
    DISTRIBUTION_NORMALIZATION_TOLERANCE,
    TAU_MAX_IN_UNITS,
    TAU_UNIT_IN_DAYS,
    UNITS_IN_ONE_DAY,
)
from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
    generate_discrete_distribution_from_cdf_function,
    generate_discrete_distribution_from_pdf_function,
)
from math_utilities.distributions_collection import lognormal_cdf, weibull_pdf

# Default effective reproduction number
R0 = 1
k, lambda_ = (
    2.855,
    5.611,
)  # These parameters give a Weibull distribution with mean 5.00 and variance 3.61


def rho0(tau: float) -> float:
    """Default generation time distribution."""
    return weibull_pdf(tau, k, lambda_)


rho0_discrete = generate_discrete_distribution_from_pdf_function(
    pdf=lambda tau: rho0(tau * TAU_UNIT_IN_DAYS) * TAU_UNIT_IN_DAYS,
    tau_min=1,
    tau_max=TAU_MAX_IN_UNITS,
    normalize=True,
)

# Default (discretized) infectiousness

b0 = rho0_discrete.rescale_by_factor(R0)


# Incubation period distribution

incubation_mu = 1.644
incubation_sigma = 0.363


tauS = generate_discrete_distribution_from_cdf_function(
    cdf=lambda tau: lognormal_cdf(tau / UNITS_IN_ONE_DAY, incubation_mu, incubation_sigma),
    tau_min=1,
    tau_max=TAU_MAX_IN_UNITS,
).normalize()


# Data for the "two-components model" (asymptomatic and symptomatic individuals)

p_sym = 0.6  # Fraction of infected individuals who are symptomatic.
contribution_of_symptomatics_to_R0 = 0.95  # Contributions of symptomatic individuals to R0.


def make_scenario_parameters_for_asymptomatic_symptomatic_model(
    rho0_discrete: DiscreteDistributionOnNonNegatives = rho0_discrete,
    R0: float = R0,
    p_sym: float = p_sym,
    contribution_of_symptomatics_to_R0: float = contribution_of_symptomatics_to_R0,
) -> Tuple[Tuple[float, ...], Tuple[DiscreteDistributionOnNonNegatives, ...]]:
    """
    Returns the couples p_gs and b0_gs for the "two-components model" for the severity,
    namely for asymptomatic and symptomatic individuals.
    :param rho0: the generation time distribution.
    :param p_sym: the fraction of infected individuals that are symptomatic
    :param contribution_of_symptomatics_to_R0: the fraction of R0 due to symptomatic individuals.
    """
    p_asy = 1 - p_sym  # Fraction of infected individuals who are asymptomatic.

    R0_asy = (  # Component of R0 due to asymptomatic individuals
        (1 - contribution_of_symptomatics_to_R0) / p_asy * R0 if p_asy > 0 else 0
    )
    R0_sym = (  # Component of R0 due to symptomatic individuals
        contribution_of_symptomatics_to_R0 / p_sym * R0 if p_sym > 0 else 0
    )

    assert abs(R0 - p_sym * R0_sym - p_asy * R0_asy) < DISTRIBUTION_NORMALIZATION_TOLERANCE

    p_gs = (1 - p_sym, p_sym)

    b0_gs = (
        rho0_discrete.rescale_by_factor(R0_asy),
        rho0_discrete.rescale_by_factor(R0_sym),
    )

    return p_gs, b0_gs

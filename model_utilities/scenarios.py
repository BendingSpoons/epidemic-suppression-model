"""
The classes defining the input data of the algorithms.
See the paper for details on the parameters.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
    FunctionOfTimeUnit,
)


class ScenarioError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("""The scenario is not well-defined.""", *args)


@dataclass
class Scenario(ABC):

    # Epidemic data

    # Probabilities of having given severity:
    p_gs: Tuple[float, ...]
    # Discretized infectiousness distributions b^0_{t,g}:
    b0_gs: Tuple[DiscreteDistributionOnNonNegatives, ...]
    # Discretized distribution of symptoms onset time:
    tauS: DiscreteDistributionOnNonNegatives

    def __post_init__(self):
        assert sum(self.p_gs) == 1, ScenarioError(
            "The fractions of people infected with given severity must sum to 1."
        )
        self.check_severities()

    def check_severities(self) -> None:
        """
        Checks that all the lists referring to each severity component of
        the infected population have the same length.
        """
        tuple_length_error = ScenarioError(
            "The tuples depending on severities must have the same length."
        )
        n_severities = self.n_severities
        assert len(self.b0_gs) == n_severities, tuple_length_error
        for t in self._tuples_to_check():
            assert len(t) == n_severities, tuple_length_error

    @abstractmethod
    def _tuples_to_check(self) -> Tuple[Tuple, ...]:
        raise NotImplementedError

    @property
    def n_severities(self) -> int:
        """
        Returns the number of segments the infected population is divided into,
        each depending on the value of the severity G.
        """
        return len(self.p_gs)


@dataclass
class HomogeneousScenario(Scenario):
    """
    Wraps a simulation scenario in the "homogeneous" case, by defining the input of the model as
    a set of parameters that describe the epidemic and the isolation measures.
    """

    # Model parameters

    # Absolute time at which isolation policies begin:
    t_0: float
    # Probabilities of (immediate) notification after symptoms, given severity:
    ss: Tuple[FunctionOfTimeUnit, ...]
    # Probability of immediate notification after the source tests positive:
    sc: FunctionOfTimeUnit
    # Average reduction of the number of infected people after testing positive:
    xi: FunctionOfTimeUnit
    # Distribution of the time elapsed between notification and testing positive:
    DeltaAT: DiscreteDistributionOnNonNegatives

    def _tuples_to_check(self) -> Tuple[Tuple, ...]:
        return (self.ss,)


@dataclass
class ScenarioWithApp(Scenario):
    """
    Wraps a simulation scenario in the scenario with app usage, by defining the input of
    the model as a set of parameters that describe the epidemic and the isolation measures.
    """

    # Model parameters

    # Absolute time at which isolation policies begin:
    t_0: float
    # Probabilities of (immediate) notification after symptoms for people with app, given the
    # degree of severity:
    ssapp: Tuple[FunctionOfTimeUnit, ...]
    # Probabilities of (immediate) notification after symptoms for people without app, given the
    # degree of severity:
    ssnoapp: Tuple[FunctionOfTimeUnit, ...]
    # Probability of immediate notification after the source tests positive,
    # given that source and the recipient have the app:
    scapp: FunctionOfTimeUnit
    # Probability of immediate CTA after the source tests positive,
    # given that one between the source and the recipient does not have the app:
    scnoapp: FunctionOfTimeUnit
    # Average reduction of the number of infected people after testing positive:
    xi: FunctionOfTimeUnit
    # Fraction of people with the app, by absolute time:
    epsilon_app: FunctionOfTimeUnit
    # Distribution of the time elapsed between notification and testing positive for people with
    # the app:
    DeltaATapp: DiscreteDistributionOnNonNegatives
    # Distribution of the time elapsed between notification and testing positive for people without
    # the app:
    DeltaATnoapp: DiscreteDistributionOnNonNegatives

    def _tuples_to_check(self) -> Tuple[Tuple, ...]:
        return self.ssapp, self.ssnoapp

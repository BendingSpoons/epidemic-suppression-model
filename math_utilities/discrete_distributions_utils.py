"""
A class and some utilities to handle discrete probability measures on the real axis.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from typing import Callable, Iterator, Optional, Sequence, Union

from math_utilities.config import DISTRIBUTION_NORMALIZATION_TOLERANCE, UNITS_IN_ONE_DAY
from math_utilities.general_utilities import float_sequences_match, normalize_sequence

FunctionOfTimeUnit = Callable[[int], float]


class DiscreteDistribution:
    """
    A class describing a (possibly improper–that is, not normalized to 1) probability measure
    over the real axis, which is always thought as representing the infectious age in the usages.
    The probability measure is encoded via its PMF and/or its CDF, stored as sequences of numbers
    plus the lower bound tau_min of the support. The time separation Δ𝜏 between two
    consecutive points is given by the constant
        math_utilities.config.UNITS_IN_ONE_DAY.
    This class computes the mean and total mass of the distribution, integrals over it, and
    convolutions with other distributions.
    """

    # Maximum deviation from 1 in initialization of proper distributions:
    NORMALIZATION_TOLERANCE = DISTRIBUTION_NORMALIZATION_TOLERANCE
    _COMPUTE_PMF_AND_CDF_ON_INIT = False

    def __init__(
        self,
        *,
        tau_min: int = 0,
        pmf_values: Optional[Sequence[float]] = None,
        cdf_values: Optional[Sequence[float]] = None,
        improper: bool = False,
    ):
        """
        Initializes the object by describing the probability measure either using its PMF or
        its CDF, provided as sequences of numbers.
        :param tau_min: the lower bound of the support of the measure.
        :param pmf_values: the optional PMF values (to be given when the CDF values are None),
          i.e. the sequence of probabilities at the times tau_min, tau_min + Δ𝜏, etc.
        :param cdf_values: the optional CDF values (to be given when the PMF values are None).
        :param improper: must be True if the input values are not (approximately) normalized to 1.
        """
        if pmf_values is not None and cdf_values is None:
            # Erase trailing zeros:
            l = len(pmf_values)
            for v in reversed(pmf_values):
                if v != 0:
                    break
                l -= 1
            tau_max = tau_min + l - 1
            self._init_with_pmf(pmf_values=pmf_values[:l], improper=improper)
        elif pmf_values is None and cdf_values is not None:
            # Erase trailing constant values:
            l = len(cdf_values) + 1
            for v in reversed(cdf_values):
                if v != cdf_values[-1]:
                    break
                l -= 1
            tau_max = tau_min + l - 1
            self._init_with_cdf(cdf_values=cdf_values[:l], improper=improper)
        elif pmf_values is None and cdf_values is None:
            raise ValueError
        else:
            raise ValueError

        self.tau_min = tau_min
        self.tau_max = tau_max
        self._improper = improper

        if self._COMPUTE_PMF_AND_CDF_ON_INIT:
            self._compute_pmf_values()
            self._compute_cdf_values()

    def _init_with_pmf(self, pmf_values: Sequence[float], improper: bool = False):
        if not improper:
            total_mass_discrepancy = sum(pmf_values) - 1
            assert abs(total_mass_discrepancy) < self.NORMALIZATION_TOLERANCE
            pmf_values = normalize_sequence(seq=pmf_values)
        self._pmf_support_values = pmf_values

        self._cdf_support_values = None  # Will be computed only if asked

    def _init_with_cdf(self, cdf_values: Sequence[float], improper: bool = False):
        if not improper:
            assert abs(cdf_values[-1] - 1) < self.NORMALIZATION_TOLERANCE
            cdf_values = [x / cdf_values[-1] for x in cdf_values]
        self._cdf_support_values = tuple(cdf_values)
        self._pmf_support_values = None  # Will be computed only if asked

    @property
    def support(self) -> Iterator:
        return range(self.tau_min, self.tau_max + 1)

    def pmf(self, tau: int) -> float:
        self._compute_pmf_values()
        if self.tau_min <= tau <= self.tau_max:
            return self._pmf_support_values[tau - self.tau_min]
        return 0.0

    def cdf(self, tau: int) -> float:
        self._compute_cdf_values()
        if tau < self.tau_min:
            return 0.0
        elif tau <= self.tau_max:
            return self._cdf_support_values[tau - self.tau_min]
        else:
            return self._cdf_support_values[-1] if self._cdf_support_values else 0

    @property
    def total_mass(self) -> float:
        if not self._improper:
            return 1
        if self._cdf_support_values is not None:
            return self._cdf_support_values[-1] if self._cdf_support_values else 0
        if self._pmf_support_values is not None:
            return sum(self._pmf_support_values)

    def mean(self) -> float:
        if self._cdf_support_values is not None and self.tau_min >= 0:
            return self._mean_via_cdf()
        self._compute_pmf_values()
        return self._mean_via_pmf()

    def _mean_via_pmf(self) -> float:
        return sum(
            tau * self._pmf_support_values[tau - self.tau_min] for tau in self.support
        )

    def _mean_via_cdf(self) -> float:
        assert self.tau_min >= 0
        return self.tau_min * self.total_mass + sum(
            self.total_mass - self._cdf_support_values[tau - self.tau_min]
            for tau in self.support
        )

    def _compute_pmf_values(self) -> None:
        if self._pmf_support_values is not None:
            return
        assert self._cdf_support_values is not None
        pmf_values = []
        previous_value = 0
        for v in self._cdf_support_values:
            pmf_values.append(v - previous_value)
            previous_value = v
        self._pmf_support_values = pmf_values

    def _compute_cdf_values(self) -> None:
        """
        Computes self._cdf_support_values, if this has not been done yet.
        """
        if self._cdf_support_values is not None:
            return
        assert self._pmf_support_values is not None
        cdf_values = []
        current_value = 0
        for v in self._pmf_support_values:
            current_value += v
            cdf_values.append(current_value)
        self._cdf_support_values = cdf_values

    def rescale_by_factor(self, scale_factor: float) -> "DiscreteDistribution":
        if self._pmf_support_values is not None:
            rescaled_pmf_values = [scale_factor * v for v in self._pmf_support_values]
            return self.__class__(
                pmf_values=rescaled_pmf_values, tau_min=self.tau_min, improper=True
            )
        else:
            rescaled_cdf_values = [scale_factor * v for v in self._cdf_support_values]
            return self.__class__(
                cdf_values=rescaled_cdf_values, tau_min=self.tau_min, improper=True
            )

    def rescale_by_function(
        self, scale_function: FunctionOfTimeUnit
    ) -> "DiscreteDistribution":
        self._compute_pmf_values()
        rescaled_pmf_values = [
            scale_function(self.tau_min + i) * v
            for i, v in enumerate(self._pmf_support_values)
        ]
        return self.__class__(
            pmf_values=rescaled_pmf_values, tau_min=self.tau_min, improper=True
        )

    def normalize(self) -> "DiscreteDistribution":
        self._compute_cdf_values()
        total_mass = self.total_mass
        rescaled_cdf_values = [v / total_mass for v in self._cdf_support_values]
        return self.__class__(
            cdf_values=rescaled_cdf_values, tau_min=self.tau_min, improper=False
        )

    def convolve(self, other: "DiscreteDistribution") -> "DiscreteDistribution":
        self._compute_pmf_values()
        other._compute_pmf_values()

        def convolution(tau):
            tau1_min = max(self.tau_min, tau - other.tau_max)
            tau1_max = min(self.tau_max, tau - other.tau_min)
            return sum(
                (
                    self._pmf_support_values[tau1 - self.tau_min]
                    * other._pmf_support_values[tau - tau1 - other.tau_min]
                )
                for tau1 in range(tau1_min, tau1_max + 1)
            )

        improper = self._improper or other._improper
        return self.__class__(
            pmf_values=[
                convolution(tau)
                for tau in range(
                    self.tau_min + other.tau_min, self.tau_max + other.tau_max + 1
                )
            ],
            tau_min=self.tau_min + other.tau_min,
            improper=improper,
        )

    def integrate(
        self,
        integrand: FunctionOfTimeUnit,
        tau_min: Optional[int] = None,
        tau_max: Optional[int] = None,
    ):
        tau_min = tau_min or self.tau_min
        tau_max = tau_max or self.tau_max
        return sum(
            integrand(tau) * self.pmf(tau) for tau in range(tau_min, tau_max + 1)
        )

    def __eq__(self, other: "DiscreteDistribution"):
        if self._pmf_support_values is not None:
            other._compute_pmf_values()
            values_match = float_sequences_match(
                self._pmf_support_values, other._pmf_support_values,
            )
        else:
            other._compute_cdf_values()
            values_match = float_sequences_match(
                self._cdf_support_values, other._cdf_support_values,
            )

        return (
            self.tau_min == other.tau_min
            and self.tau_max == other.tau_max
            and values_match
        )

    def __add__(self, other: Union[int, "DiscreteDistribution"]):
        if isinstance(other, int):
            if self._pmf_support_values is not None:
                return self.__class__(
                    pmf_values=self._pmf_support_values,
                    tau_min=self.tau_min + other,
                    improper=self._improper,
                )
            else:
                return self.__class__(
                    cdf_values=self._cdf_support_values,
                    tau_min=self.tau_min + other,
                    improper=self._improper,
                )
        elif isinstance(other, DiscreteDistribution):
            return self.convolve(other=other)
        else:
            raise ValueError(f"You cannot add an object of type {type(other)}.")


class DiscreteDistributionOnNonNegatives(DiscreteDistribution):
    """
    A special case of DiscreteDistribution, assumed to be supported inside the non-negative real
    axis.
    """

    def __init__(
        self,
        *,
        tau_min: int = 0,
        pmf_values: Optional[Sequence[float]] = None,
        cdf_values: Optional[Sequence[float]] = None,
        improper: bool = False,
        truncate_at_tau_max: Optional[int] = None,
    ):
        assert tau_min >= 0
        if truncate_at_tau_max is not None:
            max_input_length = truncate_at_tau_max - tau_min + 1
            if pmf_values is not None:
                pmf_values = pmf_values[:max_input_length]
            elif cdf_values is not None:
                cdf_values = cdf_values[:max_input_length]
        super().__init__(
            tau_min=tau_min,
            pmf_values=pmf_values,
            cdf_values=cdf_values,
            improper=improper,
        )
        if truncate_at_tau_max is not None:
            assert self.tau_max <= truncate_at_tau_max


def generate_discrete_distribution_from_pdf_function(
    pdf: Callable[[float], float], tau_min: int, tau_max: int, normalize: bool = False,
) -> DiscreteDistributionOnNonNegatives:
    """
    Initializes a DiscreteDistributionOnNonNegatives object by giving a PDF, which is sampled
    unit by unit within a given interval.
    """
    pmf_values = [pdf(tau) for tau in range(tau_min, tau_max + 1)]
    total_mass = sum(pmf_values)
    distribution = DiscreteDistributionOnNonNegatives(
        pmf_values=pmf_values, tau_min=tau_min, improper=total_mass != 1
    )
    if total_mass != 1 and normalize:
        distribution = distribution.normalize()

    return distribution


def generate_discrete_distribution_from_cdf_function(
    cdf: FunctionOfTimeUnit, tau_min: int, tau_max: int,
) -> DiscreteDistributionOnNonNegatives:
    """
    Initializes a DiscreteDistributionOnNonNegatives object by giving a CDF, which is sampled
    unit by unit within a given interval.
    """
    cdf_values = [cdf(tau) for tau in range(tau_min, tau_max + 1)]

    return DiscreteDistributionOnNonNegatives(
        cdf_values=cdf_values, tau_min=tau_min, improper=cdf_values[-1] != 1
    )


def delta_distribution(peak_tau_in_days: float) -> DiscreteDistributionOnNonNegatives:
    """Initializes a DiscreteDistributionOnNonNegatives object describing a probability measure
    concentrated in a point."""
    return DiscreteDistributionOnNonNegatives(
        pmf_values=[1], tau_min=int(round(peak_tau_in_days * UNITS_IN_ONE_DAY, 0))
    )


def linear_combination_discrete_distributions_by_values(
    scalars: Sequence[float],
    seq: Sequence[DiscreteDistributionOnNonNegatives],
    use_cdfs: bool = False,
    improper: bool = True,
) -> DiscreteDistributionOnNonNegatives:
    """
    Computes a linear combination (distribution-wise, i.e. taking linear combinations of the
    probabilities) of many DiscreteDistributionOnNonNegatives.
    """
    l = len(scalars)
    assert l == len(seq)
    tau_min = min(d.tau_min for d in seq)
    tau_max = max(d.tau_max for d in seq)
    if use_cdfs:
        total_cdf = []
        for tau in range(tau_min, tau_max + 1):
            total_cdf_tau = sum(scalars[i] * seq[i].cdf(tau) for i in range(l))
            total_cdf.append(total_cdf_tau)
        return seq[0].__class__(
            cdf_values=total_cdf, tau_min=tau_min, improper=improper
        )
    total_pmf = []
    for tau in range(tau_min, tau_max + 1):
        total_pmf_tau = sum(scalars[i] * seq[i].pmf(tau) for i in range(l))
        total_pmf.append(total_pmf_tau)
    return seq[0].__class__(pmf_values=total_pmf, tau_min=tau_min, improper=improper)

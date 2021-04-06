from math_utilities.discrete_distributions_utils import (
    DiscreteDistribution,
    DiscreteDistributionOnNonNegatives,
    linear_combination_discrete_distributions_by_values,
)


class TestDiscreteDistribution:
    def test_basics(self):
        d = DiscreteDistribution(
            pmf_values=[0.2, 0.1, 0.4, 0.7, 0, 0], tau_min=3, improper=True
        )
        assert d.tau_min == 3 and d.tau_max == 6
        assert d.support == range(3, 7)
        assert round(d.total_mass, 8) == 1.4
        assert d.pmf(2) == 0 and d.pmf(3) == 0.2 and d.pmf(5) == 0.4 and d.pmf(7) == 0
        assert (
            round(d.cdf(2), 8) == 0
            and round(d.cdf(3), 8) == 0.2
            and round(d.cdf(5), 8) == 0.7
            and round(d.cdf(7), 8) == 1.4
        )

        d1 = DiscreteDistribution(
            cdf_values=[0.2, 0.3, 0.7, 1.4, 1.4, 1.4], tau_min=3, improper=True
        )
        assert d1.tau_min == 3 and d1.tau_max == 6
        assert d1.support == range(3, 7)
        assert round(d1.total_mass, 8) == 1.4
        assert d == d1

    def test_normalization_check(self):
        try:
            DiscreteDistribution(pmf_values=[0.25, 0.4, 0.25], tau_min=-1)
            raise ValueError
        except AssertionError:
            DiscreteDistribution(pmf_values=[0.25, 0.5, 0.25], tau_min=-1)
            DiscreteDistribution(
                pmf_values=[0.25, 0.4, 0.25], tau_min=-1, improper=True
            )

    def test_convolution(self):
        d1 = DiscreteDistribution(pmf_values=[0.25, 0.5, 0.25], tau_min=-1)
        d2 = DiscreteDistribution(pmf_values=[0.25, 0.5, 0.25], tau_min=1)

        d3 = d1 + d2

        assert d3.tau_min == 0
        assert d3.tau_max == 4
        assert d3._pmf_support_values == [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16]

        d4 = DiscreteDistribution(pmf_values=[1], tau_min=2)

        assert d2 == d1 + d4

    def test_rescalings(self):
        d1 = DiscreteDistribution(pmf_values=[0.25, 0.5, 0.25], tau_min=-1)
        d2 = DiscreteDistribution(pmf_values=[0.5, 1, 0.5], tau_min=-1, improper=True)
        d3 = DiscreteDistribution(
            pmf_values=[-0.25, 0.0, 0.25], tau_min=-1, improper=True
        )

        assert d1.rescale_by_factor(scale_factor=2) == d2
        assert d2.normalize() == d1
        assert d1.rescale_by_function(scale_function=lambda tau: tau) == d3

    def test_mean(self):
        d1 = DiscreteDistribution(pmf_values=[0.25, 0.5, 0.25], tau_min=-1)
        d1._compute_cdf_values()
        assert round(d1.mean(), 8) == 0
        d2 = DiscreteDistribution(pmf_values=[0.25, 0.5, 0.25], tau_min=5)
        d2._compute_cdf_values()
        assert round(d2._mean_via_pmf(), 8) == round(d2._mean_via_cdf(), 8) == 6

        d3 = DiscreteDistribution(pmf_values=[0.25, 0.25, 0.5], tau_min=3)
        d3._compute_cdf_values()
        assert round(d3._mean_via_pmf(), 8) == round(d3._mean_via_cdf(), 8) == 4.25

    def test_integration(self):
        d2 = DiscreteDistribution(pmf_values=[0.25, 0.5, 0.25], tau_min=5)
        assert d2.integrate(integrand=lambda tau: 1, tau_min=0, tau_max=10) == 1
        assert (
            d2.integrate(integrand=lambda tau: tau, tau_min=0, tau_max=10) == d2.mean()
        )
        assert (
            d2.integrate(integrand=lambda tau: -tau, tau_min=0, tau_max=10)
            == -d2.mean()
        )

    def test_linear_combination(self):
        d1 = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.1, 0.3, 0.4, 0.2, 0.5, 0.2, 0.3], tau_min=0, improper=True
        )
        d2 = DiscreteDistributionOnNonNegatives(pmf_values=[0.2, 0.5, 0.3], tau_min=3)

        d3 = linear_combination_discrete_distributions_by_values(
            scalars=[2, 3], seq=[d1, d2], use_cdfs=False, improper=True,
        )
        assert d3 == DiscreteDistributionOnNonNegatives(
            pmf_values=[0.2, 0.6, 0.8, 1, 2.5, 1.3, 0.6], improper=True
        )
        assert d3 == linear_combination_discrete_distributions_by_values(
            scalars=[2, 3], seq=[d1, d2], use_cdfs=True, improper=True,
        )

        linear_combination_discrete_distributions_by_values(
            scalars=[0.4, 0.6],
            seq=[d1.normalize(), d2],
            use_cdfs=False,
            improper=False,
        )

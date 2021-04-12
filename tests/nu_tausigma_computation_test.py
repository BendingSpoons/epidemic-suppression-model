from epidemic_suppression_algorithms.model_blocks.nu_and_tausigma import (
    check_b_negative_times,
    compute_tausigma_and_nu_at_time_t,
    compute_tausigma_and_nu_components_at_time_t,
    compute_tausigma_and_nu_components_at_time_t_with_app,
)
from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
)
from math_utilities.general_utilities import float_sequences_match, floats_match


class TestNuTausigmaComputation:
    def test_tausigma_nu_computation_one_severity_no_negative_times(self):
        b = [
            DiscreteDistributionOnNonNegatives(
                pmf_values=[1.5, 2, 1], tau_min=1, improper=True
            ),
            DiscreteDistributionOnNonNegatives(
                pmf_values=[1.5, 2.5, 1], tau_min=1, improper=True
            ),
            DiscreteDistributionOnNonNegatives(
                pmf_values=[1, 1.4, 0.9], tau_min=1, improper=True
            ),
        ]

        nu_1, tausigma_1 = compute_tausigma_and_nu_at_time_t(t=1, b=b, nu=[10],)
        assert nu_1 == 15
        assert tausigma_1 == DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=1
        )

        nu_2, tausigma_2 = compute_tausigma_and_nu_at_time_t(t=2, b=b, nu=[10, 15],)
        assert nu_2 == 10 * 2 + 15 * 1.5 == 42.5
        assert (
            tausigma_2
            == DiscreteDistributionOnNonNegatives(
                pmf_values=[15 * 1.5, 10 * 2], tau_min=1, improper=True
            ).normalize()
        )

        nu_3, tausigma_3 = compute_tausigma_and_nu_at_time_t(
            t=3, b=b, nu=[10, 15, 42.5],
        )
        assert nu_3 == 10 * 1 + 15 * 2.5 + 42.5 * 1 == 90
        assert (
            tausigma_3
            == DiscreteDistributionOnNonNegatives(
                pmf_values=[42.5 * 1, 15 * 2.5, 10 * 1], tau_min=1, improper=True
            ).normalize()
        )

    def test_check_negative_times(self):
        p_gs = (0.4, 0.6)
        rho_0 = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.3, 0.4, 0.3], tau_min=1, improper=True
        )

        # Case 1

        c_0 = 0.5
        c_1 = (-c_0 * p_gs[0] + 1) / p_gs[1]

        check_b_negative_times(
            p_gs=p_gs,
            b_negative_times=(
                rho_0.rescale_by_factor(scale_factor=c_0),
                rho_0.rescale_by_factor(scale_factor=c_1),
            ),
        )

        # Case 2

        c_0 = 0.8
        c_1 = (-c_0 * p_gs[0] + 1) / p_gs[1]

        check_b_negative_times(
            p_gs=p_gs,
            b_negative_times=(
                rho_0.rescale_by_factor(scale_factor=c_0),
                rho_0.rescale_by_factor(scale_factor=c_1),
            ),
        )

        # Case 3

        c_0 = 0.8
        c_1 = 0.6

        try:
            check_b_negative_times(
                p_gs=p_gs,
                b_negative_times=(
                    rho_0.rescale_by_factor(scale_factor=c_0),
                    rho_0.rescale_by_factor(scale_factor=c_1),
                ),
            )
            raise AssertionError
        except AssertionError:
            pass

    def test_tausigma_nu_computation_one_severity_with_negative_times(self):
        b = [
            DiscreteDistributionOnNonNegatives(
                pmf_values=[1.5, 2, 1], tau_min=1, improper=True
            ),
            DiscreteDistributionOnNonNegatives(
                pmf_values=[1.5, 2.5, 1], tau_min=1, improper=True
            ),
            DiscreteDistributionOnNonNegatives(
                pmf_values=[1, 1.4, 0.9], tau_min=1, improper=True
            ),
        ]

        b_negative_times = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.3, 0.4, 0.3], tau_min=1, improper=True
        )

        nu_0 = 10

        nu_1, tausigma_1 = compute_tausigma_and_nu_at_time_t(
            t=1,
            b=b,
            nu=[nu_0],
            b_negative_times=b_negative_times,
            nu_negative_times=nu_0,
        )
        assert nu_1 == (0.3 + 0.4 + 1.5) * 10 == 22
        assert (
            tausigma_1
            == DiscreteDistributionOnNonNegatives(
                pmf_values=[15, 4, 3], tau_min=1, improper=True
            ).normalize()
        )

        nu_2, tausigma_2 = compute_tausigma_and_nu_at_time_t(
            t=2,
            b=b,
            nu=[nu_0, nu_1],
            b_negative_times=b_negative_times,
            nu_negative_times=nu_0,
        )
        assert nu_2 == 10 * 0.3 + 10 * 2 + 22 * 1.5 == 56
        assert (
            tausigma_2
            == DiscreteDistributionOnNonNegatives(
                pmf_values=[22 * 1.5, 10 * 2, 10 * 0.3], tau_min=1, improper=True
            ).normalize()
        )

        nu_3, tausigma_3 = compute_tausigma_and_nu_at_time_t(
            t=3,
            b=b,
            nu=[nu_0, nu_1, nu_2],
            b_negative_times=b_negative_times,
            nu_negative_times=nu_0,
        )
        assert nu_3 == 10 * 1 + 22 * 2.5 + 56 * 1 == 121
        assert (
            tausigma_3
            == DiscreteDistributionOnNonNegatives(
                pmf_values=[56 * 1, 22 * 2.5, 10 * 1], tau_min=1, improper=True
            ).normalize()
        )

    def test_tausigma_nu_computation_many_severities(self):
        p_gs = (0.4, 0.6)
        rho_0 = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.3, 0.4, 0.3], tau_min=1, improper=True
        )
        c_0 = 0.5
        c_1 = (-c_0 * p_gs[0] + 1) / p_gs[1]
        assert floats_match(c_1, 4 / 3)
        b_negative_times = (
            rho_0.rescale_by_factor(scale_factor=c_0),
            rho_0.rescale_by_factor(scale_factor=c_1),
        )
        check_b_negative_times(
            p_gs=p_gs, b_negative_times=b_negative_times,
        )

        R_ts = [2]
        b = [
            (
                rho_0.rescale_by_factor(scale_factor=c_0 * R_t),
                rho_0.rescale_by_factor(scale_factor=c_1 * R_t),
            )
            for R_t in R_ts
        ]
        nu_0 = 10

        nugs_1, tausigmags_1 = compute_tausigma_and_nu_components_at_time_t(
            t=1,
            b=b,
            nu=[nu_0],
            p_gs=p_gs,
            b_negative_times=b_negative_times,
            nu_negative_times=nu_0,
        )

        expected_nu0_1_addends = [
            nu_0 * p_gs[0] * c_0 * R_ts[0] * 0.3,  # Infections from infected at t=0
            nu_0 * p_gs[0] * c_0 * 0.4,  # Infections from infected at t=-1
            nu_0 * p_gs[0] * c_0 * 0.3,  # Infections from infected at t=-2
        ]

        expected_nu0_1 = sum(expected_nu0_1_addends)

        expected_nu1_1_addends = [
            nu_0 * p_gs[1] * c_1 * R_ts[0] * 0.3,  # Infections from infected at t=0
            nu_0 * p_gs[1] * c_1 * 0.4,  # Infections from infected at t=-1
            nu_0 * p_gs[1] * c_1 * 0.3,  # Infections from infected at t=-2
        ]
        expected_nu1_1 = sum(expected_nu1_1_addends)

        assert float_sequences_match(nugs_1, (expected_nu0_1, expected_nu1_1))
        assert tausigmags_1[0] == DiscreteDistributionOnNonNegatives(
            pmf_values=expected_nu0_1_addends, tau_min=1, improper=True
        ).rescale_by_factor(scale_factor=1 / (nugs_1[0] + nugs_1[1]))
        assert tausigmags_1[1] == DiscreteDistributionOnNonNegatives(
            pmf_values=expected_nu1_1_addends, tau_min=1, improper=True
        ).rescale_by_factor(scale_factor=1 / (nugs_1[0] + nugs_1[1]))

    def test_tausigma_nu_computation_with_app(self):

        p_gs = (0.4, 0.6)
        epsilon_app = lambda t: 0.3
        rho_0 = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.3, 0.4, 0.3], tau_min=1, improper=True
        )
        c_0 = 0.5
        c_1 = (-c_0 * p_gs[0] + 1) / p_gs[1]
        assert floats_match(c_1, 4 / 3)
        b_negative_times = (
            rho_0.rescale_by_factor(scale_factor=c_0),
            rho_0.rescale_by_factor(scale_factor=c_1),
        )
        check_b_negative_times(
            p_gs=p_gs, b_negative_times=b_negative_times,
        )

        R_0_app = 2
        R_0_noapp = 3
        b_0_app = (
            rho_0.rescale_by_factor(scale_factor=c_0 * R_0_app),
            rho_0.rescale_by_factor(scale_factor=c_1 * R_0_app),
        )

        b_0_noapp = (
            rho_0.rescale_by_factor(scale_factor=c_0 * R_0_noapp),
            rho_0.rescale_by_factor(scale_factor=c_1 * R_0_noapp),
        )
        nu_0 = 10

        (
            nugsapp_1,
            tausigmagsapp_1,
            nugsnoapp_1,
            tausigmagsnoapp_1,
        ) = compute_tausigma_and_nu_components_at_time_t_with_app(
            t=1,
            b_app=[b_0_app],
            b_noapp=[b_0_noapp],
            nu=[nu_0],
            p_gs=p_gs,
            epsilon_app=epsilon_app,
            b_negative_times=b_negative_times,
            nu_negative_times=nu_0,
        )

        expected_nu0app_1_addends = [
            nu_0
            * p_gs[0]
            * epsilon_app(0)
            * c_0
            * R_0_app
            * 0.3,  # Infections from infected at t=0
            0,  # Infections from infected at t=-1
            0,  # Infections from infected at t=-2
        ]

        expected_nu0noapp_1_addends = [
            nu_0
            * p_gs[0]
            * (1 - epsilon_app(0))
            * c_0
            * R_0_noapp
            * 0.3,  # Infections from infected at t=0
            nu_0 * p_gs[0] * c_0 * 0.4,  # Infections from infected at t=-1
            nu_0 * p_gs[0] * c_0 * 0.3,  # Infections from infected at t=-2
        ]

        expected_nu1app_1_addends = [
            nu_0
            * p_gs[1]
            * epsilon_app(0)
            * c_1
            * R_0_app
            * 0.3,  # Infections from infected at t=0
            0,  # Infections from infected at t=-1
            0,  # Infections from infected at t=-2
        ]

        expected_nu1noapp_1_addends = [
            nu_0
            * p_gs[1]
            * (1 - epsilon_app(0))
            * c_1
            * R_0_noapp
            * 0.3,  # Infections from infected at t=0
            nu_0 * p_gs[1] * c_1 * 0.4,  # Infections from infected at t=-1
            nu_0 * p_gs[1] * c_1 * 0.3,  # Infections from infected at t=-2
        ]

        expected_nu0app_1 = sum(expected_nu0app_1_addends)
        expected_nu0noapp_1 = sum(expected_nu0noapp_1_addends)
        expected_nu1app_1 = sum(expected_nu1app_1_addends)
        expected_nu1noapp_1 = sum(expected_nu1noapp_1_addends)

        expected_nu_1 = (
            expected_nu0app_1
            + expected_nu0noapp_1
            + expected_nu1app_1
            + expected_nu1noapp_1
        )

        assert float_sequences_match(nugsapp_1, (expected_nu0app_1, expected_nu1app_1))
        assert float_sequences_match(
            nugsnoapp_1, (expected_nu0noapp_1, expected_nu1noapp_1)
        )
        assert tausigmagsapp_1[0] == DiscreteDistributionOnNonNegatives(
            pmf_values=expected_nu0app_1_addends, tau_min=1, improper=True
        ).rescale_by_factor(scale_factor=1 / expected_nu_1)
        assert tausigmagsnoapp_1[0] == DiscreteDistributionOnNonNegatives(
            pmf_values=expected_nu0noapp_1_addends, tau_min=1, improper=True
        ).rescale_by_factor(scale_factor=1 / expected_nu_1)
        assert tausigmagsapp_1[1] == DiscreteDistributionOnNonNegatives(
            pmf_values=expected_nu1app_1_addends, tau_min=1, improper=True
        ).rescale_by_factor(scale_factor=1 / expected_nu_1)
        assert tausigmagsnoapp_1[1] == DiscreteDistributionOnNonNegatives(
            pmf_values=expected_nu1noapp_1_addends, tau_min=1, improper=True
        ).rescale_by_factor(scale_factor=1 / expected_nu_1)

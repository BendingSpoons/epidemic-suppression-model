from epidemic_suppression_algorithms.model_blocks.testing_time_and_b_t_suppression import (
    compute_suppressed_b_t,
    compute_tauT_t,
)
from epidemic_suppression_algorithms.model_blocks.time_evolution_block import (
    compute_tauAc_t,
    compute_tauAc_t_two_components,
)
from math_utilities.config import FLOAT_TOLERANCE_FOR_EQUALITIES, UNITS_IN_ONE_DAY
from math_utilities.discrete_distributions_utils import (
    DiscreteDistributionOnNonNegatives,
    linear_combination_discrete_distributions_by_values,
)
from math_utilities.general_utilities import floats_match
from model_utilities.epidemic_data import (
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
)


class TestAlgorithmBlock:
    """Tests each block of the algorithm separately"""

    def test_b_suppression(self):
        _, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()
        R0_gs = tuple(b0_gs[g].total_mass for g in [0, 1])
        xi = 0.8  # Any value in [0,1] will do

        tauT_peak = 10 * UNITS_IN_ONE_DAY

        tauT_gs = (
            DiscreteDistributionOnNonNegatives(
                pmf_values=[0], tau_min=0, improper=True
            ),
            DiscreteDistributionOnNonNegatives(pmf_values=[1], tau_min=tauT_peak),
        )

        b_t_gs = compute_suppressed_b_t(b0_t_gs=b0_gs, tauT_t_gs=tauT_gs, xi_t=xi)

        R_t_gs = tuple(b_t_gs[g].total_mass for g in [0, 1])

        assert b_t_gs[0] == b0_gs[0]  # No suppression for asymptomatics
        assert R_t_gs[0] == R0_gs[0]

        assert all(
            b_t_gs[1].pmf(tau * UNITS_IN_ONE_DAY)
            == b0_gs[1].pmf(tau * UNITS_IN_ONE_DAY)
            for tau in (0, 3, 6, 9)
        )
        assert all(
            b_t_gs[1].pmf(tau * UNITS_IN_ONE_DAY)
            == (1 - xi) * b0_gs[1].pmf(tau * UNITS_IN_ONE_DAY)
            for tau in (10, 13, 16, 19)
        )
        assert (1 - xi) * R0_gs[1] <= R_t_gs[1] <= R0_gs[1]

    def test_compute_tauT(self):
        # No contact tracing without app
        tauAs_peak = 5 * UNITS_IN_ONE_DAY
        tauAs_t_gs = (
            DiscreteDistributionOnNonNegatives(
                pmf_values=[0], tau_min=0, improper=True
            ),
            DiscreteDistributionOnNonNegatives(pmf_values=[1], tau_min=tauAs_peak),
        )

        # Uniform distribution from 0 to 6 (excluded),  with total mass 0.6
        tau_Ac_t = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.1 / UNITS_IN_ONE_DAY] * 6 * UNITS_IN_ONE_DAY,
            tau_min=0,
            improper=True,
        )
        assert floats_match(tau_Ac_t.total_mass, 0.6)

        DeltaAT_peak = 2 * UNITS_IN_ONE_DAY
        DeltaAT = DiscreteDistributionOnNonNegatives(
            pmf_values=[1], tau_min=DeltaAT_peak
        )

        tauT_t_gs = compute_tauT_t(
            tauAs_t_gs=tauAs_t_gs, tauAc_t=tau_Ac_t, DeltaAT=DeltaAT
        )

        # Expected results
        tau_A_gs = (
            tau_Ac_t,
            DiscreteDistributionOnNonNegatives(
                pmf_values=[0.1 / UNITS_IN_ONE_DAY] * 5 * UNITS_IN_ONE_DAY + [0.5],
                tau_min=0,
                improper=True,
            ),
        )

        assert all(tauT_t_gs[g] == tau_A_gs[g] + DeltaAT_peak for g in (0, 1))

    def test_compute_tauAc(self):
        """
        Tests the time evolution equation in the homogeneous scenario.
        """

        # 1 - No suppression, one component, one contribution
        sc = 0.5
        tauAc_t = compute_tauAc_t(
            t=1,
            tauT=[
                (
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.1, 0.2, 0.3], tau_min=2, improper=True
                    ),
                )
            ],
            tausigmags_t=(
                DiscreteDistributionOnNonNegatives(pmf_values=[1], tau_min=1),
            ),
            xi=lambda t: 0,
            sc_t=sc,
        )
        expected_tauAc_t = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.1, 0.2, 0.3], tau_min=1, improper=True
        ).rescale_by_factor(sc)
        assert tauAc_t == expected_tauAc_t

        # 2 - No suppression, one component, one cut contribution
        sc = 0.5
        tauAc_t = compute_tauAc_t(
            t=1,
            tauT=[
                (
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.1, 0.2, 0.3], tau_min=1, improper=True
                    ),
                )
            ],
            tausigmags_t=(
                DiscreteDistributionOnNonNegatives(pmf_values=[1], tau_min=1),
            ),
            xi=lambda t: 0,
            sc_t=sc,
        )
        expected_tauAc_t = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.2, 0.3], tau_min=1, improper=True
        ).rescale_by_factor(sc)
        assert tauAc_t == expected_tauAc_t

        # 3 - No suppression, one component, two contributions
        sc = 0.5
        tauAc_t = compute_tauAc_t(
            t=3,
            tauT=[
                (
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.1, 0.2, 0.3], tau_min=2, improper=True
                    ),
                )
            ]
            * 3,
            tausigmags_t=(
                DiscreteDistributionOnNonNegatives(pmf_values=[0.4, 0.6, 0], tau_min=1),
            ),
            xi=lambda t: 0,
            sc_t=sc,
        )
        expected_tauAc_t = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.1 * 0.4 + 0.2 * 0.6, 0.2 * 0.4 + 0.3 * 0.6, 0.3 * 0.4],
            tau_min=1,
            improper=True,
        ).rescale_by_factor(sc)
        assert tauAc_t == expected_tauAc_t

        # 4 - With suppression, one component, two contributions
        sc = 0.5
        xi = 0.6
        tauAc_t = compute_tauAc_t(
            t=3,
            tauT=[
                (
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.1, 0.2, 0.3], tau_min=2, improper=True
                    ),
                )
            ]
            * 3,
            tausigmags_t=(
                DiscreteDistributionOnNonNegatives(pmf_values=[0.4, 0.6, 0], tau_min=1),
            ),
            xi=lambda t: xi,
            sc_t=sc,
        )
        assert all(tauAc_t.pmf(tau) >= expected_tauAc_t.pmf(tau) for tau in [1, 2, 3])
        assert tauAc_t.total_mass > expected_tauAc_t.total_mass

        # 5 - No suppression, two components, two contributions
        xi = 0
        sc = 0.6
        tauAc_t = compute_tauAc_t(
            t=4,
            tauT=[
                (
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.1, 0.2, 0], tau_min=2, improper=True
                    ),
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.1, 0.2, 0.3], tau_min=2, improper=True
                    ),
                )
            ]
            * 4,
            tausigmags_t=(
                DiscreteDistributionOnNonNegatives(
                    pmf_values=[0.1, 0.2, 0, 0], tau_min=1, improper=True,
                ),
                DiscreteDistributionOnNonNegatives(
                    pmf_values=[0.3, 0.4, 0, 0], tau_min=1, improper=True,
                ),
            ),
            xi=lambda t: xi,
            sc_t=sc,
        )
        expectedchecktauT0_4 = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.1 * 0.1 + 0.2 * 0.2, 0.2 * 0.1], tau_min=1, improper=True,
        )
        expectedchecktauT1_4 = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.1 * 0.3 + 0.2 * 0.4, 0.2 * 0.3 + 0.3 * 0.4, 0.3 * 0.3],
            tau_min=1,
            improper=True,
        )

        expected_tauAc_t = linear_combination_discrete_distributions_by_values(
            scalars=[sc, sc], seq=[expectedchecktauT0_4, expectedchecktauT1_4],
        )
        assert tauAc_t == expected_tauAc_t

    def test_compute_tauAc_with_app(self):
        """
        Tests the time evolution equation in the scenario with app.
        """

        xi = 0
        scapp = 0.6
        scnoapp = 0.2

        kwargs = dict(
            t=4,
            tauT_app=[
                (
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.1, 0.13, 0], tau_min=2, improper=True
                    ),
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.11, 0.19], tau_min=2, improper=True
                    ),
                )
            ]
            * 3
            + [
                (
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.1, 0.16, 0], tau_min=2, improper=True
                    ),
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.125, 0.19], tau_min=2, improper=True
                    ),
                )
            ],
            tauT_noapp=[
                (
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.08, 0.14, 0], tau_min=2, improper=True
                    ),
                    DiscreteDistributionOnNonNegatives(
                        pmf_values=[0.12, 0.23, 0.18], tau_min=2, improper=True
                    ),
                )
            ]
            * 4,
            tausigmagsapp_t=(
                DiscreteDistributionOnNonNegatives(
                    pmf_values=[0.05, 0.1, 0, 0], tau_min=1, improper=True,
                ),
                DiscreteDistributionOnNonNegatives(
                    pmf_values=[0.1, 0.2, 0, 0], tau_min=1, improper=True,
                ),
            ),
            tausigmagsnoapp_t=(
                DiscreteDistributionOnNonNegatives(
                    pmf_values=[0.1, 0.15, 0, 0], tau_min=1, improper=True,
                ),
                DiscreteDistributionOnNonNegatives(
                    pmf_values=[0.15, 0.25, 0, 0], tau_min=1, improper=True,
                ),
            ),
            xi=lambda t: xi,
            scapp_t=scapp,
            scnoapp_t=scnoapp,
        )

        tauAc_t_app, tauAc_t_noapp = compute_tauAc_t_two_components(**kwargs)

        expectedchecktauT0app_4 = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.1 * 0.05 + 0.13 * 0.1, 0.16 * 0.05], tau_min=1, improper=True,
        )
        expectedchecktauT1app_4 = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.125 * 0.1 + 0.19 * 0.2, 0.19 * 0.1], tau_min=1, improper=True,
        )

        expectedchecktauT0noapp_4 = DiscreteDistributionOnNonNegatives(
            pmf_values=[0.08 * 0.1 + 0.14 * 0.15, 0.14 * 0.1], tau_min=1, improper=True,
        )
        expectedchecktauT1noapp_4 = DiscreteDistributionOnNonNegatives(
            pmf_values=[
                0.12 * 0.15 + 0.23 * 0.25,
                0.23 * 0.15 + 0.18 * 0.25,
                0.18 * 0.15,
            ],
            tau_min=1,
            improper=True,
        )

        expected_tauAc_t_app = linear_combination_discrete_distributions_by_values(
            scalars=[scapp, scapp, scnoapp, scnoapp],
            seq=[
                expectedchecktauT0app_4,
                expectedchecktauT1app_4,
                expectedchecktauT0noapp_4,
                expectedchecktauT1noapp_4,
            ],
        )

        expected_tauAc_t_noapp = linear_combination_discrete_distributions_by_values(
            scalars=[scnoapp, scnoapp, scnoapp, scnoapp],
            seq=[
                expectedchecktauT0app_4,
                expectedchecktauT1app_4,
                expectedchecktauT0noapp_4,
                expectedchecktauT1noapp_4,
            ],
        )
        assert tauAc_t_app == expected_tauAc_t_app
        assert tauAc_t_noapp == expected_tauAc_t_noapp

        # Same, but this time with suppression
        kwargs["xi"] = lambda t: 0.6

        tauAc_t_app, tauAc_t_noapp = compute_tauAc_t_two_components(**kwargs)

        assert all(
            tauAc_t_app.pmf(tau)
            >= expected_tauAc_t_app.pmf(tau) - FLOAT_TOLERANCE_FOR_EQUALITIES
            for tau in [1, 2, 3]
        )
        assert tauAc_t_app.total_mass > expected_tauAc_t_app.total_mass

        assert all(
            tauAc_t_noapp.pmf(tau)
            >= expected_tauAc_t_noapp.pmf(tau) - FLOAT_TOLERANCE_FOR_EQUALITIES
            for tau in [1, 2, 3]
        )
        assert tauAc_t_noapp.total_mass > expected_tauAc_t_noapp.total_mass

from epidemic_suppression_algorithms.evolution_with_app_algorithm import (
    compute_time_evolution_with_app,
)
from epidemic_suppression_algorithms.homogeneous_evolution_algorithm import (
    compute_time_evolution_homogeneous_case,
)
from math_utilities.discrete_distributions_utils import delta_distribution
from model_utilities.epidemic_data import (
    R0,
    make_scenario_parameters_for_asymptomatic_symptomatic_model,
    tauS,
)
from model_utilities.scenarios import HomogeneousScenario, ScenarioWithApp


def check_equality_with_precision(x: float, y: float, decimal: int):
    return round(x - y, ndigits=decimal) == 0


class TestCornerCases:
    def test_no_epidemic_control_scenario(self):
        p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

        scenario = HomogeneousScenario(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ss=(lambda t: 0, lambda t: 0),
            sc=lambda t: 0,
            xi=lambda t: 1,
            DeltaAT=delta_distribution(peak_tau_in_days=2),
        )

        (
            t_in_days_list,
            nu,
            nu0,
            R,
            R_by_severity,
            FT_infty,
        ) = compute_time_evolution_homogeneous_case(
            scenario=scenario,
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=tuple(d.normalize() for d in b0_gs),
            threshold_to_stop=0.001,
            verbose=False,
        )

        assert R[-1] == R0
        assert FT_infty[-1] == 0

    def test_only_tracing_scenario(self):
        p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

        scenario = HomogeneousScenario(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ss=(lambda t: 0, lambda t: 0),
            sc=lambda t: 1,
            xi=lambda t: 1,
            DeltaAT=delta_distribution(peak_tau_in_days=2),
        )

        (
            t_in_days_list,
            nu,
            nu0,
            R,
            R_by_severity,
            FT_infty,
        ) = compute_time_evolution_homogeneous_case(
            scenario=scenario,
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=tuple(d.normalize() for d in b0_gs),
            threshold_to_stop=0.001,
            verbose=False,
        )

        assert R[-1] == R0
        assert FT_infty[-1] == 0

    def test_only_symptoms_control(self):
        p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

        scenario = HomogeneousScenario(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ss=(lambda t: 0, lambda t: 0.5),
            sc=lambda t: 0,
            xi=lambda t: 1,
            DeltaAT=delta_distribution(peak_tau_in_days=2),
        )

        (
            t_in_days_list,
            nu,
            nu0,
            R,
            R_by_severity,
            FT_infty,
        ) = compute_time_evolution_homogeneous_case(
            scenario=scenario,
            t_max_in_days=20,
            nu_start=1000,
            b_negative_times=tuple(d.normalize() for d in b0_gs),
            threshold_to_stop=0.001,
            verbose=False,
        )

        assert R[0] == R[-1] < R0
        assert FT_infty[0] == FT_infty[-1] > 0

    def test_no_differences_app_no_app_scenario(self):
        p_gs, b0_gs = make_scenario_parameters_for_asymptomatic_symptomatic_model()

        ss = 0.5
        sc = 0.6
        DeltaAT = delta_distribution(peak_tau_in_days=2)
        xi = 0.9

        t_max_in_days = 20
        nu_start = 1000

        # Homogeneous case

        scenario = HomogeneousScenario(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ss=(lambda t: 0, lambda t: ss),
            sc=lambda t: sc,
            xi=lambda t: xi,
            DeltaAT=DeltaAT,
        )

        (
            t_in_days_list,
            nu,
            nu0,
            R_homog,
            R_by_severity,
            FT_infty_homog,
        ) = compute_time_evolution_homogeneous_case(
            scenario=scenario,
            t_max_in_days=t_max_in_days,
            nu_start=nu_start,
            b_negative_times=tuple(d.normalize() for d in b0_gs),
            threshold_to_stop=0.001,
            verbose=False,
        )

        # With app case

        scenario = ScenarioWithApp(
            p_gs=p_gs,
            b0_gs=b0_gs,
            tauS=tauS,
            t_0=0,
            ssapp=(lambda t: 0, lambda t: ss),
            ssnoapp=(lambda t: 0, lambda t: ss),
            scapp=lambda t: sc,
            scnoapp=lambda t: sc,
            xi=lambda t: xi,
            DeltaATapp=DeltaAT,
            DeltaATnoapp=DeltaAT,
            epsilon_app=lambda t: 0.6,
        )

        (
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
        ) = compute_time_evolution_with_app(
            scenario=scenario,
            t_max_in_days=t_max_in_days,
            nu_start=1000,
            b_negative_times=tuple(d.normalize() for d in b0_gs),
            verbose=False,
        )

        assert abs(R[-1] - R_homog[-1]) < 0.001
        assert abs(R_app[-1] - R_homog[-1]) < 0.001
        assert abs(R_noapp[-1] - R_homog[-1]) < 0.001
        assert abs(FT_infty[-1] - FT_infty_homog[-1]) < 0.001
        assert abs(FT_app_infty[-1] - FT_infty_homog[-1]) < 0.001
        assert abs(FT_noapp_infty[-1] - FT_infty_homog[-1]) < 0.001

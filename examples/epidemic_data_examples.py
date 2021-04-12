"""
Some examples printing and plotting the epidemic data used as inputs of the model.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

from examples.plotting_utils import plot_discrete_distributions
from model_utilities.epidemic_data import b0, rho0_discrete, tauS


def plot_generation_time():
    print(
        "Expected default generation time: E(τ^C) =", rho0_discrete.mean(),
    )
    plot_discrete_distributions(
        ds=[rho0_discrete], title="The default generation time distribution ρ^0",
    )


def plot_infectiousness():
    plot_discrete_distributions(
        ds=[b0], title="The default infectiousness β^0_0",
    )


def plot_symptoms_onset_distribution():

    EtauS = tauS.mean()
    print("E(τ^S) =", EtauS)

    plot_discrete_distributions(
        ds=[tauS], title="The CDF F^S of the symptoms onset time τ^S",
    )

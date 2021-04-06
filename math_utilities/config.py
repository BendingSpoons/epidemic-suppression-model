"""
This file contains some constants that determine the numerical precision of the calculations.

Authors: Andrea Maiorana, Marco Meneghelli
Copyright 2021 Bending Spoons S.p.A.
"""

# Number of time instants in a day used in the discretization of the model. Increasing this
# number improves the precision in handling distributions and the number of steps of the
# algorithms:
UNITS_IN_ONE_DAY = 1
TAU_UNIT_IN_DAYS = 1 / UNITS_IN_ONE_DAY

# Max of the support of discrete distributions on non-negative times, when created approximating
# continuous distributions.
TAU_MAX_IN_DAYS = 25
TAU_MAX_IN_UNITS = TAU_MAX_IN_DAYS * UNITS_IN_ONE_DAY

# Admitted deviation from 1 when checking the normalization of a probability distribution.
DISTRIBUTION_NORMALIZATION_TOLERANCE = 0.001

# The tolerance in comparing float numbers
FLOAT_TOLERANCE_FOR_EQUALITIES = 1e-10

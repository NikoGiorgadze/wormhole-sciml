"""Reference implementation of the published GEB wormhole dynamics."""

from .dynamics import (
    conserved_energy,
    energy_branch_for_state,
    is_admissible,
    lorentz_factor,
    radial_acceleration,
    radial_acceleration_from_metric,
    timelike_margin,
    total_speed_squared,
    velocity_bounds,
    velocity_from_energy,
)
from .geometry import (
    areal_radius,
    areal_radius_derivative,
    metric_determinant,
    reduced_metric,
    reduced_metric_derivative,
)
from .integrate import integrate_trajectory, integrate_vector_field_continuation
from .observables import (
    azimuth,
    coordinate_visualization,
    effective_angular_velocity,
    terminal_radial_velocity,
)
from .parameters import SpiralParameters, WormholeParameters

__all__ = [
    "SpiralParameters",
    "WormholeParameters",
    "areal_radius",
    "areal_radius_derivative",
    "azimuth",
    "conserved_energy",
    "coordinate_visualization",
    "effective_angular_velocity",
    "energy_branch_for_state",
    "integrate_trajectory",
    "integrate_vector_field_continuation",
    "is_admissible",
    "lorentz_factor",
    "metric_determinant",
    "radial_acceleration",
    "radial_acceleration_from_metric",
    "reduced_metric",
    "reduced_metric_derivative",
    "terminal_radial_velocity",
    "timelike_margin",
    "total_speed_squared",
    "velocity_bounds",
    "velocity_from_energy",
]

from __future__ import annotations

import math
import unittest

import numpy as np

from wormhole_sciml import (
    SpiralParameters,
    WormholeParameters,
    areal_radius,
    areal_radius_derivative,
    conserved_energy,
    energy_branch_for_state,
    integrate_trajectory,
    metric_determinant,
    radial_acceleration,
    radial_acceleration_from_metric,
    reduced_metric,
    terminal_radial_velocity,
    timelike_margin,
    total_speed_squared,
    velocity_bounds,
    velocity_from_energy,
)


class ParameterTests(unittest.TestCase):
    def test_generalized_ellis_bronnikov_requires_even_m(self) -> None:
        for invalid in (0, 1, 3, 5):
            with self.assertRaises(ValueError):
                WormholeParameters(m=invalid)

    def test_terminal_regime_is_reported_not_silently_enforced(self) -> None:
        self.assertTrue(SpiralParameters(alpha=-1.25).has_subluminal_terminal_speed)
        self.assertFalse(SpiralParameters(alpha=-0.5).has_subluminal_terminal_speed)


class GeometryTests(unittest.TestCase):
    def test_areal_radius_symmetry_and_throat(self) -> None:
        for m in (2, 4, 10):
            geometry = WormholeParameters(throat_radius=1.7, m=m)
            l = np.linspace(-4.0, 4.0, 101)
            np.testing.assert_allclose(
                areal_radius(l, geometry), areal_radius(-l, geometry), rtol=0.0
            )
            self.assertEqual(float(areal_radius(0.0, geometry)), 1.7)
            np.testing.assert_allclose(
                areal_radius_derivative(l, geometry),
                -areal_radius_derivative(-l, geometry),
                atol=1e-15,
            )

    def test_analytic_metric_determinant(self) -> None:
        geometry = WormholeParameters(m=6)
        spiral = SpiralParameters(omega=1.0, alpha=-1.25, theta=math.pi / 6)
        l = np.linspace(-5.0, 5.0, 31)
        metric = reduced_metric(l, geometry, spiral)
        numeric = np.linalg.det(metric)
        np.testing.assert_allclose(
            numeric, metric_determinant(l, geometry, spiral), rtol=2e-14, atol=2e-14
        )
        self.assertTrue(np.all(numeric < 0.0))


class DynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = WormholeParameters(throat_radius=1.0, m=2)
        self.spiral = SpiralParameters(omega=1.0, alpha=-1.25, theta=math.pi / 2)

    def test_velocity_bounds_saturate_timelike_constraint(self) -> None:
        l = np.linspace(-20.0, 20.0, 101)
        lower, upper = velocity_bounds(l, self.geometry, self.spiral)
        np.testing.assert_allclose(
            total_speed_squared(l, lower, self.geometry, self.spiral),
            1.0,
            rtol=2e-13,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            total_speed_squared(l, upper, self.geometry, self.spiral),
            1.0,
            rtol=2e-13,
            atol=2e-13,
        )

    def test_paper_acceleration_matches_general_metric_equation(self) -> None:
        l = np.linspace(-8.0, 8.0, 41)[:, None]
        v = np.linspace(0.05, 0.98, 29)[None, :]
        for m in (2, 4, 10):
            geometry = WormholeParameters(m=m)
            closed = radial_acceleration(l, v, geometry, self.spiral)
            general = radial_acceleration_from_metric(l, v, geometry, self.spiral)
            np.testing.assert_allclose(closed, general, rtol=4e-12, atol=4e-14)

    def test_force_free_terminal_velocity_has_zero_acceleration(self) -> None:
        terminal = terminal_radial_velocity(self.spiral)
        for m in (2, 6, 10):
            geometry = WormholeParameters(m=m)
            acceleration = radial_acceleration(
                np.linspace(-50.0, 50.0, 101), terminal, geometry, self.spiral
            )
            np.testing.assert_allclose(acceleration, 0.0, atol=0.0)

    def test_acceleration_is_odd_in_l_at_fixed_velocity(self) -> None:
        for m in (2, 4, 10):
            geometry = WormholeParameters(m=m)
            positive = radial_acceleration(3.2, 0.4, geometry, self.spiral)
            negative = radial_acceleration(-3.2, 0.4, geometry, self.spiral)
            self.assertAlmostEqual(float(positive), -float(negative), places=15)

    def test_energy_notebook_value_and_branch_match(self) -> None:
        l0, v0 = -40.0, 0.7995025
        energy = float(conserved_energy(l0, v0, self.geometry, self.spiral))
        self.assertLess(abs(energy - 0.0072950858953519365), 3e-13)
        branch = energy_branch_for_state(l0, v0, self.geometry, self.spiral)
        matched_velocity = float(
            velocity_from_energy(
                l0, energy, self.geometry, self.spiral, branch=branch
            )
        )
        self.assertAlmostEqual(matched_velocity, v0, places=13)

    def test_invalid_matlab_initial_state_is_rejected(self) -> None:
        self.assertLess(
            float(timelike_margin(0.5, 0.0, self.geometry, self.spiral)), 0.0
        )
        with self.assertRaises(ValueError):
            integrate_trajectory((0.5, 0.0), (0.0, 1.0), self.geometry, self.spiral)


class IntegrationTests(unittest.TestCase):
    def test_energy_conservation_and_tolerance_convergence(self) -> None:
        geometry = WormholeParameters(m=6)
        spiral = SpiralParameters(omega=1.0, alpha=-1.25, theta=math.pi / 2)
        times = np.linspace(0.0, 15.0, 151)
        reference = integrate_trajectory(
            (0.0, 0.1),
            (0.0, 15.0),
            geometry,
            spiral,
            t_eval=times,
            rtol=1e-12,
            atol=1e-14,
        )
        comparison = integrate_trajectory(
            (0.0, 0.1),
            (0.0, 15.0),
            geometry,
            spiral,
            t_eval=times,
            rtol=1e-9,
            atol=1e-11,
        )
        energy = conserved_energy(reference.y[0], reference.y[1], geometry, spiral)
        relative_drift = np.max(np.abs(energy / energy[0] - 1.0))
        self.assertLess(relative_drift, 3e-10)
        self.assertLess(np.max(np.abs(reference.y - comparison.y)), 2e-7)
        self.assertTrue(np.all(timelike_margin(reference.y[0], reference.y[1], geometry, spiral) > 0.0))


if __name__ == "__main__":
    unittest.main()

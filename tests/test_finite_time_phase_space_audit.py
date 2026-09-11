"""Focused numerical tests for the validation-only phase-space audit."""

from __future__ import annotations

import numpy as np

from scripts.audit_finite_time_phase_space import BIN_DEFINITIONS, bin_mask, decompose_phase_error


def test_decomposition_recovers_parallel_and_normal_components() -> None:
    decomposition = decompose_phase_error(
        exact_velocity=np.asarray([2.0, 0.0]),
        exact_second_rate=np.asarray([0.0, 3.0]),
        first_error=np.asarray([4.0, -5.0]),
        second_error=np.asarray([6.0, 7.0]),
        first_scale=2.0,
        second_scale=3.0,
    )
    np.testing.assert_allclose(decomposition["e_parallel"], [2.0, 7.0 / 3.0])
    np.testing.assert_allclose(decomposition["e_perp"], [2.0, 2.5])
    np.testing.assert_allclose(
        decomposition["d_ps"] ** 2,
        decomposition["e_parallel"] ** 2 + decomposition["e_perp"] ** 2,
        rtol=0.0,
        atol=2e-15,
    )


def test_zero_tangent_is_flagged_without_inventing_direction() -> None:
    decomposition = decompose_phase_error(
        np.zeros(2), np.zeros(2), np.ones(2), np.ones(2), 1.0, 1.0
    )
    assert decomposition["invalid_count"] == 2
    assert not np.any(decomposition["valid"])
    assert np.all(np.isnan(decomposition["e_parallel"]))
    assert np.all(np.isnan(decomposition["e_perp"]))


def test_requested_bins_are_disjoint_at_boundaries() -> None:
    values = np.asarray([0.01, 0.05, 0.0500001, 0.10, 0.15, 0.20, 0.30, 0.300001])
    membership = np.vstack([bin_mask(values, definition) for definition in BIN_DEFINITIONS])
    np.testing.assert_array_equal(np.sum(membership, axis=0), np.ones(values.size, dtype=int))
    assert membership[0, 1]
    assert membership[1, 3]
    assert membership[4, 6]
    assert membership[5, 7]

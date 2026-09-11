"""Fast, archive-independent checks of the retained dataset designs."""

from __future__ import annotations

import numpy as np

from wormhole_sciml.phase_b_orbits import (
    SPLIT_COUNTS,
    STRESS_U_TH,
    build_orbit_plan,
    orbit_id_from_u_th,
)
from wormhole_sciml.phase_c_finite_time import ROWS_PER_ORBIT, orbit_design


def test_complete_orbit_plan_is_deterministic_unique_and_split_by_parent() -> None:
    first = build_orbit_plan()
    assert first == build_orbit_plan()
    assert len(first) == 4096 + 1024 + 1024
    values = np.asarray([row["u_th"] for row in first])
    identifiers = [orbit_id_from_u_th(value) for value in values]
    assert len(set(identifiers)) == len(identifiers)
    assert np.unique(values).size == values.size
    assert all(not np.any(values == stress) for stress in STRESS_U_TH)
    for split, strata in SPLIT_COUNTS.items():
        assert sum(row["split"] == split for row in first) == sum(strata.values())


def test_finite_time_row_design_has_fixed_quota_and_exact_identity_rows() -> None:
    for u_th, expected_group in ((0.05, "hard_targeted32"), (0.65, "ordinary_global32")):
        orbit_id = orbit_id_from_u_th(u_th)
        design = orbit_design("validation", orbit_id, u_th)
        assert design["desired_x0"].size == ROWS_PER_ORBIT == 96
        assert np.sum(design["sample_group"] == "global64") == 64
        assert np.sum(design["sample_group"] == expected_group) == 32
        assert np.sum(design["fraction"] == 0.0) == 4
        assert np.sum(design["fraction"] == 1.0) >= 4

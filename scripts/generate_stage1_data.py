#!/usr/bin/env python3
"""Generate the deterministic Stage-1 fixed-step data artifacts."""

from __future__ import annotations

from pathlib import Path

from wormhole_sciml.stage1_data import generate_stage1


if __name__ == "__main__":
    generate_stage1(Path(__file__).resolve().parents[1])

"""Tests for explicit DexHand evaluation profiles."""

from pathlib import Path

import pytest

from hybrid_rollout.dexhand.profiles import PROFILES, get_profile


def test_profiles_separate_smoke_from_d3_comparison():
    smoke = get_profile("cylinder_a_axis_smoke")
    formal = get_profile("cylinder_d3_heldout")

    assert smoke.task == "Isaac-Sharpa-Benchmark-Cylinder-Rotation-A-Axis-v0"
    assert "smoke" in smoke.purpose.lower()
    assert formal.task == "Isaac-Sharpa-In-Hand-Rotation-Cylinder-Dynamic-Motion-v1"
    assert formal.grasp_sampling == "group"
    assert formal.grasp_bank_probability == 1.0
    assert formal.target_speed == 1.0
    assert formal.success_tolerance == 0.1
    assert formal.warmup_steps == 20
    assert formal.wrist_position_range == 0.005
    assert formal.wrist_rotation_range == 0.0872665


def test_profile_record_resolves_and_validates_grasp_bank(tmp_path: Path):
    profile = get_profile("cylinder_d3_heldout")
    bank = tmp_path / profile.grasp_bank_relative
    bank.parent.mkdir(parents=True)
    bank.touch()

    record = profile.record(tmp_path)

    assert record["grasp_bank"] == str(bank.resolve())
    assert record["task"] == profile.task
    assert record["heldout_reference"] is None


def test_world_z_profiles_are_fixed_and_nominal():
    for name in ("cylinder_world_z_dev", "cylinder_world_z_cases", "cuboid_world_z_cases"):
        profile = get_profile(name)
        assert profile.fixed_world_axis == (0.0, 0.0, 1.0)
        assert profile.target_speed == 1.0
        assert profile.grasp_sampling == "state"
        assert profile.wrist_position_range == 0.0
        assert profile.wrist_rotation_range == 0.0


def test_cuboid_slow_profile_is_fixed_at_point_two_rad_per_second():
    profile = get_profile("cuboid_world_z_0p2_cases")
    assert profile.fixed_world_axis == (0.0, 0.0, 1.0)
    assert profile.target_speed == 0.2
    assert profile.task == "Isaac-Sharpa-In-Hand-Rotation-Cuboid-Dynamic-Motion-v1"
    assert profile.grasp_sampling == "state"
    assert profile.case_source_profile == "cuboid_world_z_cases"
    assert profile.wrist_position_range == 0.0
    assert profile.wrist_rotation_range == 0.0


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown DexHand profile"):
        get_profile("implicit-default")


def test_profile_names_are_stable():
    assert tuple(PROFILES) == (
        "cylinder_a_axis_smoke",
        "cylinder_d3_heldout",
        "cylinder_world_z_dev",
        "cylinder_world_z_cases",
        "cuboid_world_z_cases",
        "cuboid_world_z_0p2_cases",
    )

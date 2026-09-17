"""Explicit PTrack task profiles for DexHand Astra evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DexHandTaskProfile:
    """Fully specified simulator conditions for one comparison profile."""

    task: str
    grasp_bank_relative: str
    grasp_sampling: str
    grasp_bank_probability: float
    target_speed: float
    success_tolerance: float
    warmup_steps: int
    wrist_position_range: float
    wrist_rotation_range: float
    purpose: str
    fixed_world_axis: tuple[float, float, float] | None = None
    heldout_reference_relative: str | None = None

    def grasp_bank(self, ptrack_root: Path) -> Path:
        """Resolve and validate this profile's grasp bank."""
        path = (ptrack_root / self.grasp_bank_relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Profile grasp bank does not exist: {path}")
        return path

    def heldout_reference(self, ptrack_root: Path) -> Path | None:
        """Resolve the optional held-out bank used to label generated cases."""
        if self.heldout_reference_relative is None:
            return None
        path = (ptrack_root / self.heldout_reference_relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Profile held-out reference does not exist: {path}")
        return path

    def record(self, ptrack_root: Path) -> dict:
        """Return the resolved profile values stored with an evaluation run."""
        values = asdict(self)
        values["grasp_bank"] = str(self.grasp_bank(ptrack_root))
        heldout_reference = self.heldout_reference(ptrack_root)
        values["heldout_reference"] = (
            str(heldout_reference) if heldout_reference is not None else None
        )
        return values


PROFILES = {
    "cylinder_a_axis_smoke": DexHandTaskProfile(
        task="Isaac-Sharpa-Benchmark-Cylinder-Rotation-A-Axis-v0",
        grasp_bank_relative=(
            "outputs/sharpa_dynamic/cylinder_recoverable_grasps_train80_v1.pt"
        ),
        grasp_sampling="state",
        grasp_bank_probability=1.0,
        target_speed=0.5,
        success_tolerance=0.1,
        warmup_steps=20,
        wrist_position_range=0.005,
        wrist_rotation_range=0.0872665,
        purpose="Minimal A-axis integration smoke; not a formal benchmark.",
    ),
    "cylinder_d3_heldout": DexHandTaskProfile(
        task="Isaac-Sharpa-In-Hand-Rotation-Cylinder-Dynamic-Motion-v1",
        grasp_bank_relative=(
            "outputs/sharpa_dynamic/cylinder_recoverable_grasps_test20_v1.pt"
        ),
        grasp_sampling="group",
        grasp_bank_probability=1.0,
        target_speed=1.0,
        success_tolerance=0.1,
        warmup_steps=20,
        wrist_position_range=0.005,
        wrist_rotation_range=0.0872665,
        purpose="Held-out continuous-rotation comparison against the selected D3 policy.",
    ),
    "cylinder_world_z_dev": DexHandTaskProfile(
        task="Isaac-Sharpa-In-Hand-Rotation-Cylinder-Dynamic-Motion-v1",
        grasp_bank_relative=(
            "outputs/sharpa_dynamic/cylinder_recoverable_grasps_train80_v1.pt"
        ),
        grasp_sampling="state",
        grasp_bank_probability=1.0,
        target_speed=1.0,
        success_tolerance=0.1,
        warmup_steps=20,
        wrist_position_range=0.0,
        wrist_rotation_range=0.0,
        purpose="Non-held-out fixed-world +Z case for Astra integration development.",
        fixed_world_axis=(0.0, 0.0, 1.0),
    ),
    "cylinder_world_z_cases": DexHandTaskProfile(
        task="Isaac-Sharpa-In-Hand-Rotation-Cylinder-Dynamic-Motion-v1",
        grasp_bank_relative=(
            "outputs/sharpa_dynamic/cylinder_recoverable_grasps_test20_v1.pt"
        ),
        grasp_sampling="state",
        grasp_bank_probability=1.0,
        target_speed=1.0,
        success_tolerance=0.1,
        warmup_steps=20,
        wrist_position_range=0.0,
        wrist_rotation_range=0.0,
        purpose="Persistent held-out initial states with a fixed world-frame +Z target axis.",
        fixed_world_axis=(0.0, 0.0, 1.0),
    ),
    "cuboid_world_z_cases": DexHandTaskProfile(
        task="Isaac-Sharpa-In-Hand-Rotation-Cuboid-Dynamic-Motion-v1",
        grasp_bank_relative=(
            "outputs/sharpa_cuboid_transfer/cuboid_00_stable_grasps_v2.pt"
        ),
        grasp_sampling="state",
        grasp_bank_probability=1.0,
        target_speed=1.0,
        success_tolerance=0.1,
        warmup_steps=20,
        wrist_position_range=0.0,
        wrist_rotation_range=0.0,
        purpose="Persistent cuboid initial states with a fixed world-frame +Z target axis.",
        fixed_world_axis=(0.0, 0.0, 1.0),
        heldout_reference_relative=(
            "outputs/sharpa_cuboid_transfer/cuboid_00_stable_grasps_heldout20_v2.pt"
        ),
    ),
}


def get_profile(name: str) -> DexHandTaskProfile:
    """Return a named profile or raise a concise error."""
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown DexHand profile {name!r}; choose from {sorted(PROFILES)}") from error


__all__ = ["DexHandTaskProfile", "PROFILES", "get_profile"]

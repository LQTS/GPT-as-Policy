#!/usr/bin/env python3
"""Add world-axis overlays to an existing persistent DexHand case set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image

from .camera_views import CAMERA_VIEWS
from .case_visuals import annotate_world_axes, make_case_preview, make_contact_sheet


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case-set", type=Path, required=True)
args = parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = args.case_set.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Case-set manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_axis = tuple(float(value) for value in manifest["target"]["axis_unit_vector"])
    target_speed = float(manifest["target"]["speed_rad_s"])

    for entry in manifest["cases"]:
        case_dir = root / entry["case_id"]
        raw_images = {}
        for view in CAMERA_VIEWS:
            name = view["name"]
            image_path = Path(entry["images"][name])
            raw_path = case_dir / f"{name}_rgb_raw.png"
            if not raw_path.is_file():
                shutil.copy2(image_path, raw_path)
            annotated = annotate_world_axes(
                Image.open(raw_path),
                view,
                target_axis=target_axis,
            )
            annotated.save(image_path)
            raw_images[name] = str(raw_path)
        entry["raw_images"] = raw_images
        preview = make_case_preview(
            entry["case_id"],
            {name: Path(path) for name, path in entry["images"].items()},
            case_dir,
            CAMERA_VIEWS,
            target_speed=target_speed,
        )
        entry["preview"] = str(preview)

        metadata_path = case_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["artifacts"]["raw_images"] = raw_images
        metadata["artifacts"]["preview"] = str(preview)
        metadata["visualization"] = {
            "coordinate_frame": "world",
            "axis_colors": {"X": "red", "Y": "green", "Z": "blue"},
            "target_axis": list(target_axis),
        }
        write_json(metadata_path, metadata)

    contact_sheet = make_contact_sheet(manifest["cases"], root)
    manifest["contact_sheet"] = str(contact_sheet)
    manifest["visualization"] = {
        "coordinate_frame": "world",
        "axis_colors": {"X": "red", "Y": "green", "Z": "blue"},
        "target_axis": list(target_axis),
        "target_axis_emphasized": True,
        "raw_images_preserved": True,
    }
    write_json(manifest_path, manifest)
    print(contact_sheet)


if __name__ == "__main__":
    main()

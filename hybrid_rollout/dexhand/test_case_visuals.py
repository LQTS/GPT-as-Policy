"""Tests for DexHand case visualization helpers."""

import numpy as np
from PIL import Image

from hybrid_rollout.dexhand.camera_views import CAMERA_VIEWS
from hybrid_rollout.dexhand.case_visuals import annotate_world_axes


def test_world_axis_overlay_preserves_image_and_draws_pixels():
    image = Image.new("RGB", (640, 480), (128, 128, 128))

    annotated = annotate_world_axes(image, CAMERA_VIEWS[0])

    assert annotated.size == image.size
    before = np.asarray(image)
    after = np.asarray(annotated)
    assert np.count_nonzero(before != after) > 1000


def test_world_axis_overlay_supports_every_camera_view():
    image = Image.new("RGB", (320, 240), (128, 128, 128))

    outputs = [annotate_world_axes(image, view) for view in CAMERA_VIEWS]

    assert all(output.size == image.size for output in outputs)

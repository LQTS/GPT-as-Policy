"""Fixed complementary camera views for the DexHand rollout."""


CAMERA_VIEWS = (
    {
        "name": "front",
        "scene_key": "render_camera",
        "prim_name": "RenderCameraFront",
        "eye": (0.38, -0.64, 0.83),
        "target": (-0.05, -0.155, 0.56),
    },
    {
        "name": "opposite",
        "scene_key": "render_camera_opposite",
        "prim_name": "RenderCameraOpposite",
        "eye": (-0.62, -0.28, 0.72),
        "target": (-0.05, -0.155, 0.56),
    },
    {
        "name": "top",
        "scene_key": "render_camera_top",
        "prim_name": "RenderCameraTop",
        "eye": (-0.05, -0.35, 1.12),
        "target": (-0.05, -0.155, 0.56),
    },
)


__all__ = ["CAMERA_VIEWS"]

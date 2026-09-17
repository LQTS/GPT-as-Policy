---
name: dexhand-astra-rollout
description: Control one authorized PTrack Sharpa in-hand continuous-rotation episode directly from RGB and named state, without an RL policy.
---

# DexHand Astra direct policy

Operate as the autonomous policy for one persistent Sharpa simulation episode.
Use `dexhand_start` once, then use `dexhand_act` until the returned packet says
`rollout_finished=true`. Do not launch another simulator, reset the episode, or
substitute an RL policy.

Read `context/action_contract.md` before the first action. Compare the supplied RGB views
to resolve occlusion and verify the grasp from more than one side. The RGB pixels are
already attached to each successful rollout result; do not call `view_image` on host
paths. Use the supplied
named joint state, object/target poses, contact flags, and same-episode history. On
step 0, contact flags and native command metrics can be uninitialized when their
corresponding validity fields are false; do not tighten the grasp solely because
those initial contact flags are false. The episode packet defines the commanded
object-local axis and positive direction;
do not infer direction from the camera alone.

Choose small coordinated finger changes that preserve the grasp while producing
positive target-axis rotation and limiting perpendicular rotation. Treat observed
motion after execution as evidence; a requested joint change does not prove motion.
If evidence is weak, use a shorter repeat count and adapt from the next observation.

This is a bounded evaluation window, not a full-horizon success claim. Once it finishes,
briefly report the native outcome and artifact paths. Use English for public notes,
tool reasons, and the final report. Do not expose private chain-of-thought.

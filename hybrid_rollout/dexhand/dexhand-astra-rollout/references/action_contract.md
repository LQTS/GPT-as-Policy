# Sharpa action contract

Each observation gives the authoritative 22-joint order. A `joint_delta` entry is
added to that joint's current normalized target. Every entry must be within
`[-0.1, 0.1]`. The host clips the accumulated absolute target to `[-1, 1]` and
reports saturated indices. PTrack then maps the target to physical joint limits
and applies its unchanged EMA/PD controller.

Call `dexhand_act` with:

```json
{
  "response": {
    "request_id": "current request id",
    "joint_delta": [22 finite numbers],
    "repeat_steps": 1,
    "reason": "brief visible evidence and intended effect"
  }
}
```

`repeat_steps` is an integer from 1 through 10. One control step is 0.05 seconds.
The host validates the entire response before stepping physics; rejected input
executes nothing. Executed actions are never rolled back or silently replayed.

Joint suffixes identify flexion/extension (`FE`), abduction/adduction (`AA`),
proximal/interphalangeal joints (`PIP`, `DIP`, `IP`), and the pinky/thumb bases
(`CMC`). Positive delta means motion toward that asset joint's upper limit; use
the returned state and measured object motion to determine its physical effect.

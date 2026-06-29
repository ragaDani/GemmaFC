# Google Research Football Setup

GemmaFC uses Google Research Football as the local soccer simulation environment.
The upstream project recommends Docker for Linux-based builds to avoid package
version conflicts. On macOS, native rendering is possible, but the Docker path is
more reliable for a hackathon setup and can run headless API checks with `xvfb`.

## Local Source Checkout

The upstream source is kept as an ignored local dependency checkout:

```shell
git clone https://github.com/google-research/football.git external/google-research-football
```

`external/` is ignored because it is a third-party checkout, not GemmaFC source.

## Build Docker Image

Build from the upstream checkout while using GemmaFC's compatibility Dockerfile:

```shell
docker build \
  --build-arg DOCKER_BASE=ubuntu:20.04 \
  -f docker/gfootball.Dockerfile \
  -t gemmafc-gfootball:latest \
  external/google-research-football
```

The Dockerfile follows the upstream apt dependency list and install flow, but
pins `pip`, `setuptools`, and `wheel` because newer packaging tools reject the old
`gym<=0.21.0` metadata used by Google Research Football. It also installs `six`,
which the runtime imports but the current upstream package metadata does not pull in.

## Verify Python API

```shell
docker run --rm gemmafc-gfootball:latest python3 - <<'PY'
import json
import gfootball.env as football_env

env = football_env.create_environment(
    env_name="academy_empty_goal_close",
    representation="simple115v2",
    render=False,
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
)
obs = env.reset()
step_obs, reward, done, info = env.step(0)
env.close()
print(json.dumps({
    "status": "ok",
    "obs_type": type(obs).__name__,
    "obs_shape": list(getattr(obs, "shape", [])),
    "step_obs_shape": list(getattr(step_obs, "shape", [])),
    "reward": float(reward),
    "done": bool(done),
    "info_type": type(info).__name__,
}))
PY
```

## Verify Headless Rendering

```shell
docker run --rm gemmafc-gfootball:latest timeout 30s xvfb-run -s "-screen 0 1280x720x24" python3 -u - <<'PY'
import json
import numpy as np
import gfootball.env as football_env

env = football_env.create_environment(
    env_name="academy_empty_goal_close",
    representation="pixels",
    render=True,
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
)
obs = env.reset()
last = obs
rewards = []
for _ in range(5):
    last, reward, done, info = env.step(0)
    rewards.append(float(reward))
env.close()
print(json.dumps({
    "status": "render_ok",
    "obs_shape": list(obs.shape),
    "last_shape": list(last.shape),
    "pixel_min": int(np.min(last)),
    "pixel_max": int(np.max(last)),
    "pixel_mean": float(np.mean(last)),
    "steps": len(rewards),
    "done": bool(done),
    "rewards": rewards,
}))
PY
```

## Interactive Play

The upstream command is:

```shell
python3 -m gfootball.play_game --action_set=full
```

On Linux with X11, run the container with display access as described in the
upstream Docker guide. On macOS, the upstream docs recommend native install for
visible rendering; for GemmaFC's agent loop, headless Docker rendering is enough.

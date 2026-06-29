# GemmaFC App UI

GemmaFC ships a small Docker-runnable cockpit for the Google Research Football prototype. It serves a static frontend from a FastAPI app and steps the rendered GRF environment inside the container under Xvfb.

## Build

Build the Google Research Football base image first:

```shell
docker build \
  --build-arg DOCKER_BASE=ubuntu:20.04 \
  -f docker/gfootball.Dockerfile \
  -t gemmafc-gfootball:latest \
  external/google-research-football
```

Then build the app image:

```shell
docker build -f docker/app.Dockerfile -t gemmafc-app:latest .
```

## Run

```shell
docker run --rm --env-file app/.env -p 8001:8000 gemmafc-app:latest
```

Open `http://localhost:8001`.

With Compose:

```shell
docker compose up --build app
```

## What The UI Shows

- Live rendered GRF frames from selectable Google Research Football scenarios.
- Scenario and opponent selectors, including a built-in AI opponent mode backed by GRF's bundled bot player.
- Controller controls for Cerebras and GPU. Both configured providers can run continuously or tick once.
- Tick controls for run, single step, and reset.
- Current proposed action, evaluator score, and repair note.
- Observe -> propose -> evaluate -> repair -> act telemetry.
- Runtime counters for step count, reward, latency, and repair count.

## Current Limitation

Cerebras and GPU can now drive live GRF ticks from the cockpit using structured state plus the current rendered frame as image input. The local scout remains as a backend fallback/debug path. Official PPO checkpoint opponents are still deferred because they require the separate TensorFlow 1.15/Baselines setup described by upstream GRF.

## Port note

The container listens on port `8000`. The local host mapping uses `8001` because other development services commonly occupy `8000`.

## Render quality

The UI requests `320x240` pixel observations from Google Research Football and sends that native frame to the browser. CSS scales the image to the available display size, which shifts upscale work out of Docker and onto the browser compositor.

## Recording View

Open `http://localhost:8001/record` for the hackathon capture surface. The page is designed for a `1280x720` browser recording: the game frame is encoded as `960x720`, and the right-side telemetry rail fills the remaining `320x720`.

The standard cockpit at `http://localhost:8001` also includes a `Live` / `720p` quality selector. Use `Live` for lower-latency iteration and `720p` when recording.

## Provider Comparison

The cockpit includes a `Provider race` panel with `Cerebras` and `GPU` lanes. It runs the same provider decision harness used by live control, reports latency and proposed next action, and shows a speed multiplier for the faster lane.

Secrets stay out of the image. The app reads provider keys from runtime environment variables. For Docker, pass them with `--env-file app/.env`; do not bake `.env` into the image.

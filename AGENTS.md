# GemmaFC Agent Guide

GemmaFC is a hackathon prototype where Gemma controls a local Google Research Football soccer environment.

## Project Goal

Build a concise demo that shows Cerebras-hosted Gemma 4 31B improving real-time gameplay UX through fast observation, strategy, play, repair, and action loops.

## Product Framing

- Project name: GemmaFC
- Primary demo surface: Google Research Football running locally in Docker.
- Core thesis: Cerebras speed lets Gemma observe, strategize, choose coordinated plays, repair invalid actions, and act within an interactive control window.
- Primary hackathon tracks: Multiverse Agents, Enterprise/real-world impact by analogy to real-time decision systems, and social reach through soccer/gameplay.
- Demo limit: 60 seconds.

## Core Constraints

- Cerebras Gemma 4 31B should remain the primary model path.
- The useful latency metric is time to a corrected action, not just first token.
- The demo should make the control loop visible: observe frame/state, strategize, choose play actions, repair invalid actions, act again.
- Comparisons should be framed fairly: same environment, same observation budget, same action schema, same scenario.
- Google Research Football is a local simulation dependency, not vendored GemmaFC source.

## Architecture Direction

Target the smallest compelling prototype before broadening scope:

1. Environment adapter resets and steps Google Research Football scenarios.
2. Observation encoder produces compact state plus optional rendered frames.
3. Strategist builds a concise scoring plan from the image and structured state.
4. Play caller proposes legal actions using the same action schema for single-player and squad control.
5. Repairer adjusts missing, unsupported, or invalid actions before the next tick.
6. Telemetry records phase latency, correction count, reward, possession, shots, goals, and scenario completion.
7. Demo UI or replay view shows GemmaFC acting faster than slower control lanes.

## Local Setup

- Upstream source checkout: `external/google-research-football`.
- Build image: `docker build --build-arg DOCKER_BASE=ubuntu:20.04 -f docker/gfootball.Dockerfile -t gemmafc-gfootball:latest external/google-research-football`.
- API setup details live in `docs/google-research-football-setup.md`.
- App image: `docker build -f docker/app.Dockerfile -t gemmafc-app:latest .`.
- App run: `docker run --rm -p 8001:8000 gemmafc-app:latest`, then open `http://localhost:8001`.
- `external/` is ignored because it contains third-party source checked out locally for builds.

## Documentation Rules

- Capture durable product and technical choices in `docs/design-decisions.md`.
- Add dated entries as decisions change instead of rewriting history.
- Keep hackathon-specific rationale explicit so future agents preserve the demo story.
- Avoid unrelated refactors while the prototype is still taking shape.

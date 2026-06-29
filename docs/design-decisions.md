# Design Decisions

This is the living decision log for GemmaFC, formerly ProofLoop. Add dated entries as product, technical, benchmark, and demo choices change.

## 2026-06-28: Make Verification The Speed Story

**Decision:** ProofLoop will frame Cerebras speed as the ability to complete a full generate -> verify -> critique -> repair loop fast enough for an interactive user experience.

**Rationale:** The hackathon asks teams to show how Cerebras speed improves UX. "Faster first token" is less compelling than "verified answer before slower systems finish drafting."

**Implications:**

- The main metric is time to first verified answer.
- The UI should show progress through multiple agents, not just one chat response.
- Repair loops are part of the value proposition, not hidden internal work.

## 2026-06-28: Use A Fair Same-Harness Benchmark

**Decision:** The primary comparison should use the same agent graph, prompts, schemas, evidence packet, and repair policy across providers.

**Rationale:** The demo is stronger if Cerebras wins on latency while the workflow stays constant.

**Initial lanes:**

- Cerebras Gemma 4 31B as the primary proof-loop lane.
- GPU-hosted Gemma 4 31B as the apples-to-apples provider comparison lane.
- A larger hosted model as an optional challenger/control lane when it does not confuse the story.

## 2026-06-28: Optimize For A Curated Enterprise Compliance Demo

**Decision:** The first demo task should be a curated compliance-risk workflow using vendor security PDFs plus website evidence.

**Target prompt:** "From these vendor security PDFs and the vendor website, identify the top 3 compliance risks, cite the exact visual or textual evidence, and flag any contradictions."

**Rationale:** This task naturally exercises multimodal evidence, citation verification, enterprise relevance, and contradiction repair.

**Killer moment:** A website claims SOC 2 Type II while uploaded report evidence shows SOC 2 Type I. The verifier catches the unsupported claim and the repairer changes the final answer with a page or region citation.

## 2026-06-28: Build The Smallest Compelling Prototype First

**Decision:** Prioritize a narrow, polished path over a broad generic document platform.

**Rationale:** The hackathon demo is limited to 60 seconds. A reliable curated path with visible agents, timers, citations, and repair diff is more useful than a larger incomplete system.

**Prototype acceptance criteria:**

- Shows at least one multimodal or layout-sensitive evidence item.
- Streams visible agent state.
- Produces an initial answer, verification finding, repair diff, and final verified answer.
- Shows time-to-draft and time-to-verified answer per lane.
- Makes the Cerebras lane visibly complete the proof loop faster than the comparison lane.

## Open Questions

- Which exact GPU-hosted Gemma 4 model ID and provider will be available during the hackathon?
- Should the first version call live providers, use a replay fixture, or support both?
- What evidence packet will be safest to show in a public 60-second demo?

## 2026-06-28: GemmaFC Uses Google Research Football As The Soccer Sandbox

**Decision:** Pivot the prototype from ProofLoop document verification to GemmaFC, where Gemma controls a local Google Research Football environment.

**Rationale:** Soccer gameplay is visual, real-time, demo-friendly, and naturally shows Cerebras speed as better perception-action latency rather than only faster text generation.

**Implementation choice:** Use Google Research Football through Docker first. Keep the upstream checkout under `external/`, ignore it from git, and keep GemmaFC-specific setup in `docker/gfootball.Dockerfile` plus `docs/google-research-football-setup.md`.

**Implications:**

- The first acceptance target is a headless Docker environment where the Python API can reset and step a scenario.
- Visible macOS rendering is a secondary target because upstream recommends native install for macOS/Windows rendering.
- GemmaFC should own the agent loop and control policy while treating Google Research Football as a local simulation dependency.

## 2026-06-28: Pin Legacy Python Packaging For Google Research Football

**Decision:** Keep the upstream Ubuntu 20.04 Docker setup, but pin `pip==23.3.2`, `setuptools==65.5.0`, and `wheel==0.37.1`, then install `six` explicitly.

**Rationale:** The upstream Dockerfile upgrades to latest packaging tools, which now reject the legacy `gym<=0.21.0` metadata. The runtime also imports `six` without pulling it from current package metadata.

**Implications:**

- Docker is the primary local environment for the hackathon prototype.
- The compatibility Dockerfile lives in GemmaFC, while the upstream Google Research Football checkout remains ignored under `external/`.
- The verified setup can reset and step both `simple115v2` observations and Xvfb-backed pixel observations.


## 2026-06-28: Serve The First GemmaFC UI From The Simulation Container

**Decision:** Build the first GemmaFC app UI as a FastAPI plus static frontend container layered on top of the working Google Research Football image.

**Rationale:** The fastest useful demo path is a single Docker app that can render frames, step the environment, and show the observe -> propose -> evaluate -> repair -> act loop without requiring a separate frontend toolchain.

**Implications:**

- The first UI is operational, not a marketing page.
- The local scout policy is a placeholder behind the same adapter shape intended for Gemma.
- Future Cerebras integration should replace proposal/evaluation internals without changing the frontend contract.


## 2026-06-28: Improve Demo Rendering Without Changing The Control Loop

**Decision:** Increase the rendered observation from `96x72` to `320x240`, upscale to `960x720` with Lanczos filtering, and remove nearest-neighbor browser scaling. This was later superseded for live playback by browser-side scaling.

**Rationale:** The first UI made the game state visibly too pixelated for a demo recording. The improved render path keeps the same GRF environment and action loop while making the soccer field readable.

**Implications:**

- API frames are larger, so frame encoding is slightly heavier.
- The control contract remains unchanged.
- If latency becomes the limiting factor, make render quality configurable instead of reducing the default demo quality.


## 2026-06-28: Move Live Upscaling To The Browser

**Decision:** Keep GRF pixel observations at `320x240`, encode that native frame in Docker, and let browser CSS scale it to the UI display size.

**Rationale:** Server-side `960x720` upscaling made frames clearer, but it increased Docker CPU work and frame payload size. Browser-side scaling tests whether the host browser/compositor can provide acceptable visual quality with lower simulation-loop overhead.

**Implications:**

- Live API frame payloads should be much smaller than server-upscaled frames.
- The displayed image may be slightly softer than Lanczos-upscaled JPEGs.
- If demo quality is insufficient, add a render-quality toggle instead of hardcoding one path.


## 2026-06-28: Compare Cerebras Against Anonymous GPU Inference

**Decision:** The UI labels provider lanes as `Cerebras` and `GPU`, while the backend keeps the specific GPU provider implementation behind an adapter.

**Rationale:** The hackathon story needs a clear Cerebras speed comparison without overemphasizing the competing provider brand. Both lanes receive the same compact game-state prompt and action schema.

**Implications:**

- Provider keys are runtime environment variables loaded via `--env-file app/.env`.
- The app image must not copy `.env`.
- Remote inference is opt-in from the UI so the local football loop remains responsive.


## 2026-06-28: Add Scenario And Built-In AI Matchup Selection

**Decision:** GemmaFC should expose a small curated scenario catalog in the UI and default to `academy_run_to_score_with_keeper` against a built-in Google Research Football AI opponent.

**Rationale:** The soccer demo needs visible pressure and variation without taking on the TensorFlow 1.15/OpenAI Baselines setup required by the official PPO checkpoints. GRF's native scenario behavior plus the bundled `bot` player gives us a reliable local opponent path inside the current Docker image.

**Implications:**

- The backend owns the scenario and opponent catalog so provider prompts, UI selectors, and environment creation stay aligned.
- The first opponent modes are `Scenario default` and `Built-in AI`; PPO checkpoints remain a follow-up once the demo loop is stable.
- The action catalog includes passes and GRF's `builtin_ai` action so future provider-driven play can choose richer football actions from the same harness.


## 2026-06-28: Make Provider Inference The Live Controller

**Decision:** The main cockpit should let the operator run the match with `Cerebras`, `GPU`, or the local fallback scout as the active controller. Provider output is parsed into the same action schema, checked by the evaluator, repaired when unsupported or unsafe, and then applied to the Google Research Football step.

**Rationale:** A provider race panel is useful, but the demo is stronger when the run button itself shows Cerebras speed changing the gameplay loop. The UI should foreground controller choice and decision latency while keeping the field render smaller.

**Implications:**

- `Run Cerebras` and `Run GPU` execute live inference ticks instead of only comparing text responses.
- The local scout remains a deterministic fallback and debugging control.
- The cockpit layout is now a control desk: smaller pitch monitor, larger controller/action/provider telemetry area.


## 2026-06-28: Remote-Only Demo Controls With Vision Frames

**Decision:** The public cockpit should show only `Cerebras` and `GPU` as live controllers. Cerebras can run continuously; GPU is manual tick only. Each remote controller request includes the current rendered football frame as an image input plus compact structured state.

**Rationale:** The hackathon story is a provider speed comparison, so the local fallback should not compete for visual attention. GPU latency is high enough that continuous run creates a stalled demo. Sending the frame lets Gemma use visual game understanding rather than only synthetic state summaries.

**Implications:**

- Local scout remains available only as a backend fallback/debug path.
- `Run` is enabled for Cerebras and disabled for GPU; `Tick GPU` remains available.
- Provider prompts use multimodal message content with text state and a JPEG `image_url` frame.


## 2026-06-28: Expose Full GRF Action Set To Gemma

**Decision:** Gemma receives the complete Google Research Football `full_action_set` catalog: movement, pass variants, shot, sprint/dribble, defensive pressure, keeper rush, switch, built-in AI, and release actions.

**Rationale:** The soccer-agent demo should let Gemma make real football choices instead of only moving right and occasionally shooting. High pass and long pass cover cross-like behavior because GRF does not expose a separate `cross` action.

**Implications:**

- Provider prompts include all 33 full-action ids.
- Superseded on 2026-06-29: the verifier no longer applies tactical repairs to valid actions; it only handles unsupported action ids.
- The app still controls one active/designated player per tick; full multi-player action arrays remain future work.

## 2026-06-29: Curate The GemmaFC Cockpit Around Live Control

**Decision:** The main app UI should read as a touchline control desk: compact field monitor on the left, controller choice and run controls as the dominant rail, then action/provider/loop/runtime telemetry in fixed zones.

**Rationale:** The hackathon recording needs viewers to understand the Cerebras-vs-GPU control story in seconds. Treating every component as an equal card scattered attention across the page and made the operator workflow harder to scan.

**Implications:**

- Controller selection and run/tick/reset controls occupy the top of the telemetry area.
- The rendered football screen stays smaller so model decisions and timing take center stage.
- Provider race, current action, loop trace, and runtime counters have stable grid positions for recording and narration.

## 2026-06-29: Remove Tactical Bias From Gemma Control

**Decision:** GemmaFC now passes valid provider actions through directly instead of repairing early shots, repeated sprint, idle, or other legal choices into `Right`.

**Rationale:** The demo should show Gemma making football decisions from the current game state, not the app steering it through hardcoded attacking heuristics. The evaluator remains useful for validity handling, but it should not impose a scripted tactic.

**Implementation choice:** Provider context now includes a neutral objective plus raw GRF tactical state from `env.unwrapped.observation()`: ball position, possession, active player, game mode, sticky actions, nearby teammates/opponents, recent actions, and the full action catalog.

**Implications:**

- Cerebras/GPU actions are executed directly when they are legal GRF action ids.
- The repair counter should stay at zero unless the provider returns an unsupported action.
- More varied play is now possible, but quality depends on Gemma's ability to use the tactical state and frame rather than on backend heuristics.

## 2026-06-29: Image-First Football Reasoning And Squad Agents

**Decision:** Gemma prompts now require image-first deduction of the current football state before using raw GRF coordinates, and the cockpit exposes a `Single` / `Squad` control mode. In Squad mode, GemmaFC runs one Gemma decision per controlled yellow-jersey player and sends the resulting action list to Google Research Football.

**Rationale:** A soccer-agent demo should show visual game understanding, not just structured-state exploitation. Gemma needs the rules, team identity, field orientation, and practical attacking heuristics in the system prompt, while still seeing raw state for verification.

**Implementation choice:** The prompt identifies Gemma's team as the yellow jerseys, explains the halfway line and scoring objective, and includes an attacking checklist: observe shape, use width/depth, pass-and-move, create overloads, use through balls/cross-like high passes, shoot from credible chances, and press or recover when out of possession.

**Implications:**

- Superseded by the coordinated strategy/play pipeline below: Squad mode should not make one provider call per player.
- Squad mode is capped to curated scenario player counts, with a maximum of four controlled yellow players for demo viability.
- The UI keeps Single as the default and exposes Squad as an explicit operator choice.

## 2026-06-29: Share One Strategy-Play Pipeline Across Single And Squad

**Decision:** Single-player and Squad control both use the same `observe -> strategize -> play -> repair -> act` pipeline. The difference is only the number of controlled yellow players and the action payload sent to Google Research Football.

**Rationale:** The previous Squad shape could drift into sequential per-player provider calls, which creates coordination problems and makes timeouts likely. A shared pipeline keeps the code cleaner and makes the product behavior easier to explain in the demo.

**Implementation choice:** `FootballSession.step()` now delegates to typed phase handlers. `Strategize` creates one coordinated scoring plan, `Play` returns one action per controlled player in a single response, `Repair` validates each action independently, and `Act` sends either one action id or a GRF action list.

**Implications:**

- Cerebras and GPU paths use the same phase order, prompts, schemas, and repair policy.
- Squad mode makes a team-level strategy call and a team-level play call rather than one call per player.
- Missing or unsupported per-player actions are repaired by delegating only that player to GRF built-in AI for the tick.

## 2026-06-29: Shape Cerebras Prompts For Cached Tokens

**Decision:** Cerebras live control now uses one combined decision flow per tick with a stable controller manual in the system prompt, a compact dynamic state suffix, a smaller model frame, medium reasoning for the strategize/decision call, low reasoning for the play-action schema call, no reasoning for other calls, and a stable `prompt_cache_key`.

**Rationale:** Hackathon accounts have a 100K uncached-token-per-minute limit, while cached prompt tokens do not count against that limit. The fastest reliable demo path is therefore not just shorter prompts; it is a repeated prompt prefix that Cerebras can cache while the current frame and GRF state remain at the end.

**Implementation choice:** The reusable prefix contains the role, football rules, tactical checklist, response schema, and model-facing GRF action catalog. The changing user message contains only tick metadata and compact state. The strategy and play-action calls both use strict JSON schema response format; the play schema includes an action-id enum that excludes GRF built-in delegation. The backend parses the schema payload, repairs invalid actions, and executes the local GRF step. Streaming responses include usage telemetry so cached prompt-token hits can be checked from the app.

**Implications:**

- The visible loop remains `observe -> strategize -> play -> repair -> act`, but remote strategy and play run as two Cerebras calls inside one coordinated decision flow.
- Auto-run schedules the next `/api/step` after the previous tick returns and the post-action frame image has loaded, relying on prefetch and provider wait rather than a fixed browser delay.
- `CEREBRAS_STRATEGIZE_REASONING_EFFORT`, `CEREBRAS_PLAY_TOOL_REASONING_EFFORT`, `CEREBRAS_REASONING_EFFORT`, and `CEREBRAS_PROMPT_CACHE_KEY` can override the default `medium` strategize reasoning, default `low` play-tool reasoning, default `none` reasoning for other calls, and `gemmafc-live-control-v3-keeper-dribble` cache key.
- Missing, malformed, unsupported, or delegated provider actions repair to `Idle` instead of `Built-in AI` so the demo never silently hands the live controller to GRF.
- `CEREBRAS_STRATEGY_MAX_TOKENS`, `CEREBRAS_PLAY_TOOL_MAX_TOKENS`, and `CEREBRAS_DEFAULT_MAX_TOKENS` cap the SDK per-call max output/completion budget; defaults are 1600 for each.

## 2026-06-29: Post-Action Frame-Gated Auto-Run

**Decision:** The frontend no longer uses a fixed remote run interval. When Run is active, the next `/api/step` is queued after the previous legal action response is rendered and the new post-action `frame.jpg` has loaded in the cockpit.

**Rationale:** The fixed interval was added as a conservative TPM safety throttle during prompt-cache work, but it made the scenario counter advance slowly even when a prepared decision was already available. Queuing the next tick immediately after JSON render can advance the loop before the visible post-action frame has loaded. Waiting for the post-action frame keeps the UI and Gemma planning cadence aligned.

**Implications:**

- Auto-run cadence is now bounded by provider readiness, GRF `env.step`, post-action frame loading, and response rendering rather than an arbitrary browser timer.
- This can consume more Cerebras requests per minute when decisions are prepared quickly.

## 2026-06-29: Possession-Safe Keeper Strategy

**Decision:** Keeper-beating guidance now lives only in the strategy prompt. The play prompt is a compact action-schema prompt that follows `strategy_plan`, `STATE_JSON`, and GRF sticky-action mechanics without adding its own keeper maneuver.

**Rationale:** The previous play prompt repeated the keeper maneuver instruction and caused actions like `Top right` to escape the keeper while leaving a loose ball behind. GRF's `Dribble` action is sticky, effective when the player has the ball, and improves close control, while `Sprint` worsens ball handling. Strategy should therefore plan possession-safe lane creation before play chooses the next legal action.

**Implications:**

- Strategy responses should recover/control loose balls before any keeper maneuver.
- If the ball carrier has possession and the keeper blocks the direct shot, strategy should plan close-control dribble, then a top/bottom angle, then a shot only after the lane opens.
- Play responses should execute the strategy with one legal GRF action, not introduce new keeper-maneuver logic.

## 2026-06-29: Reduce Per-Tick Transport Payloads

**Decision:** Runtime state snapshots now expose the rendered game frame through `/api/frame.jpg` instead of embedding the base64 JPEG in every JSON response. The live Cerebras decision flow sends the frame only to the strategize phase; the play-action phase receives the compact state and strict strategy JSON without a duplicate image payload.

**Rationale:** The network payload was dominated by repeated frame and state bytes. Splitting the UI frame transport from JSON state keeps the cockpit responsive, while dropping the duplicate image from the play call leaves more token budget for the actual structured action payload.

**Implications:**

- `/api/state`, `/api/reset`, and `/api/step` carry `frame_url` rather than inline `frame` data.
- The frontend still accepts legacy inline `frame` fields for compatibility.
- Provider comparison now uses the compact provider context so manual races match the live loop more closely.

## 2026-06-29: Separate Visible Wait From Background Gemma Time

**Decision:** Provider telemetry now reports visible wait separately from background Gemma wall time. For prefetched decisions, the main provider card uses the time spent waiting for the prepared decision during `/api/step`, while the explanatory text still shows the full background Gemma duration and its strategy/play split.

**Rationale:** Prefetch intentionally moves the model call after the previous action. The browser can see a `/api/step/cerebras` response in about 100 ms while the provider result contains a 2 s background decision. Showing that 2 s value as the card's main total made the UI look like it contradicted DevTools.

**Implications:**

- Combined strategy/play decisions expose strategy latency, play latency, and first-token timings for each phase.
- The combined `first_token_ms` now refers to the first streamed JSON content from the whole decision, not the play phase's first token relative to the play call.
- `/api/step` request time, visible wait, background Gemma time, and GRF action time should be interpreted as different measurements.

## 2026-06-29: Prefetch The Next Gemma Decision After Acting

**Decision:** After a remote-controlled action is successfully applied to Google Research Football, the backend starts the next Cerebras decision in the background for the resulting frame. The next tick consumes that prepared decision when it matches the current scenario, control mode, controller, player count, and step.

**Rationale:** The live cockpit should measure time to a corrected action, not force every action to wait synchronously on Gemma. Cerebras may still spend around a second on medium-reasoning tool calls, but when the decision is prepared between ticks the next visible action can execute with only the remaining wait plus GRF act time.

**Implications:**

- The first remote tick after reset may still call Gemma synchronously; subsequent ticks can use post-action prepared decisions.
- Provider telemetry still reports Gemma's actual inference latency, while the current action latency reports how long the tick waited for a prepared decision.
- Stale prepared decisions are discarded when scenario, controller, control mode, player count, step, or done state no longer match.

## 2026-06-29: Enable GPU Auto-Run For Direct Provider Comparison

**Decision:** The cockpit allows any configured remote controller to use the continuous `Run` loop. `Run Cerebras` posts to `/api/step/cerebras`; `Run GPU` posts to `/api/step/gpu`; tick controls use the same explicit per-controller routes.

**Rationale:** The demo still foregrounds Cerebras as the primary path, but operators need to auto-run the GPU lane to see how it fares under the same frame-gated loop, scenario, control mode, action schema, repair policy, and GRF step cadence.

**Implications:**

- The older GPU manual-only UI restriction is superseded.
- GPU auto-run can make the cockpit appear stalled while the provider request is in flight; that latency is part of the comparison.
- Remote prefetch remains keyed by controller, so prepared GPU and Cerebras decisions cannot be consumed by the other lane.

## 2026-06-29: Guard Strategy Against Penalty-Spot Ball Confusion

**Decision:** Strategy prompts explicitly tell Gemma that the white penalty spot/dot is a fixed field marking, not the ball. The strategy agent must identify the ball using nearby player attention, possession, motion context, and `STATE_JSON` ball fields before planning control or scoring actions.

**Rationale:** In keeper scenarios, the rendered white penalty mark can resemble the ball and cause the model to plan around the wrong object. The strategy phase is the right place to correct this because it performs the image-first scene read before play-action selection.

**Implications:**

- The strategy checklist includes the penalty-spot guardrail before possession and scoring decisions.
- The strategy user message repeats the rule near the image-first instruction.
- The default Cerebras prompt cache key is bumped so live runs use the updated prompt prefix.

## 2026-06-29: Maneuver Around Rushing Goalkeepers

**Decision:** The strategy prompt explicitly handles the case where the goalkeeper rushes the ball carrier. If yellow has possession and the keeper charges out, Gemma should keep close control with `Dribble`, maneuver through the open top/bottom or diagonal lane, and only shoot or pass after moving around the keeper.

**Rationale:** The live scoring scenarios often put the keeper between the attacker and goal. When the keeper rushes, a direct shot or sprint can lose the ball; the strategy phase should plan a possession-safe dribble around the challenge before the play phase selects the next legal action.

**Implications:**

- Keeper-beating logic remains in the strategist, not the action-schema prompt.
- The default Cerebras prompt cache key is bumped again so live runs pick up the new keeper-rush instruction.

## 2026-06-29: Compare Provider Harnesses Honestly

**Decision:** Live control and provider comparison both keep the provider-specific harness shape: Cerebras runs a strategy call followed by an `execute_play` tool-call for legal actions, while GPU remains a single chat-completion API call using the combined decision prompt. The `Provider race` panel displays the resulting x-speed multiplier.

**Rationale:** The demo should compare the practical control path, not force both providers through an artificial shape. Cerebras can use its fast two-phase strategy/tool loop, while GPU shows the slower single-call lane under the same scenario, frame, action catalog, repair policy, and GRF act step.

**Implications:**

- `/api/providers/compare` calls both providers with `task="decision"` so the same dispatch path as live remote ticks is used.
- Cerebras play-action responses prefer the `execute_play` tool when enabled; GPU remains a normal chat response parsed as JSON.
- The frontend computes and displays the race winner as a multiplier, such as `Cerebras 4.2x faster`.

from __future__ import annotations

import asyncio
import concurrent.futures
import base64
import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gfootball.env as football_env
from gfootball.env import config as football_config
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
JPEG_QUALITY = 92
MODEL_FRAME_WIDTH = 240
MODEL_FRAME_HEIGHT = 180
MODEL_JPEG_QUALITY = 72
RENDER_MODES = {
    "live": {"label": "Live", "width": FRAME_WIDTH, "height": FRAME_HEIGHT, "upscale": False, "quality": 92},
    "recording": {"label": "Recording 720p", "width": 960, "height": 720, "upscale": True, "quality": 94},
}

SCENARIO_OPTIONS = [
    {
        "id": "academy_run_to_score_with_keeper",
        "label": "Run to score + keeper",
        "description": "One attacker starts near midfield and finishes against a keeper while defenders chase.",
    },
    {
        "id": "academy_pass_and_shoot_with_keeper",
        "label": "Pass and shoot",
        "description": "Two attackers choose between carrying, passing, and shooting against a keeper.",
    },
    {
        "id": "academy_3_vs_1_with_keeper",
        "label": "3v1 + keeper",
        "description": "Three attackers break against one defender and a keeper.",
    },
    {
        "id": "academy_counterattack_easy",
        "label": "Counterattack easy",
        "description": "A 4v1 break with trailing players recovering toward the ball.",
    },
    {
        "id": "1_vs_1_easy",
        "label": "1v1 easy",
        "description": "A compact head-to-head attacking duel.",
    },
    {
        "id": "5_vs_5",
        "label": "5v5",
        "description": "A small-sided match with more realistic spacing and pressure.",
    },
    {
        "id": "11_vs_11_easy_stochastic",
        "label": "11v11 easy",
        "description": "The easier full-match benchmark scenario.",
    },
]
SCENARIO_BY_ID = {item["id"]: item for item in SCENARIO_OPTIONS}
DEFAULT_SCENARIO = "academy_run_to_score_with_keeper"

OPPONENT_OPTIONS = [
    {
        "id": "scenario_default",
        "label": "Scenario default",
        "description": "Use the scenario's native keeper, chasers, and match AI.",
        "extra_players": [],
    },
    {
        "id": "built_in_ai",
        "label": "Built-in AI",
        "description": "Add the Google Research Football bot as a right-side AI opponent.",
        "extra_players": ["bot:right_players=1"],
    },
]
OPPONENT_BY_ID = {item["id"]: item for item in OPPONENT_OPTIONS}
DEFAULT_OPPONENT = "built_in_ai"

CONTROL_MODE_OPTIONS = [
    {
        "id": "single",
        "label": "Single",
        "description": "Gemma controls one active yellow player per tick.",
    },
    {
        "id": "squad",
        "label": "Squad",
        "description": "Run coordinated strategy and play phases across controlled yellow players.",
    },
]
CONTROL_MODE_BY_ID = {item["id"]: item for item in CONTROL_MODE_OPTIONS}
DEFAULT_CONTROL_MODE = "single"
SCENARIO_SQUAD_PLAYERS = {
    "academy_run_to_score_with_keeper": 1,
    "academy_pass_and_shoot_with_keeper": 2,
    "academy_3_vs_1_with_keeper": 3,
    "academy_counterattack_easy": 4,
    "1_vs_1_easy": 1,
    "5_vs_5": 4,
    "11_vs_11_easy_stochastic": 4,
}
MAX_SQUAD_PLAYERS = 4

CONTROLLER_OPTIONS = [
    {"id": "cerebras", "label": "Cerebras", "kind": "remote", "model_label": "Gemma 4 31B"},
    {"id": "gpu", "label": "GPU", "kind": "remote", "model_label": "Gemma 4 31B"},
]
LOCAL_CONTROLLER = {"id": "local", "label": "Local scout", "kind": "local", "model_label": "Deterministic fallback"}
CONTROLLER_BY_ID = {item["id"]: item for item in CONTROLLER_OPTIONS + [LOCAL_CONTROLLER]}
DEFAULT_CONTROLLER = "cerebras"

PROVIDER_TIMEOUT_SECONDS = 90
_provider_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gemma-4-31b")
GPU_MODEL = os.getenv("GPU_MODEL", "google/gemma-4-31B-it")
GPU_REASONING_EFFORT = os.getenv("GPU_REASONING_EFFORT", "none").strip().lower()
GPU_STRATEGIZE_REASONING_EFFORT = os.getenv("GPU_STRATEGIZE_REASONING_EFFORT", "medium").strip().lower()
GPU_PLAY_TOOL_REASONING_EFFORT = os.getenv("GPU_PLAY_TOOL_REASONING_EFFORT", "low").strip().lower()
CEREBRAS_REASONING_EFFORT = os.getenv("CEREBRAS_REASONING_EFFORT", "none").strip().lower()
CEREBRAS_STRATEGIZE_REASONING_EFFORT = os.getenv("CEREBRAS_STRATEGIZE_REASONING_EFFORT", "medium").strip().lower()
CEREBRAS_PLAY_TOOL_REASONING_EFFORT = os.getenv("CEREBRAS_PLAY_TOOL_REASONING_EFFORT", "low").strip().lower()
CEREBRAS_DISABLE_REASONING = os.getenv("CEREBRAS_DISABLE_REASONING", "").strip().lower() in {"1", "true", "yes", "on"}
CEREBRAS_PROMPT_CACHE_KEY = os.getenv("CEREBRAS_PROMPT_CACHE_KEY", "gemmafc-live-control-v3-keeper-dribble").strip()
CEREBRAS_USE_ACTION_TOOL = os.getenv("CEREBRAS_USE_ACTION_TOOL", "1").strip().lower() not in {"0", "false", "no", "off"}
CEREBRAS_STRATEGY_MAX_TOKENS = int(os.getenv("CEREBRAS_STRATEGY_MAX_TOKENS", "1600"))
CEREBRAS_PLAY_TOOL_MAX_TOKENS = int(os.getenv("CEREBRAS_PLAY_TOOL_MAX_TOKENS", "1600"))
CEREBRAS_DEFAULT_MAX_TOKENS = int(os.getenv("CEREBRAS_DEFAULT_MAX_TOKENS", "1600"))
GPU_MAX_TOKENS = int(os.getenv("GPU_MAX_TOKENS", "1600"))
VALID_REASONING_EFFORTS = {"none", "low", "medium", "high"}

GAME_MODE_LABELS = {
    0: "normal play",
    1: "kickoff",
    2: "goal kick",
    3: "free kick",
    4: "corner",
    5: "throw in",
    6: "penalty",
}
STICKY_ACTION_LABELS = [
    "left",
    "top_left",
    "top",
    "top_right",
    "right",
    "bottom_right",
    "bottom",
    "bottom_left",
    "sprint",
    "dribble",
    "keeper_rush",
    "pressure",
    "team_pressure",
]
CONTROLLER_OBJECTIVE = (
    "Control the yellow-jersey left team. Maximize the football outcome: score when a real chance exists, "
    "create better chances with movement or passing, preserve or regain possession, and defend when out of possession. "
    "Choose exactly one legal next action from the catalog for the assigned yellow player. "
    "No action is privileged by the app prompt or evaluator."
)
FOOTBALL_RULES_PROMPT = (
    "Football rules and field orientation: Gemma controls the yellow jerseys. The opponent wears non-yellow kits. "
    "The objective is to score more goals by moving the ball wholly over the opponent goal line between the posts and under the crossbar. "
    "The halfway line divides own half from attacking half; the yellow team attacks toward positive x / the right-side goal in GRF coordinates. "
    "Keep possession: recover loose balls first, and do not leave a controlled ball behind just to change lanes. "
    "Use passes or dribbles to progress, shoot when a credible chance exists, and defend or press when the opponent has possession. "
    "Only choose actions from the provided Google Research Football action catalog."
)
SCORING_STRATEGY_PROMPT = (
    "Strategy checklist: 1) Read the image first: ball, yellow spacing, defenders, keeper, goal, and open lanes. "
    "2) Ball localization guardrail: the fixed white penalty spot/dot in the penalty box is a field marking, not the ball. "
    "Identify the real ball by nearby player attention, possession, motion context, and ball coordinates in STATE_JSON; ignore a static white dot with no player interaction. "
    "3) Possession first: if yellow does not control the ball, plan recovery, support, or pressure before any scoring move. "
    "4) When the ball carrier has space, keep close control: in GRF Dribble is sticky and effective only with the ball; avoid Sprint when control matters. "
    "5) Use width, depth, support passes, or high/long passes to create a better lane instead of forcing a dribble into pressure. "
    "6) Shoot only from a credible lane; otherwise improve angle, pass, or keep possession."
)
GOALKEEPER_STRATEGY_PROMPT = (
    "Keeper strategy: if the ball carrier has possession and the keeper blocks the direct shot, keep or start Dribble, "
    "angle top/bottom while staying with the ball, then shoot only after the lane opens. "
    "If the goalkeeper is rushing at the ball carrier, do not shoot or sprint directly into the keeper; keep close control with Dribble "
    "and maneuver around the keeper with the open top/bottom or diagonal lane before shooting or passing. "
    "If possession is loose, recover/control first."
)

ACTIONS = {
    0: {"name": "Idle", "intent": "Hold position and avoid changing the current play state"},
    1: {"name": "Left", "intent": "Move toward the left side of the screen"},
    2: {"name": "Top left", "intent": "Move diagonally toward the upper-left channel"},
    3: {"name": "Top", "intent": "Move toward the top touchline"},
    4: {"name": "Top right", "intent": "Move diagonally toward the upper-right channel"},
    5: {"name": "Right", "intent": "Carry or move toward the attacking goal"},
    6: {"name": "Bottom right", "intent": "Move diagonally toward the lower-right channel"},
    7: {"name": "Bottom", "intent": "Move toward the bottom touchline"},
    8: {"name": "Bottom left", "intent": "Move diagonally toward the lower-left channel"},
    9: {"name": "Long pass", "intent": "Play a driven pass into space or toward a distant teammate"},
    10: {"name": "High pass", "intent": "Lift the ball over pressure; use as a cross-like action near wide areas"},
    11: {"name": "Short pass", "intent": "Move the ball to a nearby teammate"},
    12: {"name": "Shot", "intent": "Shoot at goal when in a realistic finishing position"},
    13: {"name": "Sprint", "intent": "Start sprinting; sticky speed boost with worse ball control"},
    14: {"name": "Release direction", "intent": "Stop the current sticky movement direction"},
    15: {"name": "Release sprint", "intent": "Stop sprinting to regain control"},
    16: {"name": "Sliding", "intent": "Attempt a slide tackle when defending"},
    17: {"name": "Dribble", "intent": "Start close-control dribbling; sticky action"},
    18: {"name": "Release dribble", "intent": "Stop close-control dribbling"},
    19: {"name": "Built-in AI", "intent": "Delegate this tick to the game engine's built-in controller"},
    20: {"name": "Keeper rush", "intent": "Bring the goalkeeper out to challenge the ball"},
    21: {"name": "Pressure", "intent": "Press the opponent ball carrier"},
    22: {"name": "Team pressure", "intent": "Ask teammates to collectively press"},
    23: {"name": "Switch", "intent": "Switch active controlled player while defending"},
    24: {"name": "Release long pass", "intent": "Release a held long-pass action"},
    25: {"name": "Release high pass", "intent": "Release a held high-pass action"},
    26: {"name": "Release short pass", "intent": "Release a held short-pass action"},
    27: {"name": "Release shot", "intent": "Release a held shot action"},
    28: {"name": "Release keeper rush", "intent": "Stop goalkeeper rush"},
    29: {"name": "Release sliding", "intent": "Release slide-tackle action"},
    30: {"name": "Release pressure", "intent": "Stop pressing the ball carrier"},
    31: {"name": "Release team pressure", "intent": "Stop team pressure"},
    32: {"name": "Release switch", "intent": "Release player-switch action"},
}
MODEL_ACTION_IDS = tuple(action_id for action_id in sorted(ACTIONS) if action_id != 19)


@dataclass
class ActionDecision:
    action_id: int
    name: str
    intent: str
    score: float
    agent_index: int = 0
    repaired: bool = False
    repair_reason: str = ""
    candidate_id: Optional[int] = None
    candidate_name: str = ""
    provider_id: str = "local"
    provider_label: str = "Local scout"
    provider_latency_ms: Optional[float] = None
    first_token_ms: Optional[float] = None
    raw_response: str = ""


@dataclass
class ObservationPhase:
    frame_view: np.ndarray
    frame_stats: Dict[str, float]
    frame_image_url: Optional[str]
    ms: float


@dataclass
class StrategyPhase:
    plan: Dict[str, Any]
    detail: str
    provider_result: Optional[Dict[str, Any]]
    ms: float


@dataclass
class PlayPhase:
    candidates: List[ActionDecision]
    detail: str
    provider_result: Optional[Dict[str, Any]]
    ms: float


@dataclass
class RepairPhase:
    decisions: List[ActionDecision]
    ms: float


@dataclass
class ActPhase:
    reward: float
    ms: float


class FootballSession:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.env = None
        self.scenario = DEFAULT_SCENARIO
        self.opponent_mode = DEFAULT_OPPONENT
        self.frame: Optional[np.ndarray] = None
        self.step_count = 0
        self.total_reward = 0.0
        self.repairs = 0
        self.done = False
        self.last_action: Optional[ActionDecision] = None
        self.last_actions: List[ActionDecision] = []
        self.last_provider_result: Optional[Dict[str, Any]] = None
        self.last_provider_results: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.pending_decision_future: Optional[concurrent.futures.Future] = None
        self.pending_decision_key: Optional[Tuple[Any, ...]] = None
        self.control_mode = DEFAULT_CONTROL_MODE
        self.controlled_players = 1
        self.controller_mode = DEFAULT_CONTROLLER
        self.latencies: List[float] = []
        self.started_at: Optional[float] = None
        self.render_mode = "live"

    def reset(self, controller: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if controller is not None:
                self.controller_mode = self._validate_controller(controller)
            self.controlled_players = self._controlled_player_count()
            self._clear_remote_decision_prefetch_unlocked()
            self._close_env()
            self.env = football_env.create_environment(
                env_name=self.scenario,
                representation="pixels",
                render=True,
                channel_dimensions=(FRAME_WIDTH, FRAME_HEIGHT),
                number_of_left_players_agent_controls=self.controlled_players,
                number_of_right_players_agent_controls=0,
                extra_players=self._opponent_extra_players(),
                other_config_options={"action_set": "full"},
            )
            self.frame = self.env.reset()
            self.step_count = 0
            self.total_reward = 0.0
            self.repairs = 0
            self.done = False
            self.last_action = None
            self.last_actions = []
            self.last_provider_result = None
            self.last_provider_results = []
            self.action_history = []
            self._clear_remote_decision_prefetch_unlocked()
            self.latencies = []
            self.started_at = time.perf_counter()
            return self._snapshot(loop=self._idle_loop(), reward=0.0, latency_ms=0.0)

    def step(self, controller: str = DEFAULT_CONTROLLER) -> Dict[str, Any]:
        with self._lock:
            if self.env is None or self.frame is None or self.done:
                self.controller_mode = self._validate_controller(controller)
                return self.reset()

            controller = self._validate_controller(controller)
            self.controller_mode = controller
            self.last_provider_result = None
            self.last_provider_results = []

            total_start = time.perf_counter()
            observation, strategy, play, repair, act = self._run_control_pipeline(controller)
            total_ms = self._record_tick(strategy.plan, repair.decisions, act.reward, total_start)
            self._start_remote_decision_prefetch_unlocked(controller)
            loop = self._phase_loop(observation, strategy, play, repair, act)
            return self._snapshot(loop=loop, reward=act.reward, latency_ms=total_ms)

    def _run_control_pipeline(
        self,
        controller: str,
    ) -> Tuple[ObservationPhase, StrategyPhase, PlayPhase, RepairPhase, ActPhase]:
        observation = self._observe_phase(controller)
        if controller == "local":
            strategy = self._strategize_phase(controller, observation)
            play = self._play_phase(controller, observation, strategy)
        else:
            strategy, play = self._remote_decision_phases(controller, observation)
        repair = self._repair_phase(observation, play)
        act = self._act_phase(repair)
        return observation, strategy, play, repair, act

    def _observe_phase(self, controller: str) -> ObservationPhase:
        started = time.perf_counter()
        assert self.frame is not None
        frame_view = self._display_frame(self.frame)
        frame_stats = self._frame_stats(frame_view)
        frame_image_url = None if controller == "local" else self._encode_model_frame(frame_view)
        return ObservationPhase(
            frame_view=frame_view,
            frame_stats=frame_stats,
            frame_image_url=frame_image_url,
            ms=self._elapsed_ms(started),
        )

    def _strategize_phase(self, controller: str, observation: ObservationPhase) -> StrategyPhase:
        started = time.perf_counter()
        if controller == "local":
            plan = self._local_strategy_unlocked()
            return StrategyPhase(
                plan=plan,
                detail=self._strategy_detail(plan),
                provider_result=None,
                ms=self._elapsed_ms(started),
            )

        context = self._provider_context_unlocked(observation.frame_stats, 0)
        context["phase"] = "strategize"
        provider_result = _call_provider_bounded(controller, context, observation.frame_image_url, task="strategy")
        self.last_provider_results.append(provider_result)
        if provider_result["status"] != "ok":
            error = provider_result.get("error") or "No strategy response content."
            raise HTTPException(
                status_code=502,
                detail=f"{self._controller_label(controller)} strategy failed: {error}",
            )
        plan = self._strategy_from_provider(provider_result)
        return StrategyPhase(
            plan=plan,
            detail=self._provider_strategy_detail(plan, provider_result),
            provider_result=provider_result,
            ms=self._elapsed_ms(started),
        )

    def _play_phase(self, controller: str, observation: ObservationPhase, strategy: StrategyPhase) -> PlayPhase:
        started = time.perf_counter()
        if controller == "local":
            candidates = [
                self._propose_action(observation.frame_stats, agent_index)
                for agent_index in range(self.controlled_players)
            ]
            return PlayPhase(
                candidates=candidates,
                detail=self._squad_action_label(candidates),
                provider_result=None,
                ms=self._elapsed_ms(started),
            )

        context = self._provider_context_unlocked(observation.frame_stats, 0)
        context["phase"] = "play"
        context["strategy_plan"] = strategy.plan
        provider_result = _call_provider_bounded(controller, context, observation.frame_image_url, task="play")
        self.last_provider_results.append(provider_result)
        self.last_provider_result = provider_result
        if provider_result["status"] != "ok":
            error = provider_result.get("error") or "No play response content."
            raise HTTPException(
                status_code=502,
                detail=f"{self._controller_label(controller)} play failed: {error}",
            )
        candidates = self._decisions_from_play_provider(controller, provider_result, self.controlled_players)
        return PlayPhase(
            candidates=candidates,
            detail=self._squad_action_label(candidates),
            provider_result=provider_result,
            ms=self._elapsed_ms(started),
        )

    def _remote_decision_phases(self, controller: str, observation: ObservationPhase) -> Tuple[StrategyPhase, PlayPhase]:
        started = time.perf_counter()
        provider_result = self._consume_remote_decision_prefetch_unlocked(controller)
        if provider_result is None:
            context = self._compact_provider_context_unlocked(observation.frame_stats)
            context["phase"] = "decision"
            provider_result = _call_provider_bounded(controller, context, observation.frame_image_url, task="decision")
        self.last_provider_results.append(provider_result)
        self.last_provider_result = provider_result
        if provider_result["status"] != "ok":
            error = provider_result.get("error") or "No decision response content."
            raise HTTPException(
                status_code=502,
                detail=f"{self._controller_label(controller)} decision failed: {error}",
            )

        plan = self._strategy_from_provider(provider_result)
        candidates = self._decisions_from_play_provider(controller, provider_result, self.controlled_players)
        decision_ms = float(provider_result.get("prefetch_wait_ms") or self._elapsed_ms(started))
        provider_result["visible_latency_ms"] = round(decision_ms, 2)
        return (
            StrategyPhase(
                plan=plan,
                detail=self._provider_strategy_detail(plan, provider_result),
                provider_result=provider_result,
                ms=decision_ms,
            ),
            PlayPhase(
                candidates=candidates,
                detail=f"{self._squad_action_label(candidates)} from play schema after strategy",
                provider_result=provider_result,
                ms=0.0,
            ),
        )

    def _decision_prefetch_key(self, controller: str) -> Tuple[Any, ...]:
        return (
            controller,
            self.scenario,
            self.opponent_mode,
            self.control_mode,
            self.controlled_players,
            self.step_count,
            self.done,
        )

    def _clear_remote_decision_prefetch_unlocked(self) -> None:
        if self.pending_decision_future is not None:
            self.pending_decision_future.cancel()
        self.pending_decision_future = None
        self.pending_decision_key = None

    def _start_remote_decision_prefetch_unlocked(self, controller: str) -> None:
        if controller == "local" or self.done or self.frame is None:
            self._clear_remote_decision_prefetch_unlocked()
            return

        key = self._decision_prefetch_key(controller)
        if self.pending_decision_key == key and self.pending_decision_future is not None:
            return

        frame_view = self._display_frame(self.frame)
        context = self._compact_provider_context_unlocked(self._frame_stats(frame_view))
        context["phase"] = "decision"
        frame_image_url = self._encode_model_frame(frame_view)
        self.pending_decision_key = key
        self.pending_decision_future = _provider_executor.submit(
            _call_provider,
            controller,
            context,
            frame_image_url,
            "decision",
        )

    def _consume_remote_decision_prefetch_unlocked(self, controller: str) -> Optional[Dict[str, Any]]:
        key = self._decision_prefetch_key(controller)
        future = self.pending_decision_future
        if future is None:
            return None
        if self.pending_decision_key != key:
            self._clear_remote_decision_prefetch_unlocked()
            return None

        started = time.perf_counter()
        try:
            provider_result = future.result(timeout=PROVIDER_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            provider = CONTROLLER_BY_ID[controller]
            provider_result = {
                "id": controller,
                "label": provider["label"],
                "model_label": provider["model_label"],
                "configured": True,
                "status": "timeout",
                "content": "",
                "parsed": None,
                "latency_ms": PROVIDER_TIMEOUT_SECONDS * 1000,
                "first_token_ms": None,
                "usage": None,
                "reasoning_effort": _cerebras_reasoning_effort("decision") if controller == "cerebras" else None,
                "prompt_cache_key": CEREBRAS_PROMPT_CACHE_KEY if controller == "cerebras" else None,
                "tool": "execute_play" if controller == "cerebras" and CEREBRAS_USE_ACTION_TOOL else None,
                "tool_call_used": False,
                "error": f"Prepared decision did not return within {PROVIDER_TIMEOUT_SECONDS} seconds.",
            }
        except Exception as exc:
            provider = CONTROLLER_BY_ID[controller]
            provider_result = {
                "id": controller,
                "label": provider["label"],
                "model_label": provider["model_label"],
                "configured": True,
                "status": "error",
                "content": "",
                "parsed": None,
                "latency_ms": self._elapsed_ms(started),
                "first_token_ms": None,
                "usage": None,
                "reasoning_effort": _cerebras_reasoning_effort("decision") if controller == "cerebras" else None,
                "prompt_cache_key": CEREBRAS_PROMPT_CACHE_KEY if controller == "cerebras" else None,
                "tool": "execute_play" if controller == "cerebras" and CEREBRAS_USE_ACTION_TOOL else None,
                "tool_call_used": False,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        finally:
            self.pending_decision_future = None
            self.pending_decision_key = None

        provider_result["prefetched"] = True
        provider_result["prefetch_wait_ms"] = self._elapsed_ms(started)
        return provider_result

    def _repair_phase(self, observation: ObservationPhase, play: PlayPhase) -> RepairPhase:
        started = time.perf_counter()
        decisions = [self._evaluate_and_repair(candidate, observation.frame_stats) for candidate in play.candidates]
        return RepairPhase(decisions=decisions, ms=self._elapsed_ms(started))

    def _act_phase(self, repair: RepairPhase) -> ActPhase:
        if not repair.decisions:
            raise HTTPException(status_code=502, detail="No playable action was produced.")

        action_payload: Any = repair.decisions[0].action_id
        if self.controlled_players > 1:
            action_payload = [decision.action_id for decision in repair.decisions]

        started = time.perf_counter()
        assert self.env is not None
        self.frame, reward, self.done, _info = self.env.step(action_payload)
        return ActPhase(reward=self._reward_scalar(reward), ms=self._elapsed_ms(started))

    def _record_tick(
        self,
        strategy: Dict[str, Any],
        decisions: List[ActionDecision],
        reward_value: float,
        total_start: float,
    ) -> float:
        self.step_count += 1
        self.total_reward += reward_value
        self.last_actions = decisions
        self.last_action = decisions[0] if decisions else None
        self.action_history.append({
            "step": self.step_count,
            "strategy": strategy.get("summary", ""),
            "actions": [
                {"agent_index": decision.agent_index, "id": decision.action_id, "name": decision.name}
                for decision in decisions
            ],
        })
        self.action_history = self.action_history[-12:]
        if any(decision.repaired for decision in decisions):
            self.repairs += 1

        total_ms = self._elapsed_ms(total_start)
        self.latencies.append(total_ms)
        self.latencies = self.latencies[-120:]
        return total_ms

    def _phase_loop(
        self,
        observation: ObservationPhase,
        strategy: StrategyPhase,
        play: PlayPhase,
        repair: RepairPhase,
        act: ActPhase,
    ) -> List[Dict[str, Any]]:
        return [
            {"name": "Observe", "status": "done", "ms": observation.ms, "detail": "Image frame and raw state sampled"},
            {"name": "Strategize", "status": "done", "ms": strategy.ms, "detail": strategy.detail},
            {"name": "Play", "status": "done", "ms": play.ms, "detail": play.detail},
            {
                "name": "Repair",
                "status": "done" if any(decision.repaired for decision in repair.decisions) else "skipped",
                "ms": repair.ms,
                "detail": self._repair_detail(repair.decisions),
            },
            {"name": "Act", "status": "done", "ms": act.ms, "detail": self._squad_action_label(repair.decisions)},
        ]

    def state(self) -> Dict[str, Any]:
        with self._lock:
            if self.frame is None:
                raise HTTPException(status_code=409, detail="Environment has not been reset yet.")
            return self._snapshot(loop=self._idle_loop(), reward=0.0, latency_ms=0.0)

    def provider_context(self) -> Dict[str, Any]:
        with self._lock:
            if self.frame is None:
                return {
                    "scenario": self.scenario,
                    "scenario_label": self._scenario_label(),
                    "opponent_mode": self.opponent_mode,
                    "opponent_label": self._opponent_label(),
                    "controller": self.controller_mode,
                    "controller_label": self._controller_label(self.controller_mode),
                    "control_mode": self.control_mode,
                    "controlled_players": self.controlled_players,
                    "step": self.step_count,
                    "status": "not_started",
                    "objective": CONTROLLER_OBJECTIVE,
                    "football_rules": FOOTBALL_RULES_PROMPT,
                    "scoring_strategy": SCORING_STRATEGY_PROMPT,
                    "allowed_actions": _action_catalog(),
                }

            frame_view = self._display_frame(self.frame)
            return {
                "scenario": self.scenario,
                "scenario_label": self._scenario_label(),
                "opponent_mode": self.opponent_mode,
                "opponent_label": self._opponent_label(),
                "controller": self.controller_mode,
                "controller_label": self._controller_label(self.controller_mode),
                "control_mode": self.control_mode,
                "controlled_players": self.controlled_players,
                "step": self.step_count,
                "done": self.done,
                "total_reward": round(self.total_reward, 4),
                "repairs": self.repairs,
                "render_mode": self.render_mode,
                "frame_size": {"width": int(frame_view.shape[1]), "height": int(frame_view.shape[0])},
                "frame_stats": self._frame_stats(frame_view),
                "objective": CONTROLLER_OBJECTIVE,
                "football_rules": FOOTBALL_RULES_PROMPT,
                "scoring_strategy": SCORING_STRATEGY_PROMPT,
                "tactical_state": self._tactical_state_unlocked(0),
                "squad_tactical_state": self._squad_tactical_state_unlocked(),
                "last_action": None if self.last_action is None else self._public_action(self.last_action),
                "squad_actions": [self._public_action(action) for action in self.last_actions],
                "recent_actions": self.action_history[-6:],
                "allowed_actions": _action_catalog(),
            }

    def provider_payload(self) -> Dict[str, Any]:
        with self._lock:
            if self.frame is None:
                return {"context": self.provider_context(), "frame_image_url": None}
            frame_view = self._display_frame(self.frame)
            return {
                "context": self._compact_provider_context_unlocked(self._frame_stats(frame_view)),
                "frame_image_url": self._encode_model_frame(frame_view),
            }

    def set_render_mode(self, mode: str) -> Dict[str, Any]:
        with self._lock:
            if mode not in RENDER_MODES:
                raise HTTPException(status_code=400, detail=f"Unsupported render mode: {mode}")
            self.render_mode = mode
            if self.frame is None:
                return self._render_mode_payload()
            return self._snapshot(loop=self._idle_loop(), reward=0.0, latency_ms=0.0)

    def _render_mode_payload(self) -> Dict[str, Any]:
        return {
            "render_mode": self.render_mode,
            "render_mode_label": RENDER_MODES[self.render_mode]["label"],
            "available_render_modes": [
                {"id": key, "label": value["label"], "width": value["width"], "height": value["height"]}
                for key, value in RENDER_MODES.items()
            ],
        }

    def options(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "scenarios": SCENARIO_OPTIONS,
                "opponents": [self._public_opponent(option) for option in OPPONENT_OPTIONS],
                "controllers": _public_controllers(),
                "control_modes": CONTROL_MODE_OPTIONS,
                "selected_control_mode": self.control_mode,
                "controlled_players": self.controlled_players,
                "squad_player_limit": self._squad_player_limit(),
                "selected_controller": self.controller_mode,
                "selected_scenario": self.scenario,
                "selected_opponent": self.opponent_mode,
                "scenario_label": self._scenario_label(),
                "opponent_label": self._opponent_label(),
            }

    def configure(self, scenario: str, opponent_mode: str) -> Dict[str, Any]:
        with self._lock:
            if scenario not in SCENARIO_BY_ID:
                raise HTTPException(status_code=400, detail=f"Unsupported scenario: {scenario}")
            if opponent_mode not in OPPONENT_BY_ID:
                raise HTTPException(status_code=400, detail=f"Unsupported opponent mode: {opponent_mode}")
            self.scenario = scenario
            self.opponent_mode = opponent_mode
            return self.reset()

    def set_control_mode(self, mode: str) -> Dict[str, Any]:
        with self._lock:
            if mode not in CONTROL_MODE_BY_ID:
                raise HTTPException(status_code=400, detail=f"Unsupported control mode: {mode}")
            self.control_mode = mode
            return self.reset()

    def _public_opponent(self, option: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": option["id"],
            "label": option["label"],
            "description": option["description"],
        }

    def _opponent_extra_players(self) -> List[str]:
        return list(OPPONENT_BY_ID[self.opponent_mode]["extra_players"])

    def _scenario_label(self) -> str:
        return SCENARIO_BY_ID.get(self.scenario, {}).get("label", self.scenario)

    def _opponent_label(self) -> str:
        return OPPONENT_BY_ID.get(self.opponent_mode, {}).get("label", self.opponent_mode)

    def _reward_scalar(self, reward: Any) -> float:
        if isinstance(reward, np.ndarray):
            return float(np.sum(reward))
        if isinstance(reward, (list, tuple)):
            return float(sum(reward))
        return float(reward)

    def _validate_controller(self, controller: str) -> str:
        if controller not in CONTROLLER_BY_ID:
            raise HTTPException(status_code=400, detail=f"Unsupported controller: {controller}")
        return controller

    def _controller_label(self, controller: str) -> str:
        return CONTROLLER_BY_ID.get(controller, {}).get("label", controller)

    def _provider_context_unlocked(self, stats: Dict[str, float], agent_index: int = 0) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "scenario_label": self._scenario_label(),
            "opponent_mode": self.opponent_mode,
            "opponent_label": self._opponent_label(),
            "active_controller": self.controller_mode,
            "active_controller_label": self._controller_label(self.controller_mode),
            "available_controllers": _public_controllers(),
            "control_mode": self.control_mode,
            "controlled_players": self.controlled_players,
            "agent_index": agent_index,
            "agent_label": f"yellow player {agent_index + 1}",
            "teammate_agent_indices": [index for index in range(self.controlled_players) if index != agent_index],
            "step": self.step_count,
            "done": self.done,
            "total_reward": round(self.total_reward, 4),
            "repairs": self.repairs,
            "objective": CONTROLLER_OBJECTIVE,
            "football_rules": FOOTBALL_RULES_PROMPT,
            "scoring_strategy": SCORING_STRATEGY_PROMPT,
            "image_first_instruction": (
                "First deduce the current football situation from the image snapshot: yellow players, ball, nearest opponents, "
                "goal direction, field zone, and open passing/shooting lanes. Then use tactical_state to verify coordinates, "
                "possession, sticky actions, and nearby players."
            ),
            "frame_stats": stats,
            "vision_frame": {"width": int(self._display_frame(self.frame).shape[1]), "height": int(self._display_frame(self.frame).shape[0]), "mime": "image/jpeg"} if self.frame is not None else None,
            "tactical_state": self._tactical_state_unlocked(agent_index),
            "squad_tactical_state": self._squad_tactical_state_unlocked(),
            "last_action": None if self.last_action is None else self._public_action(self.last_action),
            "squad_actions": [self._public_action(action) for action in self.last_actions],
            "recent_actions": self.action_history[-6:],
            "allowed_actions": _action_catalog(),
        }

    def _compact_provider_context_unlocked(self, stats: Dict[str, float]) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "scenario_label": self._scenario_label(),
            "opponent": self.opponent_mode,
            "opponent_label": self._opponent_label(),
            "control_mode": self.control_mode,
            "controlled_players": self.controlled_players,
            "step": self.step_count,
            "done": self.done,
            "total_reward": round(self.total_reward, 3),
            "repairs": self.repairs,
            "orientation": "yellow team attacks +x / right-side goal",
            "frame": {
                "width": MODEL_FRAME_WIDTH,
                "height": MODEL_FRAME_HEIGHT,
                "mime": "image/jpeg",
                "mean": round(float(stats.get("mean", 0.0)), 2),
            },
            "players_state": [
                self._compact_tactical_state_unlocked(index)
                for index in range(self.controlled_players)
            ],
            "last_actions": [self._compact_action(action) for action in self.last_actions],
            "recent_actions": self.action_history[-3:],
        }

    def _compact_tactical_state_unlocked(self, agent_index: int) -> Dict[str, Any]:
        raw = self._tactical_state_unlocked(agent_index)
        if not raw.get("available"):
            return {"agent_index": agent_index, "available": False}

        possession = raw.get("possession") or {}
        active = raw.get("active_player") or {}
        teams = raw.get("teams") or {}
        sticky = raw.get("sticky_actions") or {}
        game_mode = raw.get("game_mode") or {}
        return {
            "agent_index": agent_index,
            "score": raw.get("score"),
            "game_mode": game_mode.get("label"),
            "possession": {
                "team": possession.get("team"),
                "player": possession.get("player"),
                "player_has_ball": possession.get("controlled_player_has_ball"),
            },
            "ball": raw.get("ball"),
            "active_player": {
                "index": active.get("index"),
                "position": active.get("position"),
                "distance_to_ball": active.get("distance_to_ball"),
                "distance_to_goal": active.get("distance_to_opponent_goal"),
            },
            "nearest": {
                "yellows_to_ball": teams.get("nearest_yellow_players_to_ball"),
                "opponents_to_ball": teams.get("nearest_opponents_to_ball"),
                "teammates_to_active": teams.get("nearest_yellow_teammates_to_active"),
                "opponents_to_active": teams.get("nearest_opponents_to_active"),
            },
            "sticky": sticky.get("active", []),
        }

    def _compact_action(self, action: ActionDecision) -> Dict[str, Any]:
        return {"agent_index": action.agent_index, "id": action.action_id, "name": action.name}

    def _raw_agent_observations_unlocked(self) -> List[Dict[str, Any]]:
        if self.env is None:
            return []
        try:
            raw = self.env.unwrapped.observation()
        except Exception:
            return []
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, (list, tuple)):
            return [item for item in raw if isinstance(item, dict)]
        return []

    def _raw_agent_observation_unlocked(self, agent_index: int = 0) -> Optional[Dict[str, Any]]:
        observations = self._raw_agent_observations_unlocked()
        if not observations:
            return None
        index = min(max(agent_index, 0), len(observations) - 1)
        return observations[index]

    def _squad_tactical_state_unlocked(self) -> List[Dict[str, Any]]:
        return [self._tactical_state_unlocked(index) for index in range(self.controlled_players)]

    def _tactical_state_unlocked(self, agent_index: int = 0) -> Dict[str, Any]:
        obs = self._raw_agent_observation_unlocked(agent_index)
        if not obs:
            return {"available": False, "source": "pixel_frame_only", "agent_index": agent_index}

        left_team = _point_list(obs.get("left_team"))
        right_team = _point_list(obs.get("right_team"))
        ball = _xyz(obs.get("ball"))
        ball_xy = ball[:2] if ball else None
        active = _safe_int(obs.get("active"), -1)
        designated = _safe_int(obs.get("designated"), -1)
        ball_owned_team = _safe_int(obs.get("ball_owned_team"), -1)
        ball_owned_player = _safe_int(obs.get("ball_owned_player"), -1)
        game_mode = _safe_int(obs.get("game_mode"), -1)
        active_pos = left_team[active] if 0 <= active < len(left_team) else None
        goal = [1.0, 0.0]

        return {
            "available": True,
            "source": "env.unwrapped.observation",
            "agent_index": agent_index,
            "team_visual_identity": "yellow jerseys",
            "coordinate_system": {
                "x": "-1 is yellow own goal, +1 is opponent goal",
                "y": "negative/positive touchline axis",
                "halfway_line": "x = 0",
                "yellow_team_attacks": "positive_x",
            },
            "score": _int_list(obs.get("score")),
            "steps_left": _safe_int(obs.get("steps_left"), None),
            "game_mode": {"id": game_mode, "label": GAME_MODE_LABELS.get(game_mode, "unknown")},
            "possession": {
                "team_id": ball_owned_team,
                "team": _team_label(ball_owned_team),
                "player": ball_owned_player,
                "controlled_player_has_ball": ball_owned_team == 0 and ball_owned_player == active,
            },
            "ball": {
                "position": ball,
                "direction": _xyz(obs.get("ball_direction")),
                "distance_to_opponent_goal": _distance(ball_xy, goal),
            },
            "active_player": {
                "index": active,
                "designated_index": designated,
                "position": active_pos,
                "distance_to_ball": _distance(active_pos, ball_xy),
                "distance_to_opponent_goal": _distance(active_pos, goal),
            },
            "teams": {
                "left_count": len(left_team),
                "right_count": len(right_team),
                "nearest_yellow_players_to_ball": _nearest_players(left_team, ball_xy, limit=4),
                "nearest_opponents_to_ball": _nearest_players(right_team, ball_xy, limit=4),
                "nearest_yellow_teammates_to_active": _nearest_players(left_team, active_pos, exclude=active, limit=4),
                "nearest_opponents_to_active": _nearest_players(right_team, active_pos, limit=4),
            },
            "sticky_actions": _sticky_action_summary(obs.get("sticky_actions")),
        }

    def _decision_from_provider(self, controller: str, provider_result: Dict[str, Any], agent_index: int = 0) -> ActionDecision:
        return self._decision_from_parsed(controller, provider_result.get("parsed") or {}, provider_result, agent_index)

    def _decision_from_parsed(
        self,
        controller: str,
        parsed: Dict[str, Any],
        provider_result: Dict[str, Any],
        agent_index: int = 0,
    ) -> ActionDecision:
        action_id = _provider_action_id(parsed)
        score = _provider_confidence(parsed.get("confidence"))
        provider_label = self._controller_label(controller)
        raw_response = str(provider_result.get("content") or "")
        visible_latency_ms = provider_result.get("prefetch_wait_ms") if provider_result.get("prefetched") else provider_result.get("latency_ms")

        if action_id not in ACTIONS or (controller != "local" and action_id == 19):
            action = ACTIONS[0]
            return ActionDecision(
                action_id=0,
                name=action["name"],
                intent=action["intent"],
                score=0.35,
                agent_index=agent_index,
                repaired=True,
                repair_reason=f"Play response returned a missing, unsupported, or delegated action for P{agent_index + 1}; held this player instead.",
                candidate_id=None,
                candidate_name=str(parsed.get("action_name") or parsed.get("action_id") or "Unsupported"),
                provider_id=controller,
                provider_label=provider_label,
                provider_latency_ms=visible_latency_ms,
                first_token_ms=provider_result.get("first_token_ms"),
                raw_response=raw_response,
            )

        action = ACTIONS[action_id]
        rationale = str(parsed.get("rationale") or action["intent"]).strip()[:260]
        return ActionDecision(
            action_id=action_id,
            name=action["name"],
            intent=rationale or action["intent"],
            score=score,
            agent_index=agent_index,
            candidate_id=action_id,
            candidate_name=str(parsed.get("action_name") or action["name"]),
            provider_id=controller,
            provider_label=provider_label,
            provider_latency_ms=visible_latency_ms,
            first_token_ms=provider_result.get("first_token_ms"),
            raw_response=raw_response,
        )

    def _decisions_from_play_provider(
        self,
        controller: str,
        provider_result: Dict[str, Any],
        expected_players: int,
    ) -> List[ActionDecision]:
        parsed = provider_result.get("parsed") or {}
        raw_actions = parsed.get("actions")
        if not isinstance(raw_actions, list):
            first_decision = self._decision_from_provider(controller, provider_result, 0)
            if expected_players <= 1:
                return [first_decision]
            decisions = [first_decision]
            for agent_index in range(1, expected_players):
                decisions.append(self._missing_play_decision(controller, provider_result, agent_index))
            return decisions

        entries: Dict[int, Dict[str, Any]] = {}
        for fallback_index, item in enumerate(raw_actions):
            if not isinstance(item, dict):
                continue
            agent_index = _safe_int(item.get("agent_index"), fallback_index)
            if agent_index is None or agent_index < 0 or agent_index >= expected_players:
                continue
            entries[agent_index] = item

        decisions: List[ActionDecision] = []
        for agent_index in range(expected_players):
            entry = entries.get(agent_index)
            if entry is None:
                decisions.append(self._missing_play_decision(controller, provider_result, agent_index))
            else:
                decisions.append(self._decision_from_parsed(controller, entry, provider_result, agent_index))
        return decisions

    def _missing_play_decision(
        self,
        controller: str,
        provider_result: Dict[str, Any],
        agent_index: int,
    ) -> ActionDecision:
        action = ACTIONS[0]
        return ActionDecision(
            action_id=0,
            name=action["name"],
            intent=action["intent"],
            score=0.35,
            agent_index=agent_index,
            repaired=True,
            repair_reason=f"Play response omitted P{agent_index + 1}; held this player instead.",
            candidate_id=None,
            candidate_name="Missing",
            provider_id=controller,
            provider_label=self._controller_label(controller),
            provider_latency_ms=provider_result.get("prefetch_wait_ms") if provider_result.get("prefetched") else provider_result.get("latency_ms"),
            first_token_ms=provider_result.get("first_token_ms"),
            raw_response=str(provider_result.get("content") or ""),
        )

    def _provider_decide_detail(self, decision: ActionDecision, provider_result: Dict[str, Any]) -> str:
        latency = provider_result.get("latency_ms")
        first = provider_result.get("first_token_ms")
        if first is not None:
            return f"{decision.provider_label} chose {decision.name}; first token {first} ms, total {latency} ms"
        return f"{decision.provider_label} chose {decision.name}; total {latency} ms"

    def _close_env(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None

    def _local_strategy_unlocked(self) -> Dict[str, Any]:
        if self.controlled_players > 1:
            assignments = [
                {"agent_index": 0, "role": "ball progressor", "intent": "carry or pass toward goal"},
            ]
            for agent_index in range(1, self.controlled_players):
                assignments.append({
                    "agent_index": agent_index,
                    "role": "support runner",
                    "intent": "create width, depth, or a passing lane",
                })
            summary = "Coordinate yellow attackers: progress the ball, keep support lanes open, and create a credible shot."
        else:
            assignments = [{"agent_index": 0, "role": "active attacker", "intent": "progress toward a scoring chance"}]
            summary = "Use the active yellow player to progress toward goal and shoot when the chance is credible."
        return {"summary": summary, "assignments": assignments, "risks": ["avoid bunching", "avoid low-value shots"]}

    def _strategy_from_provider(self, provider_result: Dict[str, Any]) -> Dict[str, Any]:
        parsed = provider_result.get("parsed") or {}
        summary = str(
            parsed.get("summary")
            or parsed.get("scoring_plan")
            or parsed.get("plan")
            or provider_result.get("content")
            or "Create a coordinated chance from the current frame."
        ).strip()[:420]
        assignments = parsed.get("assignments")
        if not isinstance(assignments, list):
            assignments = []
        return {
            "summary": summary,
            "phase": parsed.get("phase") or "strategize",
            "assignments": assignments[: self.controlled_players],
            "risks": parsed.get("risks") if isinstance(parsed.get("risks"), list) else [],
        }

    def _strategy_detail(self, strategy: Dict[str, Any]) -> str:
        summary = str(strategy.get("summary") or "Strategy ready").strip()
        return summary[:156] + ("..." if len(summary) > 156 else "")

    def _provider_strategy_detail(self, strategy: Dict[str, Any], provider_result: Dict[str, Any]) -> str:
        latency = provider_result.get("latency_ms")
        detail = self._strategy_detail(strategy)
        if latency is None:
            return detail
        if provider_result.get("prefetched"):
            wait = provider_result.get("prefetch_wait_ms")
            return f"{detail} (prepared after last action; waited {format_provider_ms(wait)}, Gemma {format_provider_ms(latency)})"
        return f"{detail} ({format_provider_ms(latency)})"

    def _propose_action(self, stats: Dict[str, float], agent_index: int = 0) -> ActionDecision:
        cycle = (self.step_count + agent_index * 5) % 32
        if self.controlled_players > 1 and agent_index > 0:
            action_id = [11, 4, 6, 10][(agent_index - 1) % 4]
        elif self.step_count > 0 and cycle == 0:
            action_id = 0
        elif cycle in (1, 2):
            action_id = 13
        elif cycle in (10, 11):
            action_id = 17
        elif cycle in (16, 17):
            action_id = 4
        elif cycle in (23, 24):
            action_id = 12
        else:
            action_id = 5
        action = ACTIONS[action_id]
        return ActionDecision(
            action_id=action_id,
            name=action["name"],
            intent=action["intent"],
            score=0.82,
            agent_index=agent_index,
            candidate_id=action_id,
            candidate_name=action["name"],
            provider_id="local",
            provider_label="Local scout",
        )

    def _evaluate_and_repair(self, candidate: ActionDecision, stats: Dict[str, float]) -> ActionDecision:
        action_id = candidate.action_id
        score = candidate.score
        repair_reason = candidate.repair_reason
        repaired = candidate.repaired

        if action_id not in ACTIONS:
            action_id = 0
            score = min(score, 0.35)
            repaired = True
            repair_reason = "Provider returned an unsupported action; held this player instead."
        elif candidate.provider_id != "local" and action_id == 19:
            action_id = 0
            score = min(score, 0.35)
            repaired = True
            repair_reason = "Provider attempted to delegate to built-in AI; held this player instead."

        action = ACTIONS[action_id]
        return ActionDecision(
            action_id=action_id,
            name=action["name"],
            intent=candidate.intent if action_id == candidate.action_id else action["intent"],
            score=score,
            agent_index=candidate.agent_index,
            repaired=repaired,
            repair_reason=repair_reason,
            candidate_id=candidate.action_id,
            candidate_name=candidate.name,
            provider_id=candidate.provider_id,
            provider_label=candidate.provider_label,
            provider_latency_ms=candidate.provider_latency_ms,
            first_token_ms=candidate.first_token_ms,
            raw_response=candidate.raw_response,
        )

    def _snapshot(self, loop: List[Dict[str, Any]], reward: float, latency_ms: float) -> Dict[str, Any]:
        assert self.frame is not None
        frame_view = self._display_frame(self.frame)
        encoded_size = self._render_frame_size(frame_view)
        avg_latency = statistics.fmean(self.latencies) if self.latencies else latency_ms
        uptime = 0.0 if self.started_at is None else time.perf_counter() - self.started_at
        stats = self._frame_stats(frame_view)
        action = self.last_action
        payload = {
            "scenario": self.scenario,
            "scenario_label": self._scenario_label(),
            "opponent_mode": self.opponent_mode,
            "opponent_label": self._opponent_label(),
            "active_controller": self.controller_mode,
            "active_controller_label": self._controller_label(self.controller_mode),
            "step": self.step_count,
            "done": self.done,
            "frame_url": f"/api/frame.jpg?step={self.step_count}&mode={self.render_mode}&t={round(uptime * 1000)}",
            "frame_size": encoded_size,
            "source_frame_size": {"width": int(frame_view.shape[1]), "height": int(frame_view.shape[0])},
            "render_mode": self.render_mode,
            "render_mode_label": RENDER_MODES[self.render_mode]["label"],
            "control_mode": self.control_mode,
            "control_mode_label": CONTROL_MODE_BY_ID[self.control_mode]["label"],
            "controlled_players": self.controlled_players,
            "squad_player_limit": self._squad_player_limit(),
            "reward": reward,
            "total_reward": self.total_reward,
            "latency_ms": latency_ms,
            "avg_latency_ms": avg_latency,
            "repairs": self.repairs,
            "uptime_s": uptime,
            "frame_stats": stats,
            "tactical_state": self._tactical_state_unlocked(),
            "policy": {
                "provider": self._controller_label(self.controller_mode),
                "model": CONTROLLER_BY_ID[self.controller_mode]["model_label"],
                "mode": "observe -> strategize -> play -> repair -> act",
                "opponent": self._opponent_label(),
            },
            "action": None if action is None else self._public_action(action),
            "squad_actions": [self._public_action(item) for item in self.last_actions],
            "provider_decision": _public_provider_result(self.last_provider_result),
            "loop": loop,
        }
        if self.controlled_players > 1:
            payload["squad_tactical_state"] = self._squad_tactical_state_unlocked()
            payload["recent_actions"] = self.action_history[-6:]
        return payload

    def _display_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 4:
            return frame[0]
        return frame

    def frame_response(self) -> Response:
        with self._lock:
            if self.frame is None:
                raise HTTPException(status_code=409, detail="Environment has not been reset yet.")
            frame_view = self._display_frame(self.frame)
            frame_bytes, _frame_size = self._encode_frame_bytes(frame_view)
        return Response(
            content=frame_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    def _public_action(self, action: ActionDecision) -> Dict[str, Any]:
        return {
            "agent_index": action.agent_index,
            "id": action.action_id,
            "name": action.name,
            "intent": action.intent,
            "score": action.score,
            "repaired": action.repaired,
            "repair_reason": action.repair_reason,
            "candidate_id": action.candidate_id,
            "candidate_name": action.candidate_name,
            "provider_id": action.provider_id,
            "provider_label": action.provider_label,
            "provider_latency_ms": action.provider_latency_ms,
            "first_token_ms": action.first_token_ms,
        }

    def _squad_player_limit(self) -> int:
        if self.scenario in SCENARIO_SQUAD_PLAYERS:
            return max(1, min(MAX_SQUAD_PLAYERS, SCENARIO_SQUAD_PLAYERS[self.scenario]))

        configured = 1
        try:
            scenario_config = football_config.Config({"level": self.scenario}).ScenarioConfig()
            configured = int(getattr(scenario_config, "controllable_left_players", configured) or configured)
        except Exception:
            pass
        return max(1, min(MAX_SQUAD_PLAYERS, configured))

    def _controlled_player_count(self) -> int:
        if self.control_mode != "squad":
            return 1
        return self._squad_player_limit()

    def _squad_action_label(self, decisions: List[ActionDecision]) -> str:
        if not decisions:
            return "No action"
        if len(decisions) == 1:
            return decisions[0].name
        return "; ".join(f"P{decision.agent_index + 1} {decision.name}" for decision in decisions)

    def _repair_detail(self, decisions: List[ActionDecision]) -> str:
        repaired = [decision for decision in decisions if decision.repaired]
        if not repaired:
            return "No correction needed"
        return "; ".join(
            f"P{decision.agent_index + 1}: {decision.repair_reason or 'Corrected'}"
            for decision in repaired
        )

    def _decision_score_label(self, decisions: List[ActionDecision]) -> str:
        if not decisions:
            return "No candidate"
        if any(decision.score < 0.75 for decision in decisions):
            return "Low-confidence action accepted"
        return "Action accepted"

    def _render_frame_size(self, frame: np.ndarray) -> Dict[str, int]:
        mode = RENDER_MODES[self.render_mode]
        if mode["upscale"]:
            return {"width": int(mode["width"]), "height": int(mode["height"])}
        return {"width": int(frame.shape[1]), "height": int(frame.shape[0])}

    def _encode_frame_bytes(self, frame: np.ndarray) -> tuple[bytes, Dict[str, int]]:
        mode = RENDER_MODES[self.render_mode]
        encoded_frame = frame
        if mode["upscale"]:
            encoded_frame = cv2.resize(
                frame,
                (mode["width"], mode["height"]),
                interpolation=cv2.INTER_LANCZOS4,
            )

        bgr = cv2.cvtColor(encoded_frame, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), mode["quality"]])
        if not ok:
            raise RuntimeError("Failed to encode frame")
        frame_size = {"width": int(encoded_frame.shape[1]), "height": int(encoded_frame.shape[0])}
        return buffer.tobytes(), frame_size

    def _encode_frame(self, frame: np.ndarray) -> tuple[str, Dict[str, int], int]:
        frame_bytes, frame_size = self._encode_frame_bytes(frame)
        payload = base64.b64encode(frame_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{payload}", frame_size, int(len(frame_bytes))

    def _encode_model_frame(self, frame: np.ndarray) -> str:
        model_frame = cv2.resize(
            frame,
            (MODEL_FRAME_WIDTH, MODEL_FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        bgr = cv2.cvtColor(model_frame, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), MODEL_JPEG_QUALITY])
        if not ok:
            raise RuntimeError("Failed to encode model frame")
        payload = base64.b64encode(buffer).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def _frame_stats(self, frame: np.ndarray) -> Dict[str, float]:
        return {
            "min": float(np.min(frame)),
            "max": float(np.max(frame)),
            "mean": float(np.mean(frame)),
        }

    def _idle_loop(self) -> List[Dict[str, Any]]:
        return [
            {"name": "Observe", "status": "ready", "ms": 0.0, "detail": "Waiting for next tick"},
            {"name": "Strategize", "status": "ready", "ms": 0.0, "detail": "No plan yet"},
            {"name": "Play", "status": "ready", "ms": 0.0, "detail": "No action candidates"},
            {"name": "Repair", "status": "ready", "ms": 0.0, "detail": "No correction"},
            {"name": "Act", "status": "ready", "ms": 0.0, "detail": "Environment paused"},
        ]

    def _score_label(self, score: float) -> str:
        if score >= 0.75:
            return "Action accepted"
        return "Low-confidence action accepted"

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000.0, 2)


def _public_controllers() -> List[Dict[str, Any]]:
    return [dict(item) for item in CONTROLLER_OPTIONS]


def _action_catalog() -> List[Dict[str, Any]]:
    return [
        {"id": action_id, "name": action["name"], "intent": action["intent"]}
        for action_id, action in sorted(ACTIONS.items())
    ]


def _compact_action_catalog_text() -> str:
    return "; ".join(f"{action_id}:{action['name']}" for action_id, action in sorted(ACTIONS.items()))


def _model_action_catalog_text() -> str:
    return "; ".join(f"{action_id}:{ACTIONS[action_id]['name']}" for action_id in MODEL_ACTION_IDS)


def _assignment_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "agent_index": {"type": "integer"},
            "role": {"type": "string"},
            "target_space": {"type": "string"},
            "instruction": {"type": "string"},
        },
        "required": ["agent_index", "role", "target_space", "instruction"],
    }


def _strategy_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "gemmafc_strategy_plan",
            "description": "Compact GemmaFC plan for the current GRF frame.",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "phase": {"type": "string", "enum": ["strategize"]},
                    "summary": {"type": "string"},
                    "scoring_plan": {"type": "string"},
                    "assignments": {
                        "type": "array",
                        "items": _assignment_schema(),
                    },
                    "risks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["phase", "summary", "scoring_plan", "assignments", "risks"],
            },
        },
    }


def _play_response_format() -> Dict[str, Any]:
    action_id_schema = {"type": "integer", "enum": list(MODEL_ACTION_IDS)}
    action_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "agent_index": {"type": "integer"},
            "action_id": action_id_schema,
            "action_name": {"type": "string"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["agent_index", "action_id", "action_name", "confidence", "rationale"],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "gemmafc_execute_play",
            "description": "Compact action payload for one GemmaFC live control tick.",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "phase": {"type": "string", "enum": ["decision"]},
                    "summary": {"type": "string"},
                    "assignments": {"type": "array", "items": _assignment_schema()},
                    "actions": {"type": "array", "items": action_schema},
                    "action_id": action_id_schema,
                    "action_name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "phase",
                    "summary",
                    "assignments",
                    "actions",
                    "action_id",
                    "action_name",
                    "confidence",
                    "rationale",
                ],
            },
        },
    }


def _execute_play_tool() -> Dict[str, Any]:
    action_id_schema = {"type": "integer", "enum": list(MODEL_ACTION_IDS)}
    action_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "agent_index": {"type": "integer"},
            "action_id": action_id_schema,
            "action_name": {"type": "string"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["agent_index", "action_id", "action_name", "confidence", "rationale"],
    }
    return {
        "type": "function",
        "function": {
            "name": "execute_play",
            "description": "Submit the coordinated GemmaFC scoring plan and legal GRF actions for this live tick.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "phase": {"type": "string", "enum": ["decision"]},
                    "summary": {"type": "string"},
                    "assignments": {
                        "type": "array",
                        "items": _assignment_schema(),
                    },
                    "actions": {
                        "type": "array",
                        "items": action_schema,
                    },
                    "action_id": action_id_schema,
                    "action_name": {"type": "string"},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "phase",
                    "summary",
                    "assignments",
                    "actions",
                    "action_id",
                    "action_name",
                    "confidence",
                    "rationale",
                ],
            },
        },
    }


def _provider_action_id(parsed: Dict[str, Any]) -> Optional[int]:
    raw = parsed.get("action_id")
    try:
        action_id = int(raw)
        if action_id in ACTIONS:
            return action_id
    except (TypeError, ValueError):
        pass

    name = str(parsed.get("action_name") or parsed.get("action") or "").strip().lower().replace("_", " ")
    if not name:
        return None
    for action_id, action in ACTIONS.items():
        if action["name"].lower() == name:
            return action_id
    return None


def _provider_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.72
    return max(0.0, min(1.0, confidence))



def _compact_float(value: Any, digits: int = 3) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _safe_int(value: Any, default: Optional[int] = -1) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int_list(value: Any) -> List[int]:
    if value is None:
        return []
    try:
        return [int(item) for item in list(value)]
    except (TypeError, ValueError):
        return []


def _point_list(value: Any) -> List[List[float]]:
    if value is None:
        return []
    try:
        arr = np.asarray(value, dtype=float).reshape((-1, 2))
    except (TypeError, ValueError):
        return []
    result: List[List[float]] = []
    for point in arr:
        x = _compact_float(point[0])
        y = _compact_float(point[1])
        if x is not None and y is not None:
            result.append([x, y])
    return result


def _xyz(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).flatten()
    except (TypeError, ValueError):
        return None
    if len(arr) < 2:
        return None
    return [_compact_float(item) for item in arr[:3] if _compact_float(item) is not None]


def _distance(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    return _compact_float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def _nearest_players(
    points: List[List[float]],
    target: Optional[List[float]],
    *,
    exclude: Optional[int] = None,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    if not target:
        return []
    ranked = []
    for index, point in enumerate(points):
        if exclude is not None and index == exclude:
            continue
        distance = _distance(point, target)
        if distance is None:
            continue
        ranked.append({"index": index, "position": point, "distance": distance})
    ranked.sort(key=lambda item: item["distance"])
    return ranked[:limit]


def _team_label(team_id: int) -> str:
    if team_id == 0:
        return "left"
    if team_id == 1:
        return "right"
    return "loose"


def _sticky_action_summary(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"active": [], "states": {}}
    try:
        raw_states = [int(item) for item in list(value)]
    except (TypeError, ValueError):
        return {"active": [], "states": {}}
    states = {
        label: bool(raw_states[index])
        for index, label in enumerate(STICKY_ACTION_LABELS)
        if index < len(raw_states)
    }
    return {
        "active": [label for label, active in states.items() if active],
        "states": states,
    }



def format_provider_ms(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if number >= 1000:
        return f"{number / 1000:.2f}s"
    return f"{round(number)} ms"


def _public_provider_result(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return {
        "id": result.get("id"),
        "label": result.get("label"),
        "status": result.get("status"),
        "parsed": result.get("parsed"),
        "latency_ms": result.get("latency_ms"),
        "visible_latency_ms": result.get("visible_latency_ms"),
        "background_latency_ms": result.get("background_latency_ms"),
        "strategy_latency_ms": result.get("strategy_latency_ms"),
        "strategy_first_token_ms": result.get("strategy_first_token_ms"),
        "play_latency_ms": result.get("play_latency_ms"),
        "play_first_token_ms": result.get("play_first_token_ms"),
        "play_first_token_from_start_ms": result.get("play_first_token_from_start_ms"),
        "sdk_create_ms": result.get("sdk_create_ms"),
        "strategy_sdk_create_ms": result.get("strategy_sdk_create_ms"),
        "play_sdk_create_ms": result.get("play_sdk_create_ms"),
        "first_token_ms": result.get("first_token_ms"),
        "error": result.get("error"),
        "usage": result.get("usage"),
        "reasoning_effort": result.get("reasoning_effort"),
        "prompt_cache_key": result.get("prompt_cache_key"),
        "tool": result.get("tool"),
        "tool_call_used": result.get("tool_call_used"),
        "prefetched": result.get("prefetched"),
        "prefetch_wait_ms": result.get("prefetch_wait_ms"),
        "content": result.get("content", "")[:600],
    }


def _call_provider(controller: str, context: Dict[str, Any], frame_image_url: Optional[str] = None, task: str = "play") -> Dict[str, Any]:
    if controller == "cerebras":
        return _call_cerebras_provider(context, frame_image_url, task)
    if controller == "gpu":
        return _call_gpu_provider(context, frame_image_url, task)
    raise HTTPException(status_code=400, detail=f"Unsupported remote controller: {controller}")


def _call_provider_bounded(controller: str, context: Dict[str, Any], frame_image_url: Optional[str] = None, task: str = "play") -> Dict[str, Any]:
    future = _provider_executor.submit(_call_provider, controller, context, frame_image_url, task)
    try:
        return future.result(timeout=PROVIDER_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        provider = CONTROLLER_BY_ID[controller]
        return {
            "id": controller,
            "label": provider["label"],
            "model_label": provider["model_label"],
            "configured": True,
            "status": "timeout",
            "content": "",
            "parsed": None,
            "latency_ms": PROVIDER_TIMEOUT_SECONDS * 1000,
            "first_token_ms": None,
            "usage": None,
                "reasoning_effort": _cerebras_reasoning_effort(task) if controller == "cerebras" else None,
                "prompt_cache_key": CEREBRAS_PROMPT_CACHE_KEY if controller == "cerebras" else None,
                "tool": "execute_play" if controller == "cerebras" and task in {"decision", "play_tool"} and CEREBRAS_USE_ACTION_TOOL else None,
                "tool_call_used": False,
                "error": f"Provider did not return within {PROVIDER_TIMEOUT_SECONDS} seconds.",
            }


def _gpu_api_key() -> Optional[str]:
    return os.getenv("GPU_API_KEY") or os.getenv("TOGETHER_API_KEY")


def _valid_reasoning_effort(value: str, fallback: str) -> str:
    return value if value in VALID_REASONING_EFFORTS else fallback


def _cerebras_reasoning_effort(task: str = "play") -> str:
    if CEREBRAS_DISABLE_REASONING:
        return "none"
    if task in {"strategy", "decision"}:
        return _valid_reasoning_effort(CEREBRAS_STRATEGIZE_REASONING_EFFORT, "medium")
    if task == "play_tool":
        return _valid_reasoning_effort(CEREBRAS_PLAY_TOOL_REASONING_EFFORT, "low")
    return _valid_reasoning_effort(CEREBRAS_REASONING_EFFORT, "none")


def _gpu_reasoning_effort(task: str = "play") -> str:
    if task in {"strategy", "decision"}:
        return _valid_reasoning_effort(GPU_STRATEGIZE_REASONING_EFFORT, "medium")
    if task == "play_tool":
        return _valid_reasoning_effort(GPU_PLAY_TOOL_REASONING_EFFORT, "low")
    return _valid_reasoning_effort(GPU_REASONING_EFFORT, "none")


def provider_status() -> Dict[str, Any]:
    return {
        "providers": [
            {
                "id": "cerebras",
                "label": "Cerebras",
                "model_label": "Gemma 4 31B",
                "configured": bool(os.getenv("CEREBRAS_API_KEY")),
                "reasoning_effort": _cerebras_reasoning_effort("decision"),
                "play_tool_reasoning_effort": _cerebras_reasoning_effort("play_tool"),
                "default_reasoning_effort": _cerebras_reasoning_effort("play"),
                "prompt_cache_key": CEREBRAS_PROMPT_CACHE_KEY or None,
                "action_schema": "gemmafc_execute_play",
            },
            {
                "id": "gpu",
                "label": "GPU",
                "model_label": "Gemma 4 31B",
                "configured": bool(_gpu_api_key()),
                "reasoning_effort": _gpu_reasoning_effort("decision"),
                "play_tool_reasoning_effort": _gpu_reasoning_effort("play_tool"),
                "default_reasoning_effort": _gpu_reasoning_effort("play"),
            },
        ],
        "controllers": _public_controllers(),
        "timeout_seconds": PROVIDER_TIMEOUT_SECONDS,
    }


def _cached_play_system_prompt() -> str:
    return (
        "You are GemmaFC's real-time action caller for Google Research Football. "
        "This system message is intentionally stable across live ticks so the provider can reuse cached prompt tokens. "
        f"{FOOTBALL_RULES_PROMPT} "
        "Use strategy_plan and STATE_JSON; do not invent a new strategy in this phase. "
        "GRF mechanics: directional movement and Dribble are sticky; Idle preserves sticky actions. "
        "Dribble works when the player has the ball, improves close control, and is slower; Sprint is sticky but worsens ball handling. "
        "Never include chain-of-thought, markdown, commentary, or prose outside the required JSON. "
        "Keep all text fields short. "
        "Return compact JSON only with keys: phase, summary, assignments, actions, action_id, action_name, confidence, rationale. "
        "Set phase to decision. summary is a short execution note. "
        "assignments has one item per controlled yellow player: {agent_index, role, target_space, instruction}. "
        "actions has exactly one item per controlled yellow player: {agent_index, action_id, action_name, confidence, rationale}. "
        "Top-level action_id/action_name/confidence/rationale must mirror agent_index 0. "
        "Choose legal action_id values only from this Google Research Football full action catalog: "
        f"{_model_action_catalog_text()}. "
        "Use release actions when sticky movement, sprint, dribble, pressure, keeper rush, or pass/shot charging should stop. "
        "Do not delegate to GRF built-in behavior; action_id 19 is intentionally unavailable to live control."
    )


def _cached_decision_system_prompt() -> str:
    return (
        "You are GemmaFC's real-time strategist and action caller for Google Research Football. "
        f"{FOOTBALL_RULES_PROMPT} "
        f"{SCORING_STRATEGY_PROMPT} "
        f"{GOALKEEPER_STRATEGY_PROMPT} "
        "Use the current image and STATE_JSON to make one concise strategy, assign each controlled yellow player, and choose legal next actions. "
        "GRF mechanics: directional movement and Dribble are sticky; Idle preserves sticky actions. "
        "Dribble works when the player has the ball, improves close control, and is slower; Sprint is sticky but worsens ball handling. "
        "Never include chain-of-thought, markdown, commentary, or prose outside the required JSON. "
        "Keep all text fields short. "
        "Return compact JSON only with keys: phase, summary, assignments, actions, action_id, action_name, confidence, rationale. "
        "Set phase to decision. summary is a short strategy and execution note. "
        "assignments has one item per controlled yellow player: {agent_index, role, target_space, instruction}. "
        "actions has exactly one item per controlled yellow player: {agent_index, action_id, action_name, confidence, rationale}. "
        "Top-level action_id/action_name/confidence/rationale must mirror agent_index 0. "
        "Choose legal action_id values only from this Google Research Football full action catalog: "
        f"{_model_action_catalog_text()}. "
        "Use release actions when sticky movement, sprint, dribble, pressure, keeper rush, or pass/shot charging should stop. "
        "Do not delegate to GRF built-in behavior; action_id 19 is intentionally unavailable to live control."
    )


def build_provider_messages(context: Dict[str, Any], frame_image_url: Optional[str] = None, task: str = "play") -> List[Dict[str, Any]]:
    state_json = json.dumps(context, sort_keys=True, separators=(",", ":"))
    if task == "decision":
        controlled_players = int(context.get("controlled_players") or 1)
        system = _cached_decision_system_prompt()
        user_text = (
            "Live tick input follows. "
            "Set phase to decision. No markdown. "
            f"There are {controlled_players} controlled yellow players, indexed 0 through {controlled_players - 1}. "
            "Read the image, verify with STATE_JSON, summarize the strategy, assign players, and choose from the action catalog. "
            f"STATE_JSON:{state_json}"
        )
    elif task == "play_tool":
        controlled_players = int(context.get("controlled_players") or 1)
        system = _cached_play_system_prompt()
        user_text = (
            "Live play input follows. Use strategy_plan and STATE_JSON to submit this tick's action payload. "
            "Do not include prose, markdown, hidden analysis, or extra keys. "
            "Keep rationale under 18 words and summary under 16 words. "
            f"There are {controlled_players} controlled yellow players, indexed 0 through {controlled_players - 1}. "
            f"STATE_JSON:{state_json}"
        )
    elif task == "strategy":
        system = (
            "You are GemmaFC's squad strategist for Google Research Football. "
            f"{FOOTBALL_RULES_PROMPT} "
            f"{SCORING_STRATEGY_PROMPT} "
            f"{GOALKEEPER_STRATEGY_PROMPT} "
            "Your job is to make a coordinated plan for the yellow team from the current image and state. "
            "Do not choose action ids in this phase."
        )
        user_text = (
            "Return only compact JSON with keys: phase, summary, scoring_plan, assignments, risks. "
            "Set phase to strategize. Do not use markdown, prose, hidden analysis, or extra keys. "
            "Respect the response JSON schema exactly. Keep summary under 16 words and scoring_plan under 24 words. "
            "Image-first: infer yellow shape, ball, opponents, goalkeeper, goal, halfway-line zone, and open lanes from the frame. "
            "Do not confuse the white penalty spot/dot with the ball; confirm the ball by relative player positions, possession, and STATE_JSON ball fields. "
            "Use tactical_state and squad_tactical_state only to verify coordinates, possession, sticky actions, and nearby players. "
            "assignments must be an array with one item per controlled yellow player: "
            "{agent_index, role, target_space, instruction}. "
            "Make the plan concise and directly useful for scoring.\n\n"
            f"STATE_JSON: {state_json}"
        )
    else:
        controlled_players = int(context.get("controlled_players") or 1)
        system = (
            "You are GemmaFC's play caller for Google Research Football. "
            f"{FOOTBALL_RULES_PROMPT} "
            "Execute the supplied strategy_plan by choosing legal next actions for the yellow player agents. "
            "No action is preferred by the app; choose what best fulfills the strategy and current frame."
        )
        user_text = (
            "Return only compact JSON with keys: phase, action_id, action_name, confidence, rationale, actions, repair_check. "
            "Set phase to play. Do not use markdown. "
            "The actions array is required and must contain exactly one item for every controlled yellow player. "
            f"There are {controlled_players} controlled yellow players, indexed 0 through {controlled_players - 1}. "
            "Each actions item must be {agent_index, action_id, action_name, confidence, rationale}. "
            "Also set top-level action_id/action_name/confidence/rationale to mirror agent_index 0 for UI compatibility. "
            "Choose action_id values only from allowed_actions. Use release actions when a sticky action should stop. "
            "If an agent should defer to GRF built-in behavior, explicitly choose action_id 19. "
            "Use the image first, then verify with tactical_state and squad_tactical_state. "
            "Coordinate the players: avoid bunching, use support lanes, and make actions fulfill strategy_plan.\n\n"
            f"STATE_JSON: {state_json}"
        )

    if frame_image_url:
        user_content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": frame_image_url}},
        ]
    else:
        user_content = user_text
    return [{"role": "system", "content": system}, {"role": "user", "content": user_content}]

def _empty_provider_result(provider_id: str, label: str, configured: bool, model_label: str, task: str = "play") -> Dict[str, Any]:
    return {
        "id": provider_id,
        "label": label,
        "model_label": model_label,
        "configured": configured,
        "status": "ready" if configured else "not_configured",
        "content": "",
        "parsed": None,
        "latency_ms": None,
        "first_token_ms": None,
        "usage": None,
        "reasoning_effort": _cerebras_reasoning_effort(task) if provider_id == "cerebras" else None,
        "prompt_cache_key": CEREBRAS_PROMPT_CACHE_KEY if provider_id == "cerebras" else None,
        "tool": "execute_play" if provider_id == "cerebras" and task in {"decision", "play_tool"} and CEREBRAS_USE_ACTION_TOOL else None,
        "tool_call_used": False,
        "error": None,
    }


def _message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        content = "".join(parts)
    if content:
        return str(content).strip()
    reasoning = getattr(message, "reasoning", None)
    return str(reasoning or "").strip()


def _field_value(source: Any, field: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(field)
    return getattr(source, field, None)


def _usage_payload(usage: Any) -> Optional[Dict[str, Any]]:
    if usage is None:
        return None
    prompt_details = _field_value(usage, "prompt_tokens_details")
    return {
        "prompt_tokens": _field_value(usage, "prompt_tokens"),
        "completion_tokens": _field_value(usage, "completion_tokens"),
        "total_tokens": _field_value(usage, "total_tokens"),
        "cached_tokens": _field_value(prompt_details, "cached_tokens"),
    }


def _merge_usage_payloads(*items: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    merged = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    seen = False
    for item in items:
        if not item:
            continue
        seen = True
        for key in merged:
            value = item.get(key)
            if isinstance(value, (int, float)):
                merged[key] += int(value)
    return merged if seen else None


def _strategy_plan_payload(provider_result: Dict[str, Any]) -> Dict[str, Any]:
    parsed = provider_result.get("parsed") or {}
    summary = str(
        parsed.get("summary")
        or parsed.get("scoring_plan")
        or parsed.get("plan")
        or provider_result.get("content")
        or "Create a coordinated chance from the current frame."
    ).strip()[:420]
    assignments = parsed.get("assignments")
    if not isinstance(assignments, list):
        assignments = []
    risks = parsed.get("risks") if isinstance(parsed.get("risks"), list) else []
    return {"phase": "strategize", "summary": summary, "assignments": assignments, "risks": risks}


def _tool_arguments(message: Any, tool_name: str) -> Optional[Dict[str, Any]]:
    tool_calls = _field_value(message, "tool_calls") or []
    for call in tool_calls:
        function = _field_value(call, "function")
        if _field_value(function, "name") != tool_name:
            continue
        raw_arguments = _field_value(function, "arguments")
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if not raw_arguments:
            continue
        try:
            parsed = json.loads(str(raw_arguments))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _parse_provider_json(content: str) -> Optional[Dict[str, Any]]:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _call_cerebras_provider(context: Dict[str, Any], frame_image_url: Optional[str] = None, task: str = "play") -> Dict[str, Any]:
    if task == "decision":
        return _call_cerebras_strategy_play_provider(context, frame_image_url)
    return _call_cerebras_completion(context, frame_image_url, task)


def _call_cerebras_strategy_play_provider(context: Dict[str, Any], frame_image_url: Optional[str] = None) -> Dict[str, Any]:
    started = time.perf_counter()
    strategy_result = _call_cerebras_completion(context, frame_image_url, "strategy")
    if strategy_result["status"] != "ok":
        strategy_result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        return strategy_result

    strategy_plan = _strategy_plan_payload(strategy_result)
    play_context = dict(context)
    play_context["phase"] = "play"
    play_context["strategy_plan"] = strategy_plan
    play_result = _call_cerebras_completion(play_context, None, "play_tool")
    total_latency = round((time.perf_counter() - started) * 1000.0, 2)
    strategy_latency = strategy_result.get("latency_ms")
    play_latency = play_result.get("latency_ms")
    strategy_first_token = strategy_result.get("first_token_ms")
    play_first_token = play_result.get("first_token_ms")
    play_first_from_start = None
    if isinstance(strategy_latency, (int, float)) and isinstance(play_first_token, (int, float)):
        play_first_from_start = round(float(strategy_latency) + float(play_first_token), 2)
    play_result["latency_ms"] = total_latency
    play_result["background_latency_ms"] = total_latency
    play_result["strategy_latency_ms"] = strategy_latency
    play_result["strategy_first_token_ms"] = strategy_first_token
    play_result["play_latency_ms"] = play_latency
    play_result["play_first_token_ms"] = play_first_token
    play_result["play_first_token_from_start_ms"] = play_first_from_start
    play_result["strategy_sdk_create_ms"] = strategy_result.get("sdk_create_ms")
    play_result["play_sdk_create_ms"] = play_result.get("sdk_create_ms")
    play_result["first_token_ms"] = strategy_first_token if strategy_first_token is not None else play_first_from_start
    play_result["usage"] = _merge_usage_payloads(strategy_result.get("usage"), play_result.get("usage"))
    play_result["reasoning_effort"] = f"{_cerebras_reasoning_effort('strategy')} -> {_cerebras_reasoning_effort('play_tool')}"

    parsed = play_result.get("parsed") if isinstance(play_result.get("parsed"), dict) else {}
    parsed.setdefault("phase", "decision")
    parsed.setdefault("summary", strategy_plan["summary"])
    parsed.setdefault("assignments", strategy_plan.get("assignments", []))
    play_result["parsed"] = parsed
    return play_result


def _call_cerebras_completion(context: Dict[str, Any], frame_image_url: Optional[str] = None, task: str = "play") -> Dict[str, Any]:
    configured = bool(os.getenv("CEREBRAS_API_KEY"))
    result = _empty_provider_result("cerebras", "Cerebras", configured, "Gemma 4 31B", task)
    if not configured:
        result["error"] = "CEREBRAS_API_KEY is not available to the app runtime."
        return result

    started = time.perf_counter()
    first_token_ms = None
    chunks: List[str] = []
    usage_payload = None
    tool_call_used = False
    sdk_create_ms = None
    try:
        from cerebras.cloud.sdk import Cerebras

        reasoning_effort = _cerebras_reasoning_effort(task)
        if task == "strategy":
            completion_limit = CEREBRAS_STRATEGY_MAX_TOKENS
        elif task == "play_tool":
            completion_limit = CEREBRAS_PLAY_TOOL_MAX_TOKENS
        else:
            completion_limit = CEREBRAS_DEFAULT_MAX_TOKENS
        use_action_tool = CEREBRAS_USE_ACTION_TOOL and task in {"decision", "play_tool"}
        create_kwargs = {
            "messages": build_provider_messages(context, frame_image_url, task),
            "model": CEREBRAS_MODEL,
            "stream": not use_action_tool,
            "max_completion_tokens": completion_limit,
            "temperature": 0.2,
            "top_p": 1,
        }
        if not use_action_tool:
            create_kwargs["stream_options"] = {"include_usage": True}
            if task == "strategy":
                create_kwargs["response_format"] = _strategy_response_format()
            elif task in {"decision", "play_tool"}:
                create_kwargs["response_format"] = _play_response_format()
        else:
            create_kwargs["tools"] = [_execute_play_tool()]
            create_kwargs["tool_choice"] = "auto"
            create_kwargs["parallel_tool_calls"] = False

        if CEREBRAS_DISABLE_REASONING:
            create_kwargs["disable_reasoning"] = True
        elif reasoning_effort != "none":
            create_kwargs["reasoning_effort"] = reasoning_effort
        if CEREBRAS_PROMPT_CACHE_KEY:
            create_kwargs["extra_body"] = {"prompt_cache_key": CEREBRAS_PROMPT_CACHE_KEY}

        client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"), max_retries=0)
        if use_action_tool:
            sdk_started = time.perf_counter()
            response = client.chat.completions.create(**create_kwargs)
            sdk_create_ms = round((time.perf_counter() - sdk_started) * 1000.0, 2)
            usage_payload = _usage_payload(getattr(response, "usage", None))
            choices = getattr(response, "choices", None) or []
            message = getattr(choices[0], "message", None) if choices else None
            parsed = _tool_arguments(message, "execute_play")
            if parsed is None:
                content = _message_content(message)
                parsed = _parse_provider_json(content)
            else:
                tool_call_used = True
                content = json.dumps(parsed, separators=(",", ":"))
        else:
            sdk_started = time.perf_counter()
            stream = client.chat.completions.create(**create_kwargs)
            sdk_create_ms = round((time.perf_counter() - sdk_started) * 1000.0, 2)
            for chunk in stream:
                chunk_usage = _usage_payload(getattr(chunk, "usage", None))
                if chunk_usage is not None:
                    usage_payload = chunk_usage

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta_obj = getattr(choices[0], "delta", None)
                delta = getattr(delta_obj, "content", None) or ""
                if delta and first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000.0, 2)
                chunks.append(delta)

            content = "".join(chunks).strip()
            parsed = _parse_provider_json(content)
        result.update(
            {
                "status": "ok",
                "content": content,
                "parsed": parsed,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "first_token_ms": first_token_ms,
                "model_label": "Gemma 4 31B",
                "usage": usage_payload,
                "sdk_create_ms": sdk_create_ms,
                "reasoning_effort": _cerebras_reasoning_effort(task),
                "prompt_cache_key": CEREBRAS_PROMPT_CACHE_KEY or None,
                "tool": "execute_play" if use_action_tool else None,
                "tool_call_used": tool_call_used,
            }
        )
    except Exception as exc:  # Keep provider failures visible without exposing secrets.
        result.update(
            {
                "status": "error",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "first_token_ms": first_token_ms,
                "usage": usage_payload,
                "sdk_create_ms": sdk_create_ms,
                "reasoning_effort": _cerebras_reasoning_effort(task),
                "prompt_cache_key": CEREBRAS_PROMPT_CACHE_KEY or None,
                "tool": "execute_play" if use_action_tool else None,
                "tool_call_used": tool_call_used,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        )
    return result


def _call_gpu_provider(context: Dict[str, Any], frame_image_url: Optional[str] = None, task: str = "play") -> Dict[str, Any]:
    if task == "decision":
        return _call_gpu_strategy_play_provider(context, frame_image_url)
    return _call_gpu_completion(context, frame_image_url, task)


def _call_gpu_strategy_play_provider(context: Dict[str, Any], frame_image_url: Optional[str] = None) -> Dict[str, Any]:
    started = time.perf_counter()
    strategy_result = _call_gpu_completion(context, frame_image_url, "strategy")
    if strategy_result["status"] != "ok":
        strategy_result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        return strategy_result

    strategy_plan = _strategy_plan_payload(strategy_result)
    play_context = dict(context)
    play_context["phase"] = "play"
    play_context["strategy_plan"] = strategy_plan
    play_result = _call_gpu_completion(play_context, None, "play_tool")
    total_latency = round((time.perf_counter() - started) * 1000.0, 2)
    strategy_latency = strategy_result.get("latency_ms")
    play_latency = play_result.get("latency_ms")

    play_result["latency_ms"] = total_latency
    play_result["background_latency_ms"] = total_latency
    play_result["strategy_latency_ms"] = strategy_latency
    play_result["strategy_first_token_ms"] = strategy_result.get("first_token_ms")
    play_result["play_latency_ms"] = play_latency
    play_result["play_first_token_ms"] = play_result.get("first_token_ms")
    play_result["play_first_token_from_start_ms"] = None
    play_result["strategy_sdk_create_ms"] = strategy_result.get("sdk_create_ms")
    play_result["play_sdk_create_ms"] = play_result.get("sdk_create_ms")
    play_result["first_token_ms"] = strategy_result.get("first_token_ms")
    play_result["usage"] = _merge_usage_payloads(strategy_result.get("usage"), play_result.get("usage"))

    parsed = play_result.get("parsed") if isinstance(play_result.get("parsed"), dict) else {}
    parsed.setdefault("phase", "decision")
    parsed.setdefault("summary", strategy_plan["summary"])
    parsed.setdefault("assignments", strategy_plan.get("assignments", []))
    play_result["parsed"] = parsed
    return play_result


def _call_gpu_completion(context: Dict[str, Any], frame_image_url: Optional[str] = None, task: str = "play") -> Dict[str, Any]:
    configured = bool(_gpu_api_key())
    result = _empty_provider_result("gpu", "GPU", configured, "Gemma 4 31B", task)
    if not configured:
        result["error"] = "GPU_API_KEY is not available to the app runtime."
        return result

    started = time.perf_counter()
    sdk_create_ms = None
    usage_payload = None
    try:
        from together import Together

        client = Together(api_key=_gpu_api_key())
        reasoning_effort = _gpu_reasoning_effort(task)
        sdk_started = time.perf_counter()
        create_kwargs = {
            "model": GPU_MODEL,
            "messages": build_provider_messages(context, frame_image_url, task),
            "max_tokens": GPU_MAX_TOKENS,
            "temperature": 0.2,
            "top_p": 1,
            "reasoning_effort": reasoning_effort,
        }
        response = client.chat.completions.create(**create_kwargs)
        sdk_create_ms = round((time.perf_counter() - sdk_started) * 1000.0, 2)
        usage_payload = _usage_payload(getattr(response, "usage", None))
        content = _message_content(response.choices[0].message)
        result.update(
            {
                "status": "ok",
                "content": content,
                "parsed": _parse_provider_json(content),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "first_token_ms": None,
                "model_label": "Gemma 4 31B",
                "usage": usage_payload,
                "sdk_create_ms": sdk_create_ms,
                "reasoning_effort": reasoning_effort,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "usage": usage_payload,
                "sdk_create_ms": sdk_create_ms,
                "reasoning_effort": _gpu_reasoning_effort(task),
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        )
    return result


def _provider_latency_value(provider_result: Dict[str, Any]) -> Optional[float]:
    for key in ("visible_latency_ms", "latency_ms"):
        value = provider_result.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _comparison_speedup(cerebras: Dict[str, Any], gpu: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cerebras_ms = _provider_latency_value(cerebras)
    gpu_ms = _provider_latency_value(gpu)
    if cerebras_ms is None or gpu_ms is None:
        return None
    if cerebras_ms <= 0 or gpu_ms <= 0:
        return None

    ratio = gpu_ms / cerebras_ms
    if ratio >= 1:
        return {
            "winner": "cerebras",
            "label": "Cerebras",
            "ratio": round(ratio, 2),
            "cerebras_ms": round(cerebras_ms, 2),
            "gpu_ms": round(gpu_ms, 2),
        }
    return {
        "winner": "gpu",
        "label": "GPU",
        "ratio": round(1 / ratio, 2),
        "cerebras_ms": round(cerebras_ms, 2),
        "gpu_ms": round(gpu_ms, 2),
    }


async def compare_remote_providers(context: Dict[str, Any], frame_image_url: Optional[str] = None) -> Dict[str, Any]:
    started = time.perf_counter()
    loop = asyncio.get_running_loop()
    cerebras_task = loop.run_in_executor(None, _call_cerebras_provider, context, frame_image_url, "decision")
    gpu_task = loop.run_in_executor(None, _call_gpu_provider, context, frame_image_url, "decision")
    try:
        cerebras, gpu = await asyncio.wait_for(
            asyncio.gather(cerebras_task, gpu_task),
            timeout=PROVIDER_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "context": context,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "providers": [
                {**_empty_provider_result("cerebras", "Cerebras", bool(os.getenv("CEREBRAS_API_KEY")), "Gemma 4 31B"), "status": "timeout"},
                {**_empty_provider_result("gpu", "GPU", bool(_gpu_api_key()), "Gemma 4 31B"), "status": "timeout"},
            ],
        }
    return {
        "status": "ok",
        "context": context,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "speedup": _comparison_speedup(cerebras, gpu),
        "providers": [cerebras, gpu],
    }


class SessionConfigRequest(BaseModel):
    scenario: str
    opponent: str


session = FootballSession()
app = FastAPI(title="GemmaFC", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/record")
def record() -> FileResponse:
    return FileResponse(STATIC_ROOT / "record.html")


@app.get("/api/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/session/options")
def session_options() -> Dict[str, Any]:
    return session.options()


@app.post("/api/session/config")
def session_config(payload: SessionConfigRequest) -> Dict[str, Any]:
    return session.configure(payload.scenario, payload.opponent)


@app.post("/api/control-mode/{mode}")
def control_mode(mode: str) -> Dict[str, Any]:
    return session.set_control_mode(mode)


@app.get("/api/providers/status")
def providers_status() -> Dict[str, Any]:
    return provider_status()


@app.get("/api/frame.jpg")
def frame_jpg() -> Response:
    return session.frame_response()


@app.post("/api/providers/compare")
async def providers_compare() -> Dict[str, Any]:
    payload = session.provider_payload()
    return await compare_remote_providers(payload["context"], payload["frame_image_url"])


@app.post("/api/reset")
def reset() -> Dict[str, Any]:
    return session.reset()


@app.post("/api/reset/{controller}")
def reset_controller(controller: str) -> Dict[str, Any]:
    return session.reset(controller)


@app.post("/api/step")
def step() -> Dict[str, Any]:
    return session.step("local")


@app.post("/api/step/{controller}")
def step_controller(controller: str) -> Dict[str, Any]:
    return session.step(controller)


@app.post("/api/render-mode/{mode}")
def render_mode(mode: str) -> Dict[str, Any]:
    return session.set_render_mode(mode)


@app.get("/api/state")
def state() -> Dict[str, Any]:
    return session.state()

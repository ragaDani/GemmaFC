const state = {
  running: false,
  timer: null,
  scheduleToken: 0,
  frameReady: Promise.resolve(),
  frameVersion: 0,
  busy: false,
  providerBusy: false,
  controller: 'cerebras',
  controlMode: 'single',
  controlledPlayers: 1,
  providerConfigured: {},
  userSelectedController: false,
};

const els = {
  connectionStatus: document.getElementById('connectionStatus'),
  runButton: document.getElementById('runButton'),
  stepButton: document.getElementById('stepButton'),
  resetButton: document.getElementById('resetButton'),
  qualityButtons: document.querySelectorAll('[data-render-mode]'),
  agentButtons: document.querySelectorAll('[data-controller]'),
  modeButtons: document.querySelectorAll('[data-control-mode]'),
  compareProvidersButton: document.getElementById('compareProvidersButton'),
  providerSpeedup: document.getElementById('providerSpeedup'),
  providerCards: document.querySelectorAll('[data-provider-card]'),
  gameFrame: document.getElementById('gameFrame'),
  framePlaceholder: document.getElementById('framePlaceholder'),
  scenarioSelect: document.getElementById('scenarioSelect'),
  opponentSelect: document.getElementById('opponentSelect'),
  stepCount: document.getElementById('stepCount'),
  rewardValue: document.getElementById('rewardValue'),
  latencyValue: document.getElementById('latencyValue'),
  repairCount: document.getElementById('repairCount'),
  actionName: document.getElementById('actionName'),
  actionIntent: document.getElementById('actionIntent'),
  scoreValue: document.getElementById('scoreValue'),
  repairBanner: document.getElementById('repairBanner'),
  loopList: document.getElementById('loopList'),
  policyProvider: document.getElementById('policyProvider'),
  policyMode: document.getElementById('policyMode'),
  controllerName: document.getElementById('controllerName'),
  decisionLatency: document.getElementById('decisionLatency'),
  frameMean: document.getElementById('frameMean'),
  uptimeValue: document.getElementById('uptimeValue'),
};

async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function post(path) {
  return requestJson(path, { method: 'POST' });
}

async function postBody(path, payload) {
  return requestJson(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

function setStatus(text, kind = 'ready') {
  els.connectionStatus.textContent = text;
  els.connectionStatus.classList.remove('ready', 'error');
  if (kind) {
    els.connectionStatus.classList.add(kind);
  }
}

function setRunning(next) {
  if (next && !canRunController(state.controller)) {
    state.running = false;
    updateAgentButtons();
    setStatus(`${controllerLabel(state.controller)} key is missing`, 'ready');
    return;
  }
  state.running = next;
  const icon = els.runButton.querySelector('.icon-run');
  const label = els.runButton.querySelector('span:last-child');
  const controller = controllerLabel(state.controller);
  els.runButton.classList.toggle('running', next);
  icon.classList.toggle('pause', next);
  label.textContent = next ? 'Pause' : `Run ${controller}`;
  els.runButton.title = next ? 'Pause loop' : `Run ${controller}`;

  clearRunTimer();
  if (next) scheduleNextStepAfterFrame();
}

function clearRunTimer() {
  state.scheduleToken += 1;
  if (state.timer) window.clearTimeout(state.timer);
  state.timer = null;
}

function scheduleNextStep() {
  clearRunTimer();
  if (!state.running) return;
  const token = ++state.scheduleToken;
  state.timer = window.setTimeout(() => {
    if (token !== state.scheduleToken) return;
    state.timer = null;
    stepOnce(state.controller);
  }, 0);
}

function scheduleNextStepAfterFrame(frameVersion = state.frameVersion) {
  clearRunTimer();
  if (!state.running) return;
  const token = ++state.scheduleToken;
  state.frameReady.finally(() => {
    if (!state.running || token !== state.scheduleToken || frameVersion !== state.frameVersion) return;
    scheduleNextStep();
  });
}

function watchFrameLoad(frameSrc) {
  if (!frameSrc) return state.frameVersion;
  const version = state.frameVersion + 1;
  state.frameVersion = version;
  state.frameReady = new Promise((resolve) => {
    let fallback = null;
    const done = () => {
      if (fallback !== null) window.clearTimeout(fallback);
      els.gameFrame.removeEventListener('load', done);
      els.gameFrame.removeEventListener('error', done);
      resolve();
    };
    fallback = window.setTimeout(done, 1000);
    els.gameFrame.addEventListener('load', done, { once: true });
    els.gameFrame.addEventListener('error', done, { once: true });
  });
  return version;
}

async function resetEnv() {
  if (state.busy) return;
  state.busy = true;
  try {
    setStatus('Resetting environment');
    const data = await post(controllerResetPath(state.controller));
    render(data);
    setStatus('Environment ready', 'ready');
  } catch (error) {
    setStatus('Reset failed', 'error');
    console.error(error);
  } finally {
    state.busy = false;
  }
}

async function stepOnce(controller = state.controller) {
  if (state.busy) return;
  state.busy = true;
  try {
    setStatus(`${controllerLabel(controller)} thinking`);
    const data = await post(controllerStepPath(controller));
    const frameVersion = render(data);
    setStatus(state.running ? `${controllerLabel(controller)} loop running` : `${controllerLabel(controller)} tick complete`, 'ready');
    if (state.running) scheduleNextStepAfterFrame(frameVersion);
  } catch (error) {
    setRunning(false);
    setStatus(`${controllerLabel(controller)} tick failed`, 'error');
    console.error(error);
  } finally {
    state.busy = false;
  }
}

async function setRenderMode(mode) {
  if (state.busy) return;
  state.busy = true;
  try {
    setStatus(mode === 'recording' ? 'Switching to 720p' : 'Switching to live mode');
    const data = await post(`/api/render-mode/${mode}`);
    if (data.frame || data.frame_url) {
      render(data);
    }
    updateQualityButtons(data.render_mode || mode);
    setStatus(mode === 'recording' ? '720p recording frames enabled' : 'Live frames enabled', 'ready');
  } catch (error) {
    setStatus('Render mode failed', 'error');
    console.error(error);
  } finally {
    state.busy = false;
  }
}

function updateQualityButtons(mode) {
  els.qualityButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.renderMode === mode);
  });
}

function controllerLabel(id) {
  if (id === 'cerebras') return 'Cerebras';
  if (id === 'gpu') return 'GPU';
  return 'Local';
}

function controllerStepPath(id) {
  if (id === 'cerebras') return '/api/step/cerebras';
  if (id === 'gpu') return '/api/step/gpu';
  return `/api/step/${encodeURIComponent(id)}`;
}

function controllerResetPath(id) {
  if (id === 'cerebras') return '/api/reset/cerebras';
  if (id === 'gpu') return '/api/reset/gpu';
  return `/api/reset/${encodeURIComponent(id)}`;
}

function setController(id) {
  if (!id || state.controller === id) {
    updateAgentButtons();
    return;
  }
  state.controller = id;
  state.userSelectedController = true;
  setRunning(false);
  updateAgentButtons();
  setStatus(`${controllerLabel(id)} selected`, 'ready');
}

function controlModeLabel(id) {
  if (id === 'squad') return 'Squad';
  return 'Single';
}

async function setControlMode(mode) {
  if (!mode || state.controlMode === mode || state.busy) {
    updateModeButtons();
    return;
  }
  setRunning(false);
  state.busy = true;
  try {
    setStatus(`Switching to ${controlModeLabel(mode)} mode`);
    const data = await post(`/api/control-mode/${mode}`);
    render(data);
    setStatus(`${controlModeLabel(mode)} mode ready`, 'ready');
  } catch (error) {
    setStatus('Control mode failed', 'error');
    console.error(error);
  } finally {
    state.busy = false;
  }
}

function updateModeButtons() {
  els.modeButtons.forEach((button) => {
    const mode = button.dataset.controlMode;
    button.classList.toggle('active', mode === state.controlMode);
    button.title = mode === 'squad'
      ? 'Run coordinated Gemma strategy and play phases across all controlled yellow players'
      : 'Run the same strategy and play loop for the active yellow player';
  });
}

function controllerAvailable(id) {
  return state.providerConfigured[id] !== false;
}

function canRunController(id) {
  return controllerAvailable(id);
}

function updateAgentButtons() {
  els.agentButtons.forEach((button) => {
    const id = button.dataset.controller;
    const available = controllerAvailable(id);
    button.classList.toggle('active', id === state.controller);
    button.disabled = !available;
    button.dataset.configured = String(available);
    button.title = available ? `Use ${controllerLabel(id)} as the live controller` : `${controllerLabel(id)} key is missing`;
  });
  if (els.controllerName) els.controllerName.textContent = `${controllerLabel(state.controller)} / ${controlModeLabel(state.controlMode)}`;
  if (els.policyProvider) els.policyProvider.textContent = `${controllerLabel(state.controller)} ${controlModeLabel(state.controlMode)}`;
  const runLabel = els.runButton.querySelector('span:last-child');
  const tickLabel = els.stepButton.querySelector('span:last-child');
  els.runButton.disabled = !canRunController(state.controller);
  els.runButton.title = canRunController(state.controller)
    ? `Run ${controllerLabel(state.controller)}`
    : `${controllerLabel(state.controller)} key is missing`;
  if (!state.running) runLabel.textContent = canRunController(state.controller)
    ? `Run ${controllerLabel(state.controller)}`
    : `${controllerLabel(state.controller)} unavailable`;
  tickLabel.textContent = `Tick ${controllerLabel(state.controller)}`;
}

async function loadSessionOptions() {
  const data = await requestJson('/api/session/options');
  renderSessionOptions(data);
}

function renderSessionOptions(data) {
  populateSelect(els.scenarioSelect, data.scenarios || data.available_scenarios || []);
  populateSelect(els.opponentSelect, data.opponents || data.available_opponents || []);
  if (!state.userSelectedController && (data.selected_controller || data.active_controller)) {
    state.controller = data.selected_controller || data.active_controller;
  }
  if (data.selected_control_mode || data.control_mode) {
    state.controlMode = data.selected_control_mode || data.control_mode;
  }
  if (data.controlled_players !== undefined) {
    state.controlledPlayers = data.controlled_players;
  }
  updateAgentButtons();
  updateModeButtons();
  if (data.selected_scenario || data.scenario) {
    els.scenarioSelect.value = data.selected_scenario || data.scenario;
  }
  if (data.selected_opponent || data.opponent_mode) {
    els.opponentSelect.value = data.selected_opponent || data.opponent_mode;
  }
}

function populateSelect(select, items) {
  if (!select || !items.length) return;
  const existing = Array.from(select.options).map((option) => option.value).join('|');
  const incoming = items.map((item) => item.id).join('|');
  if (existing === incoming) return;

  select.innerHTML = '';
  items.forEach((item) => {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = item.label || item.id;
    option.title = item.description || item.id;
    select.appendChild(option);
  });
}

async function configureSession() {
  if (state.busy || !els.scenarioSelect.value || !els.opponentSelect.value) return;
  setRunning(false);
  state.busy = true;
  try {
    setStatus('Loading scenario');
    const data = await postBody('/api/session/config', {
      scenario: els.scenarioSelect.value,
      opponent: els.opponentSelect.value,
    });
    render(data);
    setStatus(`${data.scenario_label || data.scenario} ready`, 'ready');
  } catch (error) {
    setStatus('Scenario failed', 'error');
    console.error(error);
  } finally {
    state.busy = false;
  }
}

async function loadProviderStatus() {
  try {
    const response = await fetch('/api/providers/status');
    if (!response.ok) return;
    const data = await response.json();
    data.providers.forEach((provider) => {
      state.providerConfigured[provider.id] = Boolean(provider.configured);
      const card = providerCard(provider.id);
      if (!card) return;
      setProviderCard(provider.id, {
        status: provider.configured ? 'ready' : 'not_configured',
        label: provider.configured ? 'Ready' : 'Missing key',
        action: provider.configured ? 'Ready for live control or same-state comparison.' : 'Provider key is not available to the app runtime.',
      });
    });
    if (!state.userSelectedController) {
      state.controller = state.providerConfigured.cerebras ? 'cerebras' : 'gpu';
    } else if (!controllerAvailable(state.controller)) {
      state.controller = state.providerConfigured.cerebras ? 'cerebras' : 'gpu';
      state.userSelectedController = false;
    }
    updateAgentButtons();
    return data;
  } catch (error) {
    console.error(error);
  }
}

async function compareProviders() {
  if (state.providerBusy) return;
  state.providerBusy = true;
  els.compareProvidersButton.disabled = true;
  setProviderSpeedup('Racing...', 'running');
  setStatus('Comparing Cerebras and GPU');
  ['cerebras', 'gpu'].forEach((id) => {
    setProviderCard(id, { status: 'running', label: 'Running', total: '--', first: '--', action: 'Calling Gemma on the current game state...' });
  });

  try {
    const data = await post('/api/providers/compare');
    data.providers.forEach((provider) => renderProviderResult(provider));
    renderProviderSpeedup(data);
    setStatus('Provider comparison complete', 'ready');
  } catch (error) {
    ['cerebras', 'gpu'].forEach((id) => {
      setProviderCard(id, { status: 'error', label: 'Error', action: 'Comparison request failed.' });
    });
    setProviderSpeedup('Race failed', 'error');
    setStatus('Provider comparison failed', 'error');
    console.error(error);
  } finally {
    els.compareProvidersButton.disabled = false;
    state.providerBusy = false;
  }
}

function setProviderSpeedup(text, status = 'ready') {
  if (!els.providerSpeedup) return;
  els.providerSpeedup.textContent = text;
  els.providerSpeedup.classList.remove('ready', 'running', 'ok', 'error');
  if (status) els.providerSpeedup.classList.add(status);
}

function renderProviderSpeedup(data) {
  const speedup = data.speedup || computeProviderSpeedup(data.providers || []);
  if (!speedup) {
    setProviderSpeedup('No race result', 'error');
    return;
  }
  const ratio = Number(speedup.ratio);
  if (!Number.isFinite(ratio) || ratio <= 0) {
    setProviderSpeedup('No race result', 'error');
    return;
  }
  const label = speedup.winner === 'cerebras'
    ? `Cerebras ${formatSpeedupRatio(ratio)}x faster`
    : `GPU ${formatSpeedupRatio(ratio)}x faster`;
  setProviderSpeedup(label, speedup.winner === 'cerebras' ? 'ok' : 'ready');
}

function computeProviderSpeedup(providers) {
  const cerebras = providers.find((provider) => provider.id === 'cerebras');
  const gpu = providers.find((provider) => provider.id === 'gpu');
  if (!cerebras || !gpu) return null;
  const cerebrasMs = Number(providerVisibleLatency(cerebras));
  const gpuMs = Number(providerVisibleLatency(gpu));
  if (!Number.isFinite(cerebrasMs) || !Number.isFinite(gpuMs) || cerebrasMs <= 0 || gpuMs <= 0) return null;
  if (gpuMs >= cerebrasMs) {
    return { winner: 'cerebras', ratio: gpuMs / cerebrasMs };
  }
  return { winner: 'gpu', ratio: cerebrasMs / gpuMs };
}

function formatSpeedupRatio(value) {
  if (value >= 100) return Math.round(value).toString();
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function renderProviderResult(provider) {
  const parsed = provider.parsed || {};
  const actionName = parsed.action_name || parsed.action || '';
  const rationale = parsed.rationale || provider.content || provider.error || 'No response content.';
  const repair = parsed.repair_check ? ` Repair: ${parsed.repair_check}` : '';
  const cache = usageSummary(provider.usage);
  const tool = provider.tool_call_used ? ` Tool: ${provider.tool}.` : '';
  const timing = providerTimingSummary(provider);
  const summary = actionName ? `${actionName}. ${rationale}${repair}${tool}${timing}${cache}` : `${rationale}${repair}${tool}${timing}${cache}`;
  setProviderCard(provider.id, {
    status: provider.status,
    label: provider.status === 'ok' ? 'Complete' : statusLabel(provider.status),
    total: formatProviderMs(providerVisibleLatency(provider)),
    first: formatProviderMs(provider.first_token_ms),
    action: summary,
  });
}

function providerVisibleLatency(provider) {
  if (provider.visible_latency_ms !== null && provider.visible_latency_ms !== undefined) return provider.visible_latency_ms;
  if (provider.prefetched && provider.prefetch_wait_ms !== null && provider.prefetch_wait_ms !== undefined) return provider.prefetch_wait_ms;
  return provider.latency_ms;
}

function providerTimingSummary(provider) {
  const parts = [];
  if (provider.prefetched) parts.push(`Visible wait: ${formatProviderMs(providerVisibleLatency(provider))}`);
  if (provider.latency_ms !== null && provider.latency_ms !== undefined) parts.push(`Gemma: ${formatProviderMs(provider.latency_ms)}`);
  if (provider.strategy_latency_ms !== null && provider.strategy_latency_ms !== undefined && provider.play_latency_ms !== null && provider.play_latency_ms !== undefined) {
    parts.push(`strategy ${formatProviderMs(provider.strategy_latency_ms)}, play ${formatProviderMs(provider.play_latency_ms)}`);
  }
  if (!parts.length) return '';
  return ` ${parts.join('; ')}.`;
}

function usageSummary(usage) {
  if (!usage) return '';
  const cached = usage.cached_tokens;
  const prompt = usage.prompt_tokens;
  if (cached === null || cached === undefined) return '';
  if (prompt === null || prompt === undefined) return ` Cache: ${cached} tokens.`;
  return ` Cache: ${cached}/${prompt} prompt tokens.`;
}

function providerCard(id) {
  return document.querySelector(`[data-provider-card="${id}"]`);
}

function setProviderCard(id, data) {
  const card = providerCard(id);
  if (!card) return;
  card.classList.remove('ok', 'error', 'not_configured', 'timeout', 'running', 'ready');
  if (data.status) card.classList.add(data.status);
  const status = card.querySelector('.provider-status');
  const total = card.querySelector('[data-provider-total]');
  const first = card.querySelector('[data-provider-first]');
  const action = card.querySelector('[data-provider-action]');
  if (status && data.label) status.textContent = data.label;
  if (total && data.total) total.textContent = data.total;
  if (first && data.first) first.textContent = data.first;
  if (action && data.action) action.textContent = data.action;
}

function statusLabel(status) {
  if (status === 'not_configured') return 'Missing key';
  if (status === 'timeout') return 'Timeout';
  if (status === 'error') return 'Error';
  if (status === 'ready') return 'Ready';
  return status || 'Unknown';
}

function formatProviderMs(value) {
  if (value === null || value === undefined) return '--';
  if (value < 1) return '<1 ms';
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${Math.round(value)} ms`;
}

function render(data) {
  renderSessionOptions(data);
  if (data.active_controller && state.userSelectedController) state.controller = data.active_controller;
  if (data.control_mode) state.controlMode = data.control_mode;
  if (data.controlled_players !== undefined) state.controlledPlayers = data.controlled_players;
  updateAgentButtons();
  updateModeButtons();
  els.stepCount.textContent = data.step.toString();
  els.rewardValue.textContent = data.total_reward.toFixed(2);
  els.latencyValue.textContent = `${Math.round(data.avg_latency_ms)} ms`;
  els.repairCount.textContent = data.repairs.toString();
  const playersLabel = state.controlledPlayers > 1 ? `${state.controlledPlayers} agents` : '1 agent';
  els.policyProvider.textContent = `${data.policy.provider} ${controlModeLabel(state.controlMode)}`;
  els.policyMode.textContent = `${data.policy.mode} / ${playersLabel}`;
  if (els.controllerName) els.controllerName.textContent = `${data.active_controller_label || data.policy.provider} / ${controlModeLabel(state.controlMode)}`;
  els.frameMean.textContent = data.frame_stats.mean.toFixed(1);
  els.uptimeValue.textContent = `${Math.round(data.uptime_s)}s`;
  updateQualityButtons(data.render_mode || 'live');

  const frameSrc = data.frame_url || data.frame;
  let frameVersion = state.frameVersion;
  if (frameSrc) {
    frameVersion = watchFrameLoad(frameSrc);
    els.gameFrame.src = frameSrc;
    els.gameFrame.style.display = 'block';
    els.framePlaceholder.style.display = 'none';
  }

  if (data.action) {
    const squadActions = Array.isArray(data.squad_actions) ? data.squad_actions : [];
    if (squadActions.length > 1) {
      els.actionName.textContent = 'Squad command';
      els.actionIntent.textContent = squadActions
        .map((action) => `P${(action.agent_index ?? 0) + 1}: ${action.name}`)
        .join(' | ');
    } else {
      els.actionName.textContent = data.action.name;
      const providerPrefix = data.action.provider_label ? `${data.action.provider_label}: ` : '';
      els.actionIntent.textContent = `${providerPrefix}${data.action.intent}`;
    }
    els.scoreValue.textContent = data.action.score.toFixed(2);
    if (els.decisionLatency) {
      els.decisionLatency.textContent = formatProviderMs(data.action.provider_latency_ms ?? data.latency_ms);
    }
    if (data.action.repaired) {
      els.repairBanner.classList.add('active');
      els.repairBanner.textContent = `${data.action.candidate_name} repaired to ${data.action.name}. ${data.action.repair_reason}`;
    } else {
      els.repairBanner.classList.remove('active');
      els.repairBanner.textContent = 'No repair applied.';
    }
  } else {
    els.actionName.textContent = 'Waiting';
    els.actionIntent.textContent = 'The environment is ready for the next tick.';
    els.scoreValue.textContent = '0.00';
    els.repairBanner.classList.remove('active');
    els.repairBanner.textContent = 'No repair applied.';
    if (els.decisionLatency) els.decisionLatency.textContent = '--';
  }

  if (data.provider_decision) {
    renderProviderResult(data.provider_decision);
  }

  els.loopList.innerHTML = '';
  data.loop.forEach((item) => {
    const row = document.createElement('div');
    row.className = `loop-item ${item.status}`;
    row.innerHTML = `
      <div class="loop-name">${escapeHtml(item.name)}</div>
      <div class="loop-detail">${escapeHtml(item.detail)}</div>
      <div class="loop-time">${formatMs(item.ms)}</div>
    `;
    els.loopList.appendChild(row);
  });

  if (data.done) {
    setRunning(false);
    setStatus('Episode ended', 'ready');
  }
  return frameVersion;
}

function formatMs(value) {
  if (!value) return '0 ms';
  if (value < 1) return '<1 ms';
  return `${Math.round(value)} ms`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

els.runButton.addEventListener('click', () => {
  if (!canRunController(state.controller)) {
    setStatus(`${controllerLabel(state.controller)} key is missing`, 'ready');
    return;
  }
  setRunning(!state.running);
});

els.stepButton.addEventListener('click', () => {
  setRunning(false);
  stepOnce(state.controller);
});

els.resetButton.addEventListener('click', () => {
  setRunning(false);
  resetEnv();
});

els.qualityButtons.forEach((button) => {
  button.addEventListener('click', () => setRenderMode(button.dataset.renderMode));
});

els.compareProvidersButton.addEventListener('click', compareProviders);
els.agentButtons.forEach((button) => {
  button.addEventListener('click', () => setController(button.dataset.controller));
});
els.modeButtons.forEach((button) => {
  button.addEventListener('click', () => setControlMode(button.dataset.controlMode));
});
els.scenarioSelect.addEventListener('change', configureSession);
els.opponentSelect.addEventListener('change', configureSession);

async function boot() {
  try {
    await loadSessionOptions();
  } catch (error) {
    console.error(error);
  }
  await loadProviderStatus();
  resetEnv();
}

boot();

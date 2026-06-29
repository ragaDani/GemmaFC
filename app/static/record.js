const recordState = {
  running: false,
  busy: false,
  timer: null,
};

const recordEls = {
  frame: document.getElementById('recordFrame'),
  loading: document.getElementById('loadingState'),
  run: document.getElementById('recordRun'),
  step: document.getElementById('recordStep'),
  reset: document.getElementById('recordReset'),
  meta: document.getElementById('recordMeta'),
  stepCount: document.getElementById('recordStepCount'),
  latency: document.getElementById('recordLatency'),
  frameSize: document.getElementById('recordFrameSize'),
  repairs: document.getElementById('recordRepairs'),
  action: document.getElementById('recordAction'),
  intent: document.getElementById('recordIntent'),
  trace: document.getElementById('recordTrace'),
};

async function postJson(path) {
  const response = await fetch(path, { method: 'POST' });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function setRunning(next) {
  recordState.running = next;
  recordEls.run.textContent = next ? 'Pause' : 'Run';
  recordEls.run.classList.toggle('primary', next);
  if (recordState.timer) {
    window.clearInterval(recordState.timer);
    recordState.timer = null;
  }
  if (next) {
    recordState.timer = window.setInterval(() => stepOnce(), 420);
  }
}

async function resetRecording() {
  if (recordState.busy) return;
  recordState.busy = true;
  try {
    recordEls.loading.textContent = 'Preparing 720p recording mode';
    await postJson('/api/render-mode/recording');
    const data = await postJson('/api/reset');
    render(data);
  } finally {
    recordState.busy = false;
  }
}

async function stepOnce() {
  if (recordState.busy) return;
  recordState.busy = true;
  try {
    const data = await postJson('/api/step');
    render(data);
    if (data.done) setRunning(false);
  } finally {
    recordState.busy = false;
  }
}

function render(data) {
  if (data.frame) {
    recordEls.frame.src = data.frame;
    recordEls.frame.style.display = 'block';
    recordEls.loading.style.display = 'none';
  }
  const scenario = data.scenario_label || data.scenario || 'Scenario';
  const opponent = data.opponent_label || data.opponent_mode || 'Opponent';
  recordEls.meta.textContent = `${scenario} vs ${opponent}`;
  recordEls.stepCount.textContent = String(data.step ?? 0);
  recordEls.latency.textContent = `${Math.round(data.avg_latency_ms ?? data.latency_ms ?? 0)} ms`;
  recordEls.repairs.textContent = String(data.repairs ?? 0);
  if (data.frame_size) {
    recordEls.frameSize.textContent = `${data.frame_size.width}x${data.frame_size.height}`;
  }
  if (data.action) {
    recordEls.action.textContent = data.action.name;
    recordEls.intent.textContent = data.action.repaired
      ? `${data.action.candidate_name} repaired to ${data.action.name}. ${data.action.repair_reason}`
      : data.action.intent;
  } else {
    recordEls.action.textContent = 'Waiting';
    recordEls.intent.textContent = 'Recording mode loads a 960x720 game frame.';
  }

  recordEls.trace.innerHTML = '';
  (data.loop || []).forEach((item) => {
    const row = document.createElement('div');
    row.className = `trace-item ${item.status}`;
    row.innerHTML = `<strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.detail)}</span>`;
    recordEls.trace.appendChild(row);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

recordEls.run.addEventListener('click', () => {
  setRunning(!recordState.running);
  if (recordState.running) stepOnce();
});
recordEls.step.addEventListener('click', () => {
  setRunning(false);
  stepOnce();
});
recordEls.reset.addEventListener('click', () => {
  setRunning(false);
  resetRecording();
});

resetRecording();

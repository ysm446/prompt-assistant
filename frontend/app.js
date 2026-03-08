'use strict';

// =========================================================
// Prompt Assistant - フロントエンドロジック
// =========================================================

// ---------------------------------------------------------------------------
// コピーボタン
// ---------------------------------------------------------------------------

document.querySelectorAll('.btn-copy').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = document.getElementById(btn.dataset.target);
    if (!target) return;
    navigator.clipboard.writeText(target.value).then(() => {
      btn.classList.add('copied');
      setTimeout(() => btn.classList.remove('copied'), 1500);
    });
  });
});

// ---------------------------------------------------------------------------
// SSE ストリーミング ユーティリティ
// ---------------------------------------------------------------------------

/**
 * POST リクエストで SSE を受信する。
 * onEvent(data) は各データオブジェクトで呼ばれる。
 * signal で AbortController によるキャンセルが可能。
 */
async function fetchSSE(url, body, onEvent, signal) {
  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (e.name !== 'AbortError') throw e;
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            onEvent(JSON.parse(line.slice(6)));
          } catch (_) { /* ignore parse errors */ }
        }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') throw e;
  } finally {
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// 状態
// ---------------------------------------------------------------------------

let chatAbortCtrl = null;
let videoPromptAbortCtrl = null;

// 設定保存デバウンス
let saveTimer = null;

const PANEL_LAYOUT_KEY = 'panel-layout-v1';

// ---------------------------------------------------------------------------
// スライダー値表示
// ---------------------------------------------------------------------------

function bindSlider(sliderId, displayId, format) {
  const slider = document.getElementById(sliderId);
  const display = document.getElementById(displayId);
  const fmt = format || (v => v);
  slider.addEventListener('input', () => {
    display.textContent = fmt(slider.value);
    scheduleSave();
  });
}

bindSlider('steps-slider', 'steps-val');
bindSlider('cfg-slider', 'cfg-val', v => parseFloat(v).toFixed(1));
bindSlider('width-slider', 'width-val');
bindSlider('height-slider', 'height-val');
bindSlider('comfyui-width', 'comfyui-width-val');
bindSlider('comfyui-height', 'comfyui-height-val');
bindSlider('video-width', 'video-width-val');
bindSlider('video-height', 'video-height-val');

// ---------------------------------------------------------------------------
// バックエンド切り替え
// ---------------------------------------------------------------------------

function updateBackendVisibility() {
  const isComfy = document.querySelector('input[name="backend"]:checked')?.value === 'ComfyUI';
  document.getElementById('comfyui-params').style.display = isComfy ? '' : 'none';
  document.getElementById('forge-params').style.display = isComfy ? 'none' : '';
  document.getElementById('comfyui-seed-controls').style.display = isComfy ? '' : 'none';
  document.getElementById('forge-seed-controls').style.display = isComfy ? 'none' : '';
}

document.querySelectorAll('input[name="backend"]').forEach(r => {
  r.addEventListener('change', () => {
    updateBackendVisibility();
    scheduleSave();
    checkConnection();
  });
});

async function checkConnection() {
  const backend = document.querySelector('input[name="backend"]:checked')?.value;
  const statusEl = document.getElementById('connection-status');
  statusEl.textContent = '接続確認中...';
  try {
    // Just show URL info; actual check happens on generate
    if (backend === 'ComfyUI') {
      statusEl.textContent = 'ComfyUI バックエンドを選択中';
    } else {
      statusEl.textContent = 'WebUI Forge バックエンドを選択中';
    }
  } catch (e) {
    statusEl.textContent = String(e);
  }
}

// ---------------------------------------------------------------------------
// 設定読み込み / 保存
// ---------------------------------------------------------------------------

function getSettings() {
  const backend = document.querySelector('input[name="backend"]:checked')?.value || 'WebUI Forge';
  const sections = Array.from(
    document.querySelectorAll('input[name="video-section"]:checked')
  ).map(el => el.value);

  return {
    model: document.getElementById('model-dropdown').value,
    steps: parseInt(document.getElementById('steps-slider').value),
    cfg: parseFloat(document.getElementById('cfg-slider').value),
    sampler: document.getElementById('sampler-dropdown').value,
    width: parseInt(document.getElementById('width-slider').value),
    height: parseInt(document.getElementById('height-slider').value),
    seed: parseInt(document.getElementById('seed-input').value) || -1,
    backend,
    comfyui_workflow: document.getElementById('comfyui-workflow').value,
    comfyui_width: parseInt(document.getElementById('comfyui-width').value),
    comfyui_height: parseInt(document.getElementById('comfyui-height').value),
    comfyui_seed: parseInt(document.getElementById('comfyui-seed').value) || -1,
    image_save_path: document.getElementById('image-save-path').value,
    video_save_path: document.getElementById('video-save-path').value,
    unload_llm_before_video: document.getElementById('unload-llm-before-video').checked,
    video_workflow: document.getElementById('video-workflow').value,
    video_sections: sections,
    video_width: parseInt(document.getElementById('video-width').value),
    video_height: parseInt(document.getElementById('video-height').value),
    video_seed: parseInt(document.getElementById('video-seed').value) || -1,
  };
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(getSettings()),
    });
  }, 600);
}

// 変更イベントで自動保存をスケジュール
['positive-prompt', 'negative-prompt',
  'seed-input', 'comfyui-seed', 'video-seed',
  'image-save-path', 'video-save-path',
  'model-dropdown', 'sampler-dropdown', 'comfyui-workflow', 'video-workflow',
].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', scheduleSave);
});

document.querySelectorAll('input[name="video-section"]').forEach(cb => {
  cb.addEventListener('change', scheduleSave);
});

document.getElementById('unload-llm-before-video').addEventListener('change', scheduleSave);

async function loadSettings() {
  const s = await fetch('/api/settings').then(r => r.json());

  // Model
  const modelDd = document.getElementById('model-dropdown');
  if (modelDd.querySelector(`option[value="${s.model}"]`)) {
    modelDd.value = s.model;
  }

  // Forge params
  document.getElementById('steps-slider').value = s.steps ?? 28;
  document.getElementById('steps-val').textContent = s.steps ?? 28;
  document.getElementById('cfg-slider').value = s.cfg ?? 7;
  document.getElementById('cfg-val').textContent = parseFloat(s.cfg ?? 7).toFixed(1);
  document.getElementById('width-slider').value = s.width ?? 512;
  document.getElementById('width-val').textContent = s.width ?? 512;
  document.getElementById('height-slider').value = s.height ?? 768;
  document.getElementById('height-val').textContent = s.height ?? 768;
  document.getElementById('seed-input').value = s.seed ?? -1;

  // Sampler
  const samplerDd = document.getElementById('sampler-dropdown');
  if (samplerDd.querySelector(`option[value="${s.sampler}"]`)) {
    samplerDd.value = s.sampler;
  }

  // ComfyUI params
  document.getElementById('comfyui-width').value = s.comfyui_width ?? 1024;
  document.getElementById('comfyui-width-val').textContent = s.comfyui_width ?? 1024;
  document.getElementById('comfyui-height').value = s.comfyui_height ?? 1024;
  document.getElementById('comfyui-height-val').textContent = s.comfyui_height ?? 1024;
  document.getElementById('comfyui-seed').value = s.comfyui_seed ?? -1;

  // ComfyUI workflow
  const wfDd = document.getElementById('comfyui-workflow');
  if (s.comfyui_workflow && wfDd.querySelector(`option[value="${s.comfyui_workflow}"]`)) {
    wfDd.value = s.comfyui_workflow;
  }

  // Backend
  const backendVal = s.backend === 'Forge 2' ? 'WebUI Forge' : (s.backend || 'WebUI Forge');
  const radioEl = document.querySelector(`input[name="backend"][value="${backendVal}"]`);
  if (radioEl) radioEl.checked = true;
  updateBackendVisibility();

  // Save image path
  document.getElementById('image-save-path').value = s.image_save_path || './outputs/images';

  // Video workflow
  const vwfDd = document.getElementById('video-workflow');
  if (s.video_workflow && vwfDd.querySelector(`option[value="${s.video_workflow}"]`)) {
    vwfDd.value = s.video_workflow;
  }

  // Video params
  document.getElementById('video-width').value = s.video_width ?? 848;
  document.getElementById('video-width-val').textContent = s.video_width ?? 848;
  document.getElementById('video-height').value = s.video_height ?? 480;
  document.getElementById('video-height-val').textContent = s.video_height ?? 480;
  document.getElementById('video-seed').value = s.video_seed ?? -1;

  // Save video path
  document.getElementById('video-save-path').value = s.video_save_path || './outputs/videos';

  // Unload LLM before video
  document.getElementById('unload-llm-before-video').checked = !!s.unload_llm_before_video;

  // Video sections
  if (Array.isArray(s.video_sections)) {
    document.querySelectorAll('input[name="video-section"]').forEach(cb => {
      cb.checked = s.video_sections.includes(cb.value);
    });
  }
}

// ---------------------------------------------------------------------------
// ドロップダウン初期化
// ---------------------------------------------------------------------------

async function initDropdowns() {
  // モデル一覧
  const presetsResp = await fetch('/api/model_presets').then(r => r.json());
  const modelDd = document.getElementById('model-dropdown');
  modelDd.innerHTML = '';
  for (const preset of presetsResp.presets) {
    const opt = document.createElement('option');
    opt.value = preset;
    opt.textContent = preset;
    modelDd.appendChild(opt);
  }

  // サンプラー一覧
  const samplersResp = await fetch('/api/samplers').then(r => r.json());
  const samplerDd = document.getElementById('sampler-dropdown');
  samplerDd.innerHTML = '';
  for (const s of samplersResp.samplers) {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    samplerDd.appendChild(opt);
  }

  // 画像ワークフロー
  const wfResp = await fetch('/api/workflows').then(r => r.json());
  const wfDd = document.getElementById('comfyui-workflow');
  wfDd.innerHTML = '';
  for (const w of wfResp.workflows) {
    const opt = document.createElement('option');
    opt.value = w;
    opt.textContent = w;
    wfDd.appendChild(opt);
  }

  // 動画ワークフロー
  const vwfResp = await fetch('/api/video_workflows').then(r => r.json());
  const vwfDd = document.getElementById('video-workflow');
  vwfDd.innerHTML = '';
  for (const w of vwfResp.workflows) {
    const opt = document.createElement('option');
    opt.value = w;
    opt.textContent = w;
    vwfDd.appendChild(opt);
  }
}

// ---------------------------------------------------------------------------
// 画像アップロード
// ---------------------------------------------------------------------------

const imageDropArea = document.getElementById('image-drop-area');
const imageFileInput = document.getElementById('image-file-input');
const imageDisplay = document.getElementById('image-display');
const imagePlaceholder = document.getElementById('image-placeholder');
const imageStatus = document.getElementById('image-status');

function showImage(src) {
  imageDisplay.src = src;
  imageDisplay.style.display = 'block';
  imagePlaceholder.style.display = 'none';
}

function setSavePathsFromFilePath(filePath) {
  if (!filePath) return;
  const normalized = filePath.replace(/\\/g, '/');
  const idx = normalized.lastIndexOf('/');
  if (idx < 0) return;
  const dir = filePath.slice(0, idx);
  document.getElementById('image-save-path').value = dir;
  document.getElementById('video-save-path').value = dir;
  scheduleSave();
}

imageDropArea.addEventListener('click', () => imageFileInput.click());

imageDropArea.addEventListener('dragover', e => {
  e.preventDefault();
  imageDropArea.classList.add('dragover');
});

imageDropArea.addEventListener('dragleave', () => {
  imageDropArea.classList.remove('dragover');
});

imageDropArea.addEventListener('drop', async e => {
  e.preventDefault();
  imageDropArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const filePath = window.electronAPI?.getPathForFile?.(file) || '';
  setSavePathsFromFilePath(filePath);
  if (file.type.startsWith('image/')) {
    await uploadImage(file);
    return;
  }
  if (file.name.toLowerCase().endsWith('.json')) {
    await uploadJson(file);
  }
});

imageFileInput.addEventListener('change', async () => {
  const file = imageFileInput.files[0];
  if (file) await uploadImage(file);
  imageFileInput.value = '';
});

async function uploadImage(file) {
  imageStatus.textContent = 'アップロード中...';
  const formData = new FormData();
  formData.append('file', file);
  const filePath = window.electronAPI?.getPathForFile?.(file) || '';
  if (filePath) formData.append('image_path', filePath);
  try {
    const resp = await fetch('/api/image/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    showImage(data.image);
    imageStatus.textContent = data.status;
    if (filePath) {
      document.getElementById('image-file-path').textContent = filePath;
    }
    if (data.meta) {
      applyMetadata(data.meta);
    }
    if (data.saved_json) {
      applySavedJson(data.saved_json);
    }
  } catch (e) {
    imageStatus.textContent = `エラー: ${e}`;
  }
}

async function uploadJson(file) {
  imageStatus.textContent = 'JSON 読み込み中...';
  const formData = new FormData();
  formData.append('file', file);
  try {
    const resp = await fetch('/api/json/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    if (!data.ok) {
      imageStatus.textContent = data.message || 'JSON の読み込みに失敗しました';
      return;
    }
    showImage(data.image);
    imageStatus.textContent = data.status;
    if (data.image_path) {
      document.getElementById('image-file-path').textContent = data.image_path;
    }
    if (data.meta) {
      applyMetadata(data.meta);
    }
    if (data.saved_json) {
      applySavedJson(data.saved_json);
    }
  } catch (e) {
    imageStatus.textContent = `エラー: ${e}`;
  }
}

function applyMetadata(meta) {
  if (meta.positive != null) document.getElementById('positive-prompt').value = meta.positive;
  if (meta.negative != null) document.getElementById('negative-prompt').value = meta.negative;
  if (meta.steps != null) {
    document.getElementById('steps-slider').value = meta.steps;
    document.getElementById('steps-val').textContent = meta.steps;
  }
  if (meta.cfg_scale != null) {
    document.getElementById('cfg-slider').value = meta.cfg_scale;
    document.getElementById('cfg-val').textContent = parseFloat(meta.cfg_scale).toFixed(1);
  }
  if (meta.sampler != null) {
    const dd = document.getElementById('sampler-dropdown');
    if (dd.querySelector(`option[value="${meta.sampler}"]`)) dd.value = meta.sampler;
  }
  if (meta.width != null) {
    document.getElementById('width-slider').value = meta.width;
    document.getElementById('width-val').textContent = meta.width;
  }
  if (meta.height != null) {
    document.getElementById('height-slider').value = meta.height;
    document.getElementById('height-val').textContent = meta.height;
  }
  if (meta.seed != null) {
    document.getElementById('seed-input').value = meta.seed;
    document.getElementById('comfyui-seed').value = meta.seed;
  }
  scheduleSave();
}

function applySavedJson(data) {
  if (data.prompt != null) document.getElementById('video-prompt').value = data.prompt;
  if (data.additional_instruction != null) {
    document.getElementById('video-extra-instruction').value = data.additional_instruction;
  }

  if (data.comfyui_workflow != null) {
    const dd = document.getElementById('comfyui-workflow');
    if (dd.querySelector(`option[value="${data.comfyui_workflow}"]`)) {
      dd.value = data.comfyui_workflow;
    }
  }

  if (data.video_workflow != null) {
    const dd = document.getElementById('video-workflow');
    if (dd.querySelector(`option[value="${data.video_workflow}"]`)) {
      dd.value = data.video_workflow;
    }
  }

  scheduleSave();
}

// Seed ボタン
document.getElementById('seed-random-btn').addEventListener('click', () => {
  document.getElementById('seed-input').value = -1;
  scheduleSave();
});

document.getElementById('comfyui-seed-random-btn').addEventListener('click', () => {
  document.getElementById('comfyui-seed').value = -1;
  scheduleSave();
});

async function seedFromImage(targetId) {
  const resp = await fetch('/api/seed_from_image', { method: 'POST' }).then(r => r.json());
  if (resp.ok) {
    document.getElementById(targetId).value = resp.seed;
    imageStatus.textContent = resp.message;
    scheduleSave();
  } else {
    imageStatus.textContent = resp.message;
  }
}

document.getElementById('seed-from-image-btn').addEventListener('click',
  () => seedFromImage('seed-input'));
document.getElementById('comfyui-seed-from-image-btn').addEventListener('click',
  () => seedFromImage('comfyui-seed'));

document.getElementById('video-seed-random-btn').addEventListener('click', () => {
  document.getElementById('video-seed').value = -1;
  scheduleSave();
});

async function seedFromVideo() {
  const resp = await fetch('/api/seed_from_video', { method: 'POST' }).then(r => r.json());
  const videoStatus = document.getElementById('video-status');
  if (resp.ok) {
    document.getElementById('video-seed').value = resp.seed;
    if (videoStatus) videoStatus.textContent = resp.message;
    scheduleSave();
  } else {
    if (videoStatus) videoStatus.textContent = resp.message;
  }
}

document.getElementById('video-seed-from-video-btn').addEventListener('click', seedFromVideo);

// ---------------------------------------------------------------------------
// 統合生成キュー（画像・動画共通）
// ---------------------------------------------------------------------------

let genQueue = [];       // { type: 'image'|'video', params: {} }
let genProcessing = false;
let genAbortCtrl = null;

function getQueueLabel() {
  const n = genQueue.length;
  return n > 0 ? `（キュー: ${n}件）` : '';
}

function setImageBusy(busy) {
  document.getElementById('generate-btn').disabled = false; // 常に押せる
  document.getElementById('stop-btn').disabled = !busy;
}

function setVideoBusy(busy) {
  document.getElementById('generate-video-btn').disabled = false;
  document.getElementById('stop-video-btn').disabled = !busy;
  document.querySelector('.video-area').classList.toggle('generating', busy);
}

async function processNextJob() {
  if (genQueue.length === 0) {
    genProcessing = false;
    setImageBusy(false);
    setVideoBusy(false);
    return;
  }

  genProcessing = true;
  genAbortCtrl = new AbortController();
  const job = genQueue.shift();

  if (job.type === 'image') {
    setImageBusy(true);
    setVideoBusy(genQueue.length > 0);
    try {
      await fetchSSE('/api/generate_image/stream', job.params, data => {
        const q = getQueueLabel();
        if (data.type === 'status') {
          imageStatus.textContent = data.content + q;
        } else if (data.type === 'image') {
          showImage(data.image);
          imageStatus.textContent = data.status + q;
          if (data.saved_path) document.getElementById('image-file-path').textContent = data.saved_path;
        } else if (data.type === 'error') {
          imageStatus.textContent = data.content + q;
        }
      }, genAbortCtrl.signal);
    } catch (e) { /* abort */ }

  } else if (job.type === 'video') {
    setVideoBusy(true);
    setImageBusy(genQueue.length > 0);
    try {
      await fetchSSE('/api/generate_video/stream', job.params, data => {
        const q = getQueueLabel();
        if (data.type === 'status') {
          videoStatus.textContent = data.content + q;
        } else if (data.type === 'video') {
          videoDisplay.src = data.url;
          videoDisplay.style.display = 'block';
          videoPlaceholder.style.display = 'none';
          videoStatus.textContent = data.status + q;
          if (data.saved_path) document.getElementById('video-file-path').textContent = data.saved_path;
        } else if (data.type === 'error') {
          videoStatus.textContent = data.content + q;
        }
      }, genAbortCtrl.signal);
    } catch (e) { /* abort */ }
  }

  genAbortCtrl = null;
  processNextJob();
}

function stopAllGeneration() {
  genQueue = [];
  if (genAbortCtrl) genAbortCtrl.abort();
  fetch('/api/interrupt_image', { method: 'POST' });
  fetch('/api/stop_video', { method: 'POST' });
  genProcessing = false;
  setImageBusy(false);
  setVideoBusy(false);
}

// ---------------------------------------------------------------------------
// 画像生成
// ---------------------------------------------------------------------------

document.getElementById('generate-btn').addEventListener('click', () => {
  const backend = document.querySelector('input[name="backend"]:checked')?.value;
  genQueue.push({ type: 'image', params: {
    positive: document.getElementById('positive-prompt').value,
    negative: document.getElementById('negative-prompt').value,
    steps: parseInt(document.getElementById('steps-slider').value),
    cfg: parseFloat(document.getElementById('cfg-slider').value),
    sampler: document.getElementById('sampler-dropdown').value,
    width: parseInt(document.getElementById('width-slider').value),
    height: parseInt(document.getElementById('height-slider').value),
    seed: parseInt(document.getElementById('seed-input').value) || -1,
    backend,
    comfyui_workflow: document.getElementById('comfyui-workflow').value,
    comfyui_width: parseInt(document.getElementById('comfyui-width').value),
    comfyui_height: parseInt(document.getElementById('comfyui-height').value),
    comfyui_seed: parseInt(document.getElementById('comfyui-seed').value) || -1,
    image_save_path: document.getElementById('image-save-path').value,
  }});
  if (!genProcessing) processNextJob();
  else imageStatus.textContent = `待機中...${getQueueLabel()}`;
});

document.getElementById('stop-btn').addEventListener('click', () => {
  stopAllGeneration();
  imageStatus.textContent = '停止しました';
});

// ---------------------------------------------------------------------------
// チャット
// ---------------------------------------------------------------------------

const chatbot = document.getElementById('chatbot');
const userInput = document.getElementById('user-input');

document.getElementById('send-btn').addEventListener('click', sendChat);
document.getElementById('clear-btn').addEventListener('click', clearChat);

userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});

function appendMessage(role, content, id) {
  const div = document.createElement('div');
  div.className = `chat-message ${role}`;
  if (id) div.id = id;
  const roleLabel = document.createElement('div');
  roleLabel.className = 'chat-role';
  roleLabel.textContent = role === 'user' ? 'あなた' : 'Qwen3-VL';
  const contentDiv = document.createElement('div');
  contentDiv.className = 'chat-content';
  contentDiv.textContent = content;
  div.appendChild(roleLabel);
  div.appendChild(contentDiv);
  chatbot.appendChild(div);
  chatbot.scrollTop = chatbot.scrollHeight;
  return contentDiv;
}

async function sendChat() {
  const text = userInput.value.trim();
  if (!text) return;

  if (chatAbortCtrl) chatAbortCtrl.abort();
  chatAbortCtrl = new AbortController();

  userInput.value = '';
  document.getElementById('send-btn').disabled = true;

  appendMessage('user', text);
  const assistantContent = appendMessage('assistant', '');

  const msgId = 'msg-' + Date.now();
  assistantContent.parentElement.id = msgId;

  let partial = '';

  try {
    await fetchSSE('/api/chat/stream', {
      user_input: text,
      model_label: document.getElementById('model-dropdown').value,
      positive: document.getElementById('positive-prompt').value,
      negative: document.getElementById('negative-prompt').value,
    }, data => {
      if (data.type === 'token') {
        partial += data.content;
        assistantContent.textContent = partial;
        chatbot.scrollTop = chatbot.scrollHeight;
      } else if (data.type === 'model_loaded') {
        document.getElementById('model-status').textContent = data.message;
      } else if (data.type === 'done') {
        if (data.display_text != null) {
          assistantContent.textContent = data.display_text;
        }
        if (data.positive != null) {
          document.getElementById('positive-prompt').value = data.positive;
          scheduleSave();
        }
        if (data.negative != null) {
          document.getElementById('negative-prompt').value = data.negative;
          scheduleSave();
        }
      } else if (data.type === 'error') {
        assistantContent.textContent = `エラー: ${data.content}`;
      }
    }, chatAbortCtrl.signal);
  } finally {
    document.getElementById('send-btn').disabled = false;
    chatAbortCtrl = null;
  }
}

async function clearChat() {
  await fetch('/api/chat/clear', { method: 'POST' });
  chatbot.innerHTML = '';
}

// ---------------------------------------------------------------------------
// モデルロード
// ---------------------------------------------------------------------------

document.getElementById('load-model-btn').addEventListener('click', async () => {
  const btn = document.getElementById('load-model-btn');
  btn.disabled = true;
  document.getElementById('model-status').textContent = 'ロード中...';
  try {
    const resp = await fetch('/api/load_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_label: document.getElementById('model-dropdown').value }),
    }).then(r => r.json());
    document.getElementById('model-status').textContent = resp.message;
  } finally {
    btn.disabled = false;
  }
});

document.getElementById('model-dropdown').addEventListener('change', scheduleSave);

// ---------------------------------------------------------------------------
// VRAM
// ---------------------------------------------------------------------------

document.getElementById('free-qwen-btn').addEventListener('click', async () => {
  const resp = await fetch('/api/unload_qwen', { method: 'POST' }).then(r => r.json());
  document.getElementById('vram-status').textContent = resp.message;
});

document.getElementById('free-comfy-btn').addEventListener('click', async () => {
  const resp = await fetch('/api/free_comfyui', { method: 'POST' }).then(r => r.json());
  document.getElementById('vram-status').textContent = resp.message;
});

// ---------------------------------------------------------------------------
// 動画プロンプト生成
// ---------------------------------------------------------------------------

document.getElementById('generate-video-prompt-btn').addEventListener('click', generateVideoPrompt);

async function generateVideoPrompt() {
  if (videoPromptAbortCtrl) videoPromptAbortCtrl.abort();
  videoPromptAbortCtrl = new AbortController();

  const btn = document.getElementById('generate-video-prompt-btn');
  btn.disabled = true;

  const sections = Array.from(
    document.querySelectorAll('input[name="video-section"]:checked')
  ).map(el => el.value);

  document.getElementById('video-prompt').value = '';
  let accumulated = '';

  try {
    await fetchSSE('/api/video_prompt/stream', {
      positive: document.getElementById('positive-prompt').value,
      extra_instruction: document.getElementById('video-extra-instruction').value,
      sections,
      model_label: document.getElementById('model-dropdown').value,
    }, data => {
      if (data.type === 'token') {
        accumulated += data.content;
        document.getElementById('video-prompt').value = accumulated;
      } else if (data.type === 'status') {
        document.getElementById('video-status').textContent = data.content;
      } else if (data.type === 'error') {
        document.getElementById('video-status').textContent = `エラー: ${data.content}`;
      }
    }, videoPromptAbortCtrl.signal);
  } finally {
    btn.disabled = false;
    videoPromptAbortCtrl = null;
  }
}

// ---------------------------------------------------------------------------
// 動画生成
// ---------------------------------------------------------------------------

const videoDisplay = document.getElementById('video-display');
const videoPlaceholder = document.getElementById('video-placeholder');
const videoStatus = document.getElementById('video-status');

document.getElementById('generate-video-btn').addEventListener('click', () => {
  genQueue.push({ type: 'video', params: {
    video_prompt: document.getElementById('video-prompt').value,
    workflow: document.getElementById('video-workflow').value,
    seed: parseInt(document.getElementById('video-seed').value) || -1,
    width: parseInt(document.getElementById('video-width').value),
    height: parseInt(document.getElementById('video-height').value),
    video_save_path: document.getElementById('video-save-path').value,
    unload_llm_before_video: document.getElementById('unload-llm-before-video').checked,
  }});
  if (!genProcessing) processNextJob();
  else videoStatus.textContent = `待機中...${getQueueLabel()}`;
});

document.getElementById('stop-video-btn').addEventListener('click', () => {
  if (videoPromptAbortCtrl) videoPromptAbortCtrl.abort();
  stopAllGeneration();
  videoStatus.textContent = '停止しました';
});

// ---------------------------------------------------------------------------
// 動画解像度プリセット
// ---------------------------------------------------------------------------

document.getElementById('video-res-640').addEventListener('click', () => {
  document.getElementById('video-width').value = 640;
  document.getElementById('video-width-val').textContent = 640;
  document.getElementById('video-height').value = 480;
  document.getElementById('video-height-val').textContent = 480;
  scheduleSave();
});

document.getElementById('video-res-1280').addEventListener('click', () => {
  document.getElementById('video-width').value = 1280;
  document.getElementById('video-width-val').textContent = 1280;
  document.getElementById('video-height').value = 720;
  document.getElementById('video-height-val').textContent = 720;
  scheduleSave();
});

// ---------------------------------------------------------------------------
// パス表示のフォルダアイコン
// ---------------------------------------------------------------------------

document.getElementById('open-image-path-btn').addEventListener('click', () => {
  const p = document.getElementById('image-file-path').textContent
    || document.getElementById('image-save-path').value;
  if (p) fetch('/api/open_path', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: p }) });
});

document.getElementById('open-video-path-btn').addEventListener('click', () => {
  const p = document.getElementById('video-file-path').textContent
    || document.getElementById('video-save-path').value;
  if (p) fetch('/api/open_path', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: p }) });
});

// ---------------------------------------------------------------------------
// ワークフローフォルダを開く
// ---------------------------------------------------------------------------

document.getElementById('open-image-workflow-folder-btn').addEventListener('click', () => {
  fetch('/api/open_workflow_folder', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind: 'image' }),
  });
});

document.getElementById('open-video-workflow-folder-btn').addEventListener('click', () => {
  fetch('/api/open_workflow_folder', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind: 'video' }),
  });
});

// ---------------------------------------------------------------------------
// JSON保存
// ---------------------------------------------------------------------------

document.getElementById('save-json-btn').addEventListener('click', async () => {
  const statusEl = document.getElementById('save-json-status');
  statusEl.textContent = '保存中...';
  const resp = await fetch('/api/save_json', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_prompt: document.getElementById('video-prompt').value,
      additional_instruction: document.getElementById('video-extra-instruction').value,
      comfyui_workflow: document.getElementById('comfyui-workflow').value,
      video_workflow: document.getElementById('video-workflow').value,
    }),
  }).then(r => r.json());
  statusEl.textContent = resp.message;
});

// ---------------------------------------------------------------------------
// 上下パネルのリサイズ
// ---------------------------------------------------------------------------

function initPanelResizer() {
  const imageBlock = document.getElementById('tab-image');
  const videoBlock = document.getElementById('tab-video');
  const resizer = document.getElementById('panel-resizer');
  if (!imageBlock || !videoBlock || !resizer) return;

  const minPanelHeight = 180;

  function applyHeights(topHeight, bottomHeight) {
    imageBlock.style.flex = `0 0 ${topHeight}px`;
    imageBlock.style.flexBasis = `${topHeight}px`;
    videoBlock.style.flex = `0 0 ${bottomHeight}px`;
    videoBlock.style.flexBasis = `${bottomHeight}px`;
  }

  function saveHeights(topHeight, bottomHeight) {
    try {
      localStorage.setItem(PANEL_LAYOUT_KEY, JSON.stringify({ topHeight, bottomHeight }));
    } catch (_) { /* ignore storage errors */ }
  }

  function clampHeights(topHeight) {
    const totalHeight = window.innerHeight - resizer.offsetHeight;
    const maxTopHeight = Math.max(minPanelHeight, totalHeight - minPanelHeight);
    const clampedTopHeight = Math.min(Math.max(topHeight, minPanelHeight), maxTopHeight);
    return {
      topHeight: clampedTopHeight,
      bottomHeight: totalHeight - clampedTopHeight,
    };
  }

  function restoreHeights() {
    try {
      const raw = localStorage.getItem(PANEL_LAYOUT_KEY);
      if (!raw) return false;
      const parsed = JSON.parse(raw);
      if (!Number.isFinite(parsed.topHeight)) return false;
      const { topHeight, bottomHeight } = clampHeights(parsed.topHeight);
      applyHeights(topHeight, bottomHeight);
      saveHeights(topHeight, bottomHeight);
      return Number.isFinite(parsed.bottomHeight) || Number.isFinite(bottomHeight);
    } catch (_) {
      return false;
    }
  }

  function syncToViewport() {
    const totalHeight = window.innerHeight - resizer.offsetHeight;
    const currentTop = imageBlock.getBoundingClientRect().height;
    const nextTop = currentTop > 0 ? currentTop : totalHeight / 2;
    const { topHeight, bottomHeight } = clampHeights(nextTop);
    applyHeights(topHeight, bottomHeight);
  }

  let startY = 0;
  let startTopHeight = 0;

  resizer.addEventListener('pointerdown', event => {
    startY = event.clientY;
    startTopHeight = imageBlock.getBoundingClientRect().height;
    resizer.classList.add('dragging');
    document.body.classList.add('is-resizing');
    resizer.setPointerCapture(event.pointerId);
  });

  resizer.addEventListener('pointermove', event => {
    if (!resizer.classList.contains('dragging')) return;
    const delta = event.clientY - startY;
    const { topHeight, bottomHeight } = clampHeights(startTopHeight + delta);
    applyHeights(topHeight, bottomHeight);
  });

  function finishResize(event) {
    if (!resizer.classList.contains('dragging')) return;
    const { topHeight, bottomHeight } = clampHeights(imageBlock.getBoundingClientRect().height);
    applyHeights(topHeight, bottomHeight);
    saveHeights(topHeight, bottomHeight);
    resizer.classList.remove('dragging');
    document.body.classList.remove('is-resizing');
    if (event?.pointerId != null && resizer.hasPointerCapture(event.pointerId)) {
      resizer.releasePointerCapture(event.pointerId);
    }
  }

  resizer.addEventListener('pointerup', finishResize);
  resizer.addEventListener('pointercancel', finishResize);
  window.addEventListener('resize', () => {
    syncToViewport();
    const topHeight = imageBlock.getBoundingClientRect().height;
    const bottomHeight = videoBlock.getBoundingClientRect().height;
    saveHeights(topHeight, bottomHeight);
  });

  if (!restoreHeights()) {
    syncToViewport();
  }
}

// ---------------------------------------------------------------------------
// 初期化
// ---------------------------------------------------------------------------

async function init() {
  try {
    await initDropdowns();
    await loadSettings();
  } catch (e) {
    console.error('初期化エラー:', e);
  }

  // 停止ボタンは初期無効
  document.getElementById('stop-btn').disabled = true;
  document.getElementById('stop-video-btn').disabled = true;

  updateBackendVisibility();
  initPanelResizer();
}

init();

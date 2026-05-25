const MUSIC_EXAMPLES = [
  "An 80s driving pop song with heavy drums and synth pads in the background",
  "A cheerful country song with acoustic guitars",
  "90s rock song with electric guitar and heavy drums",
  "A light and cheerful EDM track with syncopated drums and airy pads",
  "Lo-fi slow BPM electro chill with organic samples",
];

const SOUND_EXAMPLES = [
  "Dog barking in a suburban backyard",
  "Siren of an emergency vehicle approaching",
  "Footsteps echoing in a long corridor",
  "Rain falling on a tin roof",
  "Coffee machine brewing and steaming",
];

const MODEL_LABELS = {
  "facebook/musicgen-small": "MusicGen Small",
  "facebook/musicgen-medium": "MusicGen Medium",
  "facebook/musicgen-melody": "MusicGen Melody",
  "facebook/musicgen-stereo-small": "MusicGen Stereo Small",
  "facebook/musicgen-stereo-medium": "MusicGen Stereo Medium",
  "facebook/musicgen-stereo-melody": "MusicGen Stereo Melody",
  "facebook/audiogen-medium": "AudioGen Medium",
};

let mode = "music";
let musicModels = [];
let soundModels = [];
let pollTimer = null;

const els = {
  deviceBadge: document.getElementById("deviceBadge"),
  tabs: document.querySelectorAll(".tab"),
  form: document.getElementById("generateForm"),
  prompt: document.getElementById("prompt"),
  model: document.getElementById("model"),
  duration: document.getElementById("duration"),
  durationVal: document.getElementById("durationVal"),
  melodySection: document.getElementById("melodySection"),
  melody: document.getElementById("melody"),
  melodyDrop: document.getElementById("melodyDrop"),
  melodyName: document.getElementById("melodyName"),
  useDiffusion: document.getElementById("useDiffusion"),
  temperature: document.getElementById("temperature"),
  topK: document.getElementById("topK"),
  topP: document.getElementById("topP"),
  cfgCoef: document.getElementById("cfgCoef"),
  generateBtn: document.getElementById("generateBtn"),
  exampleChips: document.getElementById("exampleChips"),
  statusPill: document.getElementById("statusPill"),
  progressWrap: document.getElementById("progressWrap"),
  progressFill: document.getElementById("progressFill"),
  progressMsg: document.getElementById("progressMsg"),
  player: document.getElementById("player"),
  downloadLink: document.getElementById("downloadLink"),
  outputActions: document.getElementById("outputActions"),
  errorBox: document.getElementById("errorBox"),
  waveform: document.getElementById("waveform"),
  waveformPlaceholder: document.getElementById("waveformPlaceholder"),
};

function labelForModel(id) {
  return MODEL_LABELS[id] || id.replace("facebook/", "");
}

function populateModels() {
  const models = mode === "music" ? musicModels : soundModels;
  els.model.innerHTML = models
    .map((m) => `<option value="${m}">${labelForModel(m)}</option>`)
    .join("");
  if (mode === "music" && models.includes("facebook/musicgen-stereo-melody")) {
    els.model.value = "facebook/musicgen-stereo-melody";
  }
}

function renderExamples() {
  const examples = mode === "music" ? MUSIC_EXAMPLES : SOUND_EXAMPLES;
  els.exampleChips.innerHTML = examples
    .map(
      (text) =>
        `<button type="button" class="chip" data-text="${encodeURIComponent(text)}">${text.slice(0, 42)}${text.length > 42 ? "…" : ""}</button>`
    )
    .join("");
  els.exampleChips.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.prompt.value = decodeURIComponent(chip.dataset.text);
    });
  });
}

function setMode(nextMode) {
  mode = nextMode;
  els.tabs.forEach((tab) => {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  els.melodySection.classList.toggle("hidden", mode !== "music");
  populateModels();
  renderExamples();
}

function setStatus(state, text) {
  els.statusPill.className = `status-pill ${state}`;
  els.statusPill.textContent = text;
}

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.classList.remove("hidden");
}

function clearError() {
  els.errorBox.classList.add("hidden");
  els.errorBox.textContent = "";
}

function resetOutput() {
  els.player.classList.add("hidden");
  els.player.removeAttribute("src");
  els.outputActions.classList.add("hidden");
  els.waveform.classList.remove("visible");
  els.waveformPlaceholder.classList.remove("hidden");
  clearError();
}

async function drawWaveform(url) {
  try {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await ctx.decodeAudioData(buffer);
    const data = audioBuffer.getChannelData(0);
    const canvas = els.waveform;
    const c = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    c.clearRect(0, 0, width, height);

    const step = Math.ceil(data.length / width);
    const amp = height / 2;
    c.fillStyle = "#e8a849";
    for (let i = 0; i < width; i++) {
      let min = 1;
      let max = -1;
      for (let j = 0; j < step; j++) {
        const v = data[i * step + j] || 0;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const y1 = (1 + min) * amp;
      const y2 = (1 + max) * amp;
      c.fillRect(i, y1, 1, Math.max(1, y2 - y1));
    }
    els.waveform.classList.add("visible");
    els.waveformPlaceholder.classList.add("hidden");
    ctx.close();
  } catch {
    // Waveform is optional; player still works.
  }
}

function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) throw new Error("Failed to fetch job status");
      const data = await res.json();

      els.progressWrap.classList.remove("hidden");
      els.progressFill.style.width = `${Math.round(data.progress * 100)}%`;
      els.progressMsg.textContent = data.message;

      if (data.status === "complete" && data.audio_url) {
        clearInterval(pollTimer);
        pollTimer = null;
        setStatus("done", "Complete");
        els.progressWrap.classList.add("hidden");
        els.generateBtn.disabled = false;

        els.player.src = data.audio_url;
        els.player.classList.remove("hidden");
        els.player.play().catch(() => {});

        els.downloadLink.href = data.audio_url;
        els.downloadLink.download = `audiocraft-${jobId.slice(0, 8)}.wav`;
        els.outputActions.classList.remove("hidden");

        drawWaveform(data.audio_url);
      } else if (data.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        setStatus("error", "Failed");
        els.progressWrap.classList.add("hidden");
        els.generateBtn.disabled = false;
        showError(data.error || "Unknown error during generation.");
      } else {
        setStatus("working", data.status === "loading" ? "Loading" : "Generating");
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      setStatus("error", "Failed");
      els.generateBtn.disabled = false;
      showError(err.message);
    }
  }, 800);
}

async function submitMusic(formData) {
  const res = await fetch("/api/generate/music", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || "Music generation request failed");
  }
  return res.json();
}

async function submitSound(payload) {
  const res = await fetch("/api/generate/sound", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || "Sound generation request failed");
  }
  return res.json();
}

els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});

els.duration.addEventListener("input", () => {
  els.durationVal.textContent = `${els.duration.value}s`;
});

els.melodyDrop.addEventListener("dragover", (e) => {
  e.preventDefault();
  els.melodyDrop.classList.add("dragover");
});

els.melodyDrop.addEventListener("dragleave", () => {
  els.melodyDrop.classList.remove("dragover");
});

els.melodyDrop.addEventListener("drop", (e) => {
  e.preventDefault();
  els.melodyDrop.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    els.melody.files = e.dataTransfer.files;
    els.melodyName.textContent = e.dataTransfer.files[0].name;
  }
});

els.melody.addEventListener("change", () => {
  els.melodyName.textContent = els.melody.files[0]?.name || "";
});

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resetOutput();
  els.generateBtn.disabled = true;
  setStatus("working", "Starting");
  els.progressWrap.classList.remove("hidden");
  els.progressFill.style.width = "2%";
  els.progressMsg.textContent = "Submitting…";

  try {
    let result;
    if (mode === "music") {
      const formData = new FormData();
      formData.append("prompt", els.prompt.value.trim());
      formData.append("model", els.model.value);
      formData.append("duration", els.duration.value);
      formData.append("top_k", els.topK.value);
      formData.append("top_p", els.topP.value);
      formData.append("temperature", els.temperature.value);
      formData.append("cfg_coef", els.cfgCoef.value);
      formData.append("use_diffusion", els.useDiffusion.checked ? "true" : "false");
      if (els.melody.files[0]) {
        formData.append("melody", els.melody.files[0]);
      }
      result = await submitMusic(formData);
    } else {
      result = await submitSound({
        prompt: els.prompt.value.trim(),
        model: els.model.value,
        duration: Number(els.duration.value),
        top_k: Number(els.topK.value),
        top_p: Number(els.topP.value),
        temperature: Number(els.temperature.value),
        cfg_coef: Number(els.cfgCoef.value),
      });
    }
    pollJob(result.job_id);
  } catch (err) {
    setStatus("error", "Failed");
    els.progressWrap.classList.add("hidden");
    els.generateBtn.disabled = false;
    showError(err.message);
  }
});

async function init() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    musicModels = data.music_models || [];
    soundModels = data.sound_models || [];
    const device = data.device || "cpu";
    els.deviceBadge.textContent = device.toUpperCase();
    if (device !== "cpu") els.deviceBadge.classList.add("gpu");
  } catch {
    musicModels = [
      "facebook/musicgen-stereo-melody",
      "facebook/musicgen-stereo-medium",
      "facebook/musicgen-stereo-small",
    ];
    soundModels = ["facebook/audiogen-medium"];
    els.deviceBadge.textContent = "OFFLINE";
  }
  setMode("music");
}

init();

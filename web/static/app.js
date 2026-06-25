const statusEls = {
  clock: document.getElementById("clock"),
  currentTask: document.getElementById("currentTask"),
  currentStep: document.getElementById("currentStep"),
  runState: document.getElementById("runState"),
  frame: document.getElementById("frame"),
  emptyFrame: document.getElementById("emptyFrame"),
  stepDetect: document.getElementById("stepDetect"),
  stepCoord: document.getElementById("stepCoord"),
  stepGrab: document.getElementById("stepGrab"),
  stepPlace: document.getElementById("stepPlace"),
  micToggle: document.getElementById("micToggle"),
  micLabel: document.getElementById("micLabel"),
  logList: document.getElementById("logList"),
};

const taskNames = {
  color_stack: "色块堆叠",
  color_sort: "色块分拣",
  component_sort: "元器件分拣",
};

let selectedMode = "component_sort";
let lastFrameMtime = 0;
let micActive = false;

async function post(url) {
  const res = await fetch(url, { method: "POST" });
  const data = await res.json();
  await refreshStatus();
  return data;
}

function setSelectedMode(key) {
  selectedMode = key;
  document.querySelectorAll(".nav-item").forEach(button => {
    button.classList.toggle("active", button.dataset.mode === key);
  });
  document.querySelectorAll("[data-camera-mode]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.cameraMode === key));
  });
  if (statusEls.runState.textContent !== "运行中") {
    statusEls.currentTask.textContent = taskNames[key] || "待机";
  }
}

function setActiveSteps(running) {
  const active = running.component_sort || running.color_stack || running.color_sort;
  for (const el of [statusEls.stepDetect, statusEls.stepCoord, statusEls.stepGrab, statusEls.stepPlace]) {
    if (el) {
      el.classList.toggle("active", active);
    }
  }
}

function renderLogs(events) {
  const items = Array.isArray(events) ? events.filter(item => item && item.message) : [];
  if (!items.length) {
    statusEls.logList.innerHTML = '<div class="log-empty">等待任务日志</div>';
    return;
  }
  statusEls.logList.innerHTML = items.slice(-18).map(item => {
    const time = String(item.time || "--").replace(/[<>&"]/g, "");
    const message = String(item.message || "").replace(/[<>&"]/g, "");
    return `<div class="log-item"><b>${time}</b><span>${message}</span></div>`;
  }).join("");
  statusEls.logList.scrollTop = statusEls.logList.scrollHeight;
}

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  statusEls.clock.textContent = data.timestamp;
  const anyRunning = Object.values(data.running).some(Boolean);
  statusEls.currentTask.textContent = anyRunning ? data.current : (taskNames[selectedMode] || "待机");
  statusEls.runState.textContent = anyRunning ? "运行中" : "空闲";
  statusEls.currentStep.textContent = anyRunning ? "任务执行" : "等待指令";
  setActiveSteps(data.running);
  renderLogs(data.events);

  if (data.image) {
    const frameMtime = data.image_mtime || Date.now();
    if (frameMtime !== lastFrameMtime) {
      lastFrameMtime = frameMtime;
      statusEls.frame.src = `/api/frame?t=${encodeURIComponent(frameMtime)}`;
    }
    statusEls.frame.style.display = "block";
    statusEls.emptyFrame.style.display = "none";
  } else {
    statusEls.frame.style.display = "none";
    statusEls.emptyFrame.style.display = "grid";
  }
}

document.querySelectorAll(".nav-item").forEach(button => {
  button.addEventListener("click", () => {
    setSelectedMode(button.dataset.mode);
  });
});

document.querySelectorAll("[data-camera-mode]").forEach(button => {
  button.addEventListener("click", () => {
    setSelectedMode(button.dataset.cameraMode);
  });
});

document.querySelectorAll("[data-action]").forEach(button => {
  button.addEventListener("click", async () => {
    const action = button.dataset.action;
    const key = button.dataset.key;
    await post(`/api/${action}/${key}`);
  });
});

const startSelectedButton = document.getElementById("startSelected");
if (startSelectedButton) {
  startSelectedButton.addEventListener("click", async () => {
    await post(`/api/start/${selectedMode}`);
  });
}

document.getElementById("topStartSelected").addEventListener("click", async () => {
  await post(`/api/start/${selectedMode}`);
});

const resetArmButton = document.getElementById("resetArm");
if (resetArmButton) {
  resetArmButton.addEventListener("click", async () => {
    await post("/api/reset");
  });
}

document.getElementById("topResetArm").addEventListener("click", async () => {
  await post("/api/reset");
});

const emergencyButton = document.getElementById("emergency");
if (emergencyButton) {
  emergencyButton.addEventListener("click", async () => {
    await post("/api/emergency");
  });
}

document.getElementById("topEmergency").addEventListener("click", async () => {
  await post("/api/emergency");
});

statusEls.micToggle.addEventListener("click", () => {
  micActive = !micActive;
  statusEls.micToggle.classList.toggle("active", micActive);
  statusEls.micToggle.setAttribute("aria-pressed", String(micActive));
  statusEls.micLabel.textContent = micActive ? "语音监听中" : "语音待机";
});

setSelectedMode(selectedMode);
refreshStatus();
setInterval(refreshStatus, 500);

console.log("app.js loaded");

/* =========================================================
   GLOBAL STATE
========================================================= */

let detector = null;
let modelReady = false;
let modelLoading = false;

let currentMode = "normal";
const MIN_CONFIDENCE = 0.4;

/* =========================================================
   DOM ELEMENTS (MATCH index.html)
========================================================= */

const normalModeBtn = document.getElementById("normalModeBtn");
const advanceModeBtn = document.getElementById("advanceModeBtn");

const normalSection = document.getElementById("normalUpload");
const advanceSection = document.getElementById("advanceUpload");

const normalInput = document.getElementById("normalInput");
const frontInput = document.getElementById("frontInput");
const sideInput = document.getElementById("sideInput");

const normalPreviewImg = document.getElementById("normalPreviewImg");
const frontPreviewImg = document.getElementById("frontPreviewImg");
const sidePreviewImg = document.getElementById("sidePreviewImg");

const runBtn = document.getElementById("runBtn");
const statusText = document.getElementById("statusText");

const canvas = document.getElementById("poseCanvas");
const ctx = canvas.getContext("2d");

/* =========================================================
   FAIL FAST IF HTML CHANGES
========================================================= */

[
  normalModeBtn, advanceModeBtn,
  normalSection, advanceSection,
  normalInput, frontInput, sideInput,
  normalPreviewImg, frontPreviewImg, sidePreviewImg,
  runBtn, statusText, canvas
].forEach(el => {
  if (!el) throw new Error("Missing DOM element. Check index.html IDs.");
});

/* =========================================================
   STATUS
========================================================= */

function setStatus(msg) {
  statusText.textContent = msg;
}

/* =========================================================
   MODE MANAGEMENT
========================================================= */

function setMode(mode) {
  currentMode = mode;

  normalSection.style.display = mode === "normal" ? "flex" : "none";
  advanceSection.style.display = mode === "advance" ? "flex" : "none";

  normalModeBtn.style.background =
    mode === "normal" ? "var(--accent)" : "#fff";
  normalModeBtn.style.color =
    mode === "normal" ? "#fff" : "#000";

  advanceModeBtn.style.background =
    mode === "advance" ? "var(--accent)" : "#fff";
  advanceModeBtn.style.color =
    mode === "advance" ? "#fff" : "#000";

  checkRunAvailability();
}

normalModeBtn.onclick = () => setMode("normal");
advanceModeBtn.onclick = () => setMode("advance");

/* =========================================================
   RUN BUTTON LOGIC
========================================================= */

function hasFile(input) {
  return input.files && input.files.length > 0;
}

function checkRunAvailability() {
  if (currentMode === "normal") {
    runBtn.disabled = !hasFile(normalInput);
  } else {
    runBtn.disabled = !(hasFile(frontInput) && hasFile(sideInput));
  }
}

normalInput.onchange = () => {
  previewFile(normalInput, normalPreviewImg);
  checkRunAvailability();
};

frontInput.onchange = () => {
  previewFile(frontInput, frontPreviewImg);
  checkRunAvailability();
};

sideInput.onchange = () => {
  previewFile(sideInput, sidePreviewImg);
  checkRunAvailability();
};

/* =========================================================
   FILE PREVIEW (IMAGE OR VIDEO)
========================================================= */

function previewFile(input, imgEl) {
  const file = input.files[0];
  if (!file) return;

  if (file.type.startsWith("image")) {
    imgEl.src = URL.createObjectURL(file);
  } else if (file.type.startsWith("video")) {
    imgEl.src = "examples/video_placeholder.png"; // optional
  }
}

/* =========================================================
   MODEL LOADING
========================================================= */

async function loadModel() {
  if (modelReady || modelLoading) return;

  modelLoading = true;
  setStatus("Loading pose model...");

  try {
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER }
    );
    modelReady = true;
    setStatus("Model ready");
  } catch (err) {
    console.error(err);
    setStatus("Model failed to load");
  } finally {
    modelLoading = false;
  }
}

loadModel();

/* =========================================================
   POSE HELPERS
========================================================= */

function getJoint(keypoints, index) {
  const kp = keypoints[index];
  if (!kp || kp.score < MIN_CONFIDENCE) return null;
  return kp;
}

function calculateAngle(a, b, c) {
  if (!a || !b || !c) return "N/A";

  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const magAB = Math.hypot(ab.x, ab.y);
  const magCB = Math.hypot(cb.x, cb.y);

  if (!magAB || !magCB) return "N/A";

  const dot = ab.x * cb.x + ab.y * cb.y;
  const angle = Math.acos(dot / (magAB * magCB)) * (180 / Math.PI);

  return isNaN(angle) ? "N/A" : angle.toFixed(1);
}

/* =========================================================
   DRAWING (MATCH PREVIEW BOX SIZE)
========================================================= */

function drawToCanvas(sourceEl, keypoints) {
  const rect = sourceEl.getBoundingClientRect();

  canvas.width = rect.width;
  canvas.height = rect.height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(sourceEl, 0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "red";

  keypoints.forEach(kp => {
    if (kp.score >= MIN_CONFIDENCE) {
      ctx.beginPath();
      ctx.arc(
        kp.x / sourceEl.naturalWidth * canvas.width,
        kp.y / sourceEl.naturalHeight * canvas.height,
        4, 0, Math.PI * 2
      );
      ctx.fill();
    }
  });
}

/* =========================================================
   IMAGE ANALYSIS
========================================================= */

async function analyzeImage(imgEl) {
  if (!modelReady) return setStatus("Model not ready");

  setStatus("Analyzing posture...");

  const poses = await detector.estimatePoses(imgEl);
  if (!poses.length) return setStatus("No person detected");

  const keypoints = poses[0].keypoints;
  drawToCanvas(imgEl, keypoints);

  const shoulder = getJoint(keypoints, 5);
  const elbow = getJoint(keypoints, 7);
  const wrist = getJoint(keypoints, 9);

  const angle = calculateAngle(shoulder, elbow, wrist);

  setStatus(
    angle === "N/A"
      ? "Elbow angle: N/A (low confidence)"
      : `Elbow angle: ${angle}°`
  );
}

/* =========================================================
   VIDEO ANALYSIS (MIDDLE FRAME)
========================================================= */

async function analyzeVideo(file, previewImg) {
  const video = document.createElement("video");
  video.src = URL.createObjectURL(file);
  video.muted = true;

  await video.play();
  video.pause();

  video.currentTime = video.duration / 2;

  await new Promise(r => video.onseeked = r);

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;

  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(video, 0, 0);

  const img = new Image();
  img.src = tempCanvas.toDataURL();
  img.onload = () => analyzeImage(img);
}

/* =========================================================
   RUN BUTTON
========================================================= */

runBtn.onclick = () => {
  if (currentMode === "normal") {
    const file = normalInput.files[0];
    if (!file) return;

    if (file.type.startsWith("image")) {
      analyzeImage(normalPreviewImg);
    } else if (file.type.startsWith("video")) {
      analyzeVideo(file, normalPreviewImg);
    }
  }
};

/* =========================================================
   INIT
========================================================= */

setMode("normal");

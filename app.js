/* =========================================================
   BOOT CHECK
========================================================= */
console.log("app.js loaded");

/* =========================================================
   GLOBAL STATE
========================================================= */

let detector = null;
let modelReady = false;
let modelLoading = false;

let currentMode = "normal"; // "normal" | "advance"
const MIN_CONFIDENCE = 0.4;

/* =========================================================
   DOM ELEMENTS (MATCH index.html EXACTLY)
========================================================= */

// mode buttons
const normalModeBtn = document.getElementById("normalModeBtn");
const advanceModeBtn = document.getElementById("advanceModeBtn");

// sections
const normalSection = document.getElementById("normalUpload");
const advanceSection = document.getElementById("advanceUpload");

// inputs
const normalInput = document.getElementById("normalInput");
const frontInput = document.getElementById("frontInput");
const sideInput = document.getElementById("sideInput");

// run + status
const runBtn = document.getElementById("runBtn");
const statusText = document.getElementById("statusText");

// canvas
const canvas = document.getElementById("poseCanvas");
const ctx = canvas.getContext("2d");

/* =========================================================
   HARD DOM VALIDATION (FAIL FAST)
========================================================= */

const required = [
  normalModeBtn, advanceModeBtn,
  normalSection, advanceSection,
  normalInput, frontInput, sideInput,
  runBtn, statusText, canvas
];

if (required.some(el => el === null)) {
  throw new Error("Critical DOM element missing. Check index.html IDs.");
}

/* =========================================================
   STATUS HELPERS
========================================================= */

function setStatus(msg) {
  statusText.textContent = msg;
}

/* =========================================================
   MODE MANAGEMENT (SINGLE AUTHORITY)
========================================================= */

function setMode(mode) {
  if (mode !== "normal" && mode !== "advance") return;

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
   RUN BUTTON ENABLE LOGIC
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

normalInput.onchange = checkRunAvailability;
frontInput.onchange = checkRunAvailability;
sideInput.onchange = checkRunAvailability;

/* =========================================================
   MODEL LOADING (ONE TIME, GUARDED)
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
   SAFE POSE UTILITIES
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

  if (magAB === 0 || magCB === 0) return "N/A";

  const dot = ab.x * cb.x + ab.y * cb.y;
  const angle = Math.acos(dot / (magAB * magCB)) * (180 / Math.PI);

  return isNaN(angle) ? "N/A" : angle.toFixed(1);
}

/* =========================================================
   DRAWING
========================================================= */

function drawKeypoints(keypoints) {
  ctx.fillStyle = "red";
  keypoints.forEach(kp => {
    if (kp.score >= MIN_CONFIDENCE) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

/* =========================================================
   CORE ANALYSIS
========================================================= */

async function analyzeImage(img) {
  if (!modelReady) {
    setStatus("Model not ready");
    return;
  }

  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  setStatus("Analyzing posture...");

  try {
    const poses = await detector.estimatePoses(img);
    if (!poses.length) {
      setStatus("No person detected");
      return;
    }

    const keypoints = poses[0].keypoints;
    drawKeypoints(keypoints);

    // example: left elbow
    const shoulder = getJoint(keypoints, 5);
    const elbow = getJoint(keypoints, 7);
    const wrist = getJoint(keypoints, 9);

    const angle = calculateAngle(shoulder, elbow, wrist);

    setStatus(
      angle === "N/A"
        ? "Elbow angle: N/A (low confidence)"
        : `Elbow angle: ${angle}°`
    );

  } catch (err) {
    console.error(err);
    setStatus("Analysis failed");
  }
}

/* =========================================================
   RUN BUTTON
========================================================= */

runBtn.onclick = () => {
  if (currentMode === "normal" && hasFile(normalInput)) {
    const img = new Image();
    img.src = URL.createObjectURL(normalInput.files[0]);
    img.onload = () => analyzeImage(img);
  }
};

/* =========================================================
   INITIAL STATE
========================================================= */

setMode("normal");

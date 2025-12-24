console.log("app.js loaded");
/* =========================================================
   GLOBAL STATE
========================================================= */

let detector = null;
let modelLoading = false;
let modelReady = false;

let currentMode = "normal"; // "normal" | "advanced"

const MIN_CONFIDENCE = 0.4;

/* =========================================================
   DOM ELEMENTS (DECLARE ONCE)
========================================================= */

const statusText = document.getElementById("statusText");

const normalModeBtn = document.getElementById("normalModeBtn");
const advancedModeBtn = document.getElementById("advancedModeBtn");

const normalSection = document.getElementById("normalSection");
const advancedSection = document.getElementById("advancedSection");

const normalUpload = document.getElementById("normalUpload");
const advancedFrontUpload = document.getElementById("advancedFrontUpload");
const advancedSideUpload = document.getElementById("advancedSideUpload");

const canvas = document.getElementById("poseCanvas");
const ctx = canvas.getContext("2d");

/* =========================================================
   STATUS & UI HELPERS
========================================================= */

function showStatus(message) {
  if (statusText) statusText.textContent = message;
}

function resetResults() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  showStatus("Ready");
}

/* =========================================================
   MODE MANAGEMENT (SINGLE SOURCE OF TRUTH)
========================================================= */

function setMode(mode) {
  if (mode !== "normal" && mode !== "advanced") return;

  currentMode = mode;

  normalSection.style.display = mode === "normal" ? "block" : "none";
  advancedSection.style.display = mode === "advanced" ? "block" : "none";

  resetResults();
}

normalModeBtn.onclick = () => setMode("normal");
advancedModeBtn.onclick = () => setMode("advanced");

/* =========================================================
   MODEL LOADING (ONE-TIME, GUARDED)
========================================================= */

async function loadPoseModel() {
  if (modelReady || modelLoading) return;

  modelLoading = true;
  showStatus("Loading pose model...");

  try {
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER }
    );
    modelReady = true;
    showStatus("Model ready");
  } catch (err) {
    console.error("Model load failed:", err);
    showStatus("Failed to load pose model");
  } finally {
    modelLoading = false;
  }
}

loadPoseModel();

/* =========================================================
   SAFE POSE UTILITIES
========================================================= */

function getJoint(keypoints, index) {
  const kp = keypoints[index];
  if (!kp || kp.score < MIN_CONFIDENCE) return null;
  return kp;
}

function calculateAngleSafe(a, b, c) {
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
   DRAWING (CONFIDENCE-AWARE)
========================================================= */

function drawSkeleton(keypoints) {
  ctx.fillStyle = "red";
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;

  keypoints.forEach(kp => {
    if (kp.score >= MIN_CONFIDENCE) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

/* =========================================================
   CORE ANALYSIS (SAFE ENTRY)
========================================================= */

async function analyzeImage(image) {
  if (!modelReady) {
    showStatus("Model not ready");
    return;
  }

  showStatus("Analyzing posture...");

  try {
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);

    const poses = await detector.estimatePoses(image);

    if (!poses || poses.length === 0) {
      showStatus("No person detected");
      return;
    }

    const keypoints = poses[0].keypoints;
    drawSkeleton(keypoints);

    // Example: left elbow
    const shoulder = getJoint(keypoints, 5);
    const elbow = getJoint(keypoints, 7);
    const wrist = getJoint(keypoints, 9);

    const elbowAngle = calculateAngleSafe(shoulder, elbow, wrist);

    showStatus(
      elbowAngle === "N/A"
        ? "Elbow angle: N/A (low confidence)"
        : `Elbow angle: ${elbowAngle}°`
    );

  } catch (err) {
    console.error("Analysis failed:", err);
    showStatus("Analysis error");
  }
}

/* =========================================================
   FILE INPUT HANDLERS (HARDENED)
========================================================= */

normalUpload.onchange = () => {
  if (!normalUpload.files || normalUpload.files.length === 0) return;

  const img = new Image();
  img.src = URL.createObjectURL(normalUpload.files[0]);

  img.onload = () => analyzeImage(img);
};

advancedFrontUpload.onchange = () => {
  if (!advancedFrontUpload.files.length) return;
  // Placeholder for future advanced logic
};

advancedSideUpload.onchange = () => {
  if (!advancedSideUpload.files.length) return;
  // Placeholder for future advanced logic
};

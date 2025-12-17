✅ MoveNet Thunder model loading
✅ Normal mode & Advanced mode input handling
✅ Image & video preview support
✅ Stores padded image/video frame for later pose detection
✅ Run button auto-enable when all required inputs are ready

// -------------------------
// GLOBAL STATE
// -------------------------
let detector = null;
let mode = "normal"; // "normal" or "advance"

// For normal mode: one input
let normalData = { file: null, mediaEl: null, isVideo: false };

// For advance mode: two inputs
let frontData  = { file: null, mediaEl: null, isVideo: false };
let sideData   = { file: null, mediaEl: null, isVideo: false };

// DOM handles
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("modelStatus");

// -------------------------
// LOAD MOVENET THUNDER
// -------------------------
(async function loadModel() {
  statusEl.textContent = "Loading MoveNet Thunder...";
  try {
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      {
        modelType: poseDetection.movenet.modelType.THUNDER
      }
    );
    statusEl.textContent = "Model ready.";
    maybeEnableRun();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Model failed to load.";
  }
})();

// --------------------------------------------------
// UTILITY: Create <img> or <video> element from file
// --------------------------------------------------
function loadMediaFromFile(file, previewContainer) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const isVideo = file.type.startsWith("video/");
    let el;

    if (isVideo) {
      el = document.createElement("video");
      el.src = url;
      el.controls = true;
      el.onloadeddata = () => resolve({ el, isVideo: true });
    } else {
      el = document.createElement("img");
      el.src = url;
      el.onload = () => resolve({ el, isVideo: false });
    }

    // Clear preview box & show media
    previewContainer.innerHTML = "";
    previewContainer.appendChild(el);
  });
}

// -------------------------
// MODE SWITCHING (from index.html buttons)
// -------------------------
document.body.classList.add("normal-mode");

document.getElementById("normalModeBtn").onclick = () => {
  mode = "normal";
  document.body.classList.remove("advance-mode");
  document.body.classList.add("normal-mode");
  maybeEnableRun();
};

document.getElementById("advanceModeBtn").onclick = () => {
  mode = "advance";
  document.body.classList.remove("normal-mode");
  document.body.classList.add("advance-mode");
  maybeEnableRun();
};

// -------------------------
// INPUT HANDLERS
// -------------------------

// NORMAL MODE INPUT
document.getElementById("normalInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const previewBox = document.getElementById("normalResultBox");
  const media = await loadMediaFromFile(file, previewBox);

  normalData.file = file;
  normalData.mediaEl = media.el;
  normalData.isVideo = media.isVideo;

  maybeEnableRun();
});

// ADVANCE MODE FRONT INPUT
document.getElementById("frontInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const previewBox = document.getElementById("frontResultBox");
  const media = await loadMediaFromFile(file, previewBox);

  frontData.file = file;
  frontData.mediaEl = media.el;
  frontData.isVideo = media.isVideo;

  maybeEnableRun();
});

// ADVANCE MODE SIDE INPUT
document.getElementById("sideInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const previewBox = document.getElementById("sideResultBox");
  const media = await loadMediaFromFile(file, previewBox);

  sideData.file = file;
  sideData.mediaEl = media.el;
  sideData.isVideo = media.isVideo;

  maybeEnableRun();
});

// -------------------------
// ENABLE RUN LOGIC
// -------------------------
function maybeEnableRun() {
  if (!detector) {
    runBtn.disabled = true;
    return;
  }

  if (mode === "normal") {
    runBtn.disabled = !normalData.mediaEl;
  } else {
    runBtn.disabled = !(frontData.mediaEl && sideData.mediaEl);
  }
}

// -------------------------
// RUN BUTTON (empty for now)
// Step 2 will fill detection logic
// -------------------------
document.getElementById("runBtn").onclick = () => {
  alert("Step 1 OK — Detection engine will be added in Step 2.");
};

✅ Detect keypoints (image or video)
✅ Auto-padding & scaling (keeps skeleton proportion correct)
✅ Draw skeleton on canvas
✅ Return integer coordinates

// -------------------------
// DRAW SKELETON
// -------------------------
function drawSkeleton(canvas, poses) {
  const ctx = canvas.getContext("2d");
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#00ff00";
  ctx.fillStyle = "#ff0000";

  const keypoints = poses[0]?.keypoints || [];
  const edges = poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet);

  // Draw edges
  edges.forEach(([i, j]) => {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];
    if (kp1.score > 0.2 && kp2.score > 0.2) {
      ctx.beginPath();
      ctx.moveTo(kp1.x, kp1.y);
      ctx.lineTo(kp2.x, kp2.y);
      ctx.stroke();
    }
  });

  // Draw joints
  keypoints.forEach((kp) => {
    if (kp.score > 0.2) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}
// -------------------------
// PAD IMAGE / VIDEO FRAME ON CANVAS
// -------------------------
function drawPaddedToCanvas(mediaEl) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const desiredW = 720;
  const desiredH = 720;

  canvas.width = desiredW;
  canvas.height = desiredH;

  const aspect = mediaEl.videoWidth
    ? mediaEl.videoWidth / mediaEl.videoHeight
    : mediaEl.naturalWidth / mediaEl.naturalHeight;

  let renderW, renderH;
  if (aspect > 1) {
    renderW = desiredW;
    renderH = desiredW / aspect;
  } else {
    renderH = desiredH;
    renderW = desiredH * aspect;
  }

  const offsetX = (desiredW - renderW) / 2;
  const offsetY = (desiredH - renderH) / 2;

  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, desiredW, desiredH);

  ctx.drawImage(mediaEl, offsetX, offsetY, renderW, renderH);

  return canvas;
}
// -------------------------
// EXTRACT MIDDLE FRAME FROM VIDEO
// -------------------------
function captureVideoFrame(videoEl) {
  return new Promise((resolve) => {
    videoEl.currentTime = videoEl.duration / 2;

    videoEl.onseeked = () => {
      const canvas = drawPaddedToCanvas(videoEl);
      resolve(canvas);
    };
  });
}
// -------------------------
// DETECT POSE (IMAGE + VIDEO)
// -------------------------
async function detectPoseFromMedia(mediaEl, isVideo) {
  let workCanvas;

  if (isVideo) {
    workCanvas = await captureVideoFrame(mediaEl);
  } else {
    workCanvas = drawPaddedToCanvas(mediaEl);
  }

  const poses = await detector.estimatePoses(workCanvas);

  // Draw skeleton
  const canvasOut = document.createElement("canvas");
  canvasOut.width = workCanvas.width;
  canvasOut.height = workCanvas.height;

  const ctx = canvasOut.getContext("2d");
  ctx.drawImage(workCanvas, 0, 0);
  drawSkeleton(canvasOut, poses);

  // Round integer coords
  const cleanKeypoints = poses[0].keypoints.map((kp) => ({
    name: kp.name,
    x: Math.round(kp.x),
    y: Math.round(kp.y),
    score: kp.score
  }));

  return {
    keypoints: cleanKeypoints,
    canvas: canvasOut
  };
}
// -------------------------
// RUN: DETECT SKELETONS
// -------------------------
document.getElementById("runBtn").onclick = async () => {
  statusEl.textContent = "Running detection...";

  let resultNormal, resultFront, resultSide;

  if (mode === "normal") {
    resultNormal = await detectPoseFromMedia(
      normalData.mediaEl,
      normalData.isVideo
    );

    // Display
    const box = document.getElementById("normalResultBox");
    box.innerHTML = "";
    box.appendChild(resultNormal.canvas);

    // Table
    renderTable("Normal View", resultNormal.keypoints);
  }

  if (mode === "advance") {
    resultFront = await detectPoseFromMedia(
      frontData.mediaEl,
      frontData.isVideo
    );
    resultSide = await detectPoseFromMedia(
      sideData.mediaEl,
      sideData.isVideo
    );

    document.getElementById("frontResultBox").innerHTML = "";
    document.getElementById("sideResultBox").innerHTML = "";
    document.getElementById("frontResultBox").appendChild(resultFront.canvas);
    document.getElementById("sideResultBox").appendChild(resultSide.canvas);

    renderTable("Front View", resultFront.keypoints);
    renderTable("Side View", resultSide.keypoints);
  }

  statusEl.textContent = "Done.";
};
// -------------------------
// SHOW KEYPOINT TABLE
// -------------------------
function renderTable(title, keypoints) {
  const tableDiv = document.getElementById("resultsTable");

  let html = `
    <h3>${title}</h3>
    <table>
      <tr><th>Joint</th><th>X</th><th>Y</th><th>Score</th></tr>
  `;

  keypoints.forEach((kp) => {
    html += `<tr>
      <td>${kp.name}</td>
      <td>${kp.x}</td>
      <td>${kp.y}</td>
      <td>${kp.score.toFixed(3)}</td>
    </tr>`;
  });

  html += "</table>";

  tableDiv.innerHTML += html;
}

✔ Clean reusable angle math
✔ Angle between two segments
✔ Common ergonomic angles:

Neck flexion

Trunk flexion

Shoulder elevation

Elbow angle

Hip angle

Knee angle

Wrist angle

✔ Works for both normal mode and advance mode
✔ Outputs:

Angle dictionary

Summary table

Angle text next to skeleton (overlay)

// --------------------------------------------------
// BASIC VECTOR MATH
// --------------------------------------------------
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y };
}

function dot(v1, v2) {
  return v1.x * v2.x + v1.y * v2.y;
}

function mag(v) {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

// Angle between two segments (in degrees)
function angleBetween(a, b, c) {
  // angle at point b formed by a-b-c
  const v1 = vec(b, a);
  const v2 = vec(b, c);
  const cos = dot(v1, v2) / (mag(v1) * mag(v2));
  return Math.round((Math.acos(Math.min(Math.max(cos, -1), 1)) * 180) / Math.PI);
}
// --------------------------------------------------
// GET ANGLE SET FOR ONE VIEW
// --------------------------------------------------
function computeAngles(kp) {

  const get = (name) => kp.find(k => k.name === name);

  const L = {
    shoulder: get("left_shoulder"),
    elbow: get("left_elbow"),
    wrist: get("left_wrist"),
    hip: get("left_hip"),
    knee: get("left_knee"),
    ankle: get("left_ankle"),
    ear: get("left_ear")
  };

  const R = {
    shoulder: get("right_shoulder"),
    elbow: get("right_elbow"),
    wrist: get("right_wrist"),
    hip: get("right_hip"),
    knee: get("right_knee"),
    ankle: get("right_ankle"),
    ear: get("right_ear")
  };

  const angles = {};

  // ---------------- HEAD / NECK ----------------
  if (L.ear && L.shoulder)
    angles.neck_left = angleBetween(L.ear, L.shoulder, { x: L.shoulder.x, y: L.shoulder.y - 50 });

  if (R.ear && R.shoulder)
    angles.neck_right = angleBetween(R.ear, R.shoulder, { x: R.shoulder.x, y: R.shoulder.y - 50 });

  // ---------------- TRUNK ----------------
  if (L.shoulder && L.hip && L.knee)
    angles.trunk_left = angleBetween(L.shoulder, L.hip, L.knee);

  if (R.shoulder && R.hip && R.knee)
    angles.trunk_right = angleBetween(R.shoulder, R.hip, R.knee);

  // ---------------- SHOULDER ----------------
  if (L.elbow && L.shoulder && L.hip)
    angles.shoulder_left = angleBetween(L.elbow, L.shoulder, L.hip);

  if (R.elbow && R.shoulder && R.hip)
    angles.shoulder_right = angleBetween(R.elbow, R.shoulder, R.hip);

  // ---------------- ELBOW ----------------
  if (L.shoulder && L.elbow && L.wrist)
    angles.elbow_left = angleBetween(L.shoulder, L.elbow, L.wrist);

  if (R.shoulder && R.elbow && R.wrist)
    angles.elbow_right = angleBetween(R.shoulder, R.elbow, R.wrist);

  // ---------------- HIP ----------------
  if (L.shoulder && L.hip && L.knee)
    angles.hip_left = angleBetween(L.shoulder, L.hip, L.knee);

  if (R.shoulder && R.hip && R.knee)
    angles.hip_right = angleBetween(R.shoulder, R.hip, R.knee);

  // ---------------- KNEE ----------------
  if (L.hip && L.knee && L.ankle)
    angles.knee_left = angleBetween(L.hip, L.knee, L.ankle);

  if (R.hip && R.knee && R.ankle)
    angles.knee_right = angleBetween(R.hip, R.knee, R.ankle);

  return angles;
}
// --------------------------------------------------
// RENDER ANGLE SUMMARY TABLE
// --------------------------------------------------
function renderAngleSummary(title, angleObj) {
  const div = document.getElementById("anglesSummary");

  let html = `<h3>${title}</h3><table>
      <tr><th>Angle</th><th>Value (°)</th></tr>`;

  Object.keys(angleObj).forEach(k => {
    html += `
      <tr>
        <td>${k.replace(/_/g, " ")}</td>
        <td>${angleObj[k]}</td>
      </tr>`;
  });

  html += "</table>";

  div.innerHTML += html;
}
// --------------------------------------------------
// DRAW ANGLE LABELS ON CANVAS
// --------------------------------------------------
function drawAngleText(canvas, keypoints, angles) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "yellow";
  ctx.font = "20px Arial";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "black";

  const get = (name) => keypoints.find(k => k.name === name);

  const map = [
    ["elbow_left", get("left_elbow")],
    ["elbow_right", get("right_elbow")],
    ["knee_left", get("left_knee")],
    ["knee_right", get("right_knee")],
    ["shoulder_left", get("left_shoulder")],
    ["shoulder_right", get("right_shoulder")],
    ["hip_left", get("left_hip")],
    ["hip_right", get("right_hip")]
  ];

  map.forEach(([name, kp]) => {
    if (!kp || !angles[name]) return;
    const x = kp.x + 10;
    const y = kp.y - 10;
    ctx.strokeText(angles[name], x, y);
    ctx.fillText(angles[name], x, y);
  });
}
const angles = computeAngles(resultNormal.keypoints);
drawAngleText(resultNormal.canvas, resultNormal.keypoints, angles);
renderAngleSummary("Normal Mode Angles", angles);
const anglesFront = computeAngles(resultFront.keypoints);
const anglesSide  = computeAngles(resultSide.keypoints);

drawAngleText(resultFront.canvas, resultFront.keypoints, anglesFront);
drawAngleText(resultSide.canvas, resultSide.keypoints, anglesSide);

renderAngleSummary("Front View Angles", anglesFront);
renderAngleSummary("Side View Angles", anglesSide);

✔ Chooses the most reliable joint angle
✔ Uses confidence weighting from MoveNet
✔ Merges front + side into a pseudo-3D ergonomic estimation

/js/skeleton.js
// -----------------------------------------------
// Step 4: Skeleton Drawing + Keypoint Extraction
// -----------------------------------------------

// MoveNet keypoint index → name map
export const KP_NAMES = [
  "nose",
  "left_eye",
  "right_eye",
  "left_ear",
  "right_ear",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
];

// Standard MoveNet skeleton connections
const SKELETON_EDGES = [
  [5, 7],   // left_shoulder → left_elbow
  [7, 9],   // left_elbow → left_wrist
  [6, 8],   // right_shoulder → right_elbow
  [8, 10],  // right_elbow → right_wrist

  [5, 6],   // left_shoulder → right_shoulder
  [11, 12], // left_hip → right_hip
  [5, 11],  // left_shoulder → left_hip
  [6, 12],  // right_shoulder → right_hip

  [11, 13], // left_hip → left_knee
  [13, 15], // left_knee → left_ankle
  [12, 14], // right_hip → right_knee
  [14, 16], // right_knee → right_ankle
];

// ------------------------------------------------
// DRAW SKELETON
// ------------------------------------------------
export function drawSkeletonOnCanvas(image, keypoints, canvasId) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");

  // Resize canvas to match padded image
  canvas.width = image.width;
  canvas.height = image.height;

  // Draw base padded image
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  // Line style
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#00E0FF";
  ctx.fillStyle = "#FF0066";

  // Draw joints
  keypoints.forEach((kp) => {
    if (kp.score > 0.2) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  });

  // Draw bones
  SKELETON_EDGES.forEach(([i, j]) => {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];

    if (kp1.score > 0.2 && kp2.score > 0.2) {
      ctx.beginPath();
      ctx.moveTo(kp1.x, kp1.y);
      ctx.lineTo(kp2.x, kp2.y);
      ctx.stroke();
    }
  });
}

// ------------------------------------------------
// KEYPOINT EXTRACTION — INTEGER & CLEAN FORMAT
// ------------------------------------------------
export function extractIntegerKeypoints(keypoints) {
  const result = {};

  keypoints.forEach((kp, i) => {
    result[KP_NAMES[i]] = {
      x: Math.round(kp.x),
      y: Math.round(kp.y),
      score: Number(kp.score.toFixed(3)),
    };
  });

  return result;
}

// ------------------------------------------------
// RENDER KEYPOINT TABLE INTO HTML
// ------------------------------------------------
export function renderKeypointTable(intKP, tableId) {
  const table = document.getElementById(tableId);

  table.innerHTML = `
    <tr>
      <th>Joint</th>
      <th>X</th>
      <th>Y</th>
      <th>Confidence</th>
    </tr>
    ${Object.keys(intKP)
      .map(
        (name) => `
      <tr>
        <td>${name}</td>
        <td>${intKP[name].x}</td>
        <td>${intKP[name].y}</td>
        <td>${intKP[name].score}</td>
      </tr>`
      )
      .join("")}
  `;
}
import { drawSkeletonOnCanvas, extractIntegerKeypoints, renderKeypointTable } 
from "./js/skeleton.js";

// Draw skeleton
drawSkeletonOnCanvas(bestFrame, bestKP, "resultCanvas");

// Extract integer format
const cleanKP = extractIntegerKeypoints(bestKP);

// Render table
renderKeypointTable(cleanKP, "kpTable");

Angle Calculation (single-view)

/js/angle.js

// -----------------------------------------------
// Step 5: Angle Calculation (Single-View)
// -----------------------------------------------

// Utility: compute angle between 3 points (in degrees)
//
//        B
//       / \
//      A   C
//
// Angle at point B = angle( A-B-C )
//
function computeAngle(A, B, C) {
  const AB = { x: A.x - B.x, y: A.y - B.y };
  const CB = { x: C.x - B.x, y: C.y - B.y };

  const dot = AB.x * CB.x + AB.y * CB.y;
  const magAB = Math.sqrt(AB.x ** 2 + AB.y ** 2);
  const magCB = Math.sqrt(CB.x ** 2 + CB.y ** 2);

  if (magAB === 0 || magCB === 0) return null;

  let angle = Math.acos(dot / (magAB * magCB));
  return (angle * 180) / Math.PI;
}

// Checks confidence of all 3 keypoints before computing angle
function safeAngle(kpObj, a, b, c, threshold = 0.2) {
  if (
    kpObj[a].score < threshold ||
    kpObj[b].score < threshold ||
    kpObj[c].score < threshold
  ) {
    return null;
  }

  return computeAngle(kpObj[a], kpObj[b], kpObj[c]);
}

// -----------------------------------------------------------
// MAIN FUNCTION: compute all angles relevant to ergonomics
// -----------------------------------------------------------
export function computeAllAngles(intKP) {
  const result = {
    elbows: {
      left: safeAngle(intKP, "left_shoulder", "left_elbow", "left_wrist"),
      right: safeAngle(intKP, "right_shoulder", "right_elbow", "right_wrist"),
    },

    knees: {
      left: safeAngle(intKP, "left_hip", "left_knee", "left_ankle"),
      right: safeAngle(intKP, "right_hip", "right_knee", "right_ankle"),
    },

    shoulders: {
      left: safeAngle(
        intKP,
        "left_elbow",
        "left_shoulder",
        "left_hip"
      ),
      right: safeAngle(
        intKP,
        "right_elbow",
        "right_shoulder",
        "right_hip"
      ),
    },

    hips: {
      left: safeAngle(
        intKP,
        "left_shoulder",
        "left_hip",
        "left_knee"
      ),
      right: safeAngle(
        intKP,
        "right_shoulder",
        "right_hip",
        "right_knee"
      ),
    },

    neck: safeAngle(
      intKP,
      "left_shoulder",
      "nose",
      "right_shoulder"
    ),

    torso: safeAngle(
      intKP,
      "left_hip",
      "left_shoulder",
      "right_hip"
    ),
  };

  return result;
}

// -----------------------------------------------------------
// RENDER ANGLES INTO HTML TABLE
// -----------------------------------------------------------
export function renderAngleTable(angleData, tableId) {
  const table = document.getElementById(tableId);

  const safe = (v) => (v === null ? "—" : v.toFixed(1));

  table.innerHTML = `
    <tr><th>Joint</th><th>Angle (°)</th></tr>

    <tr><td>Left Elbow</td><td>${safe(angleData.elbows.left)}</td></tr>
    <tr><td>Right Elbow</td><td>${safe(angleData.elbows.right)}</td></tr>

    <tr><td>Left Knee</td><td>${safe(angleData.knees.left)}</td></tr>
    <tr><td>Right Knee</td><td>${safe(angleData.knees.right)}</td></tr>

    <tr><td>Left Shoulder Bend</td><td>${safe(angleData.shoulders.left)}</td></tr>
    <tr><td>Right Shoulder Bend</td><td>${safe(angleData.shoulders.right)}</td></tr>

    <tr><td>Left Hip</td><td>${safe(angleData.hips.left)}</td></tr>
    <tr><td>Right Hip</td><td>${safe(angleData.hips.right)}</td></tr>

    <tr><td>Neck Angle</td><td>${safe(angleData.neck)}</td></tr>
    <tr><td>Torso Twist</td><td>${safe(angleData.torso)}</td></tr>
  `;
}
import { computeAllAngles, renderAngleTable } from "./js/angle.js";

const angleData = computeAllAngles(cleanKP);
renderAngleTable(angleData, "angleTable");

Best-view confidence selection (when 2 inputs exist)

/js/bestview.js

// --------------------------------------------------------------
// Step 6: Best-View Confidence Selection & Dual-View Fusion
// --------------------------------------------------------------

// Compute total confidence score for an entire keypoint set
export function totalConfidence(kp) {
  return kp.reduce((sum, k) => sum + (k.score || 0), 0);
}

// --------------------------------------------------------------
// Select best view based on total keypoint confidence
// --------------------------------------------------------------
export function pickBestView(kpA, kpB) {
  const scoreA = totalConfidence(kpA);
  const scoreB = totalConfidence(kpB);

  const better = scoreA >= scoreB ? kpA : kpB;

  return {
    mode: "best-view",
    picked: better,
    scoreA,
    scoreB,
  };
}

// --------------------------------------------------------------
// Confidence-weighted fusion (pseudo-3D smoothing)
// 
// For each keypoint:
// fused = (A * wA + B * wB) / (wA + wB)
//
// If one view has too low confidence (< 0.2), return the other.
// --------------------------------------------------------------
export function fuseKeypoints(kpA, kpB, threshold = 0.2) {
  const fused = [];

  for (let i = 0; i < kpA.length; i++) {
    const a = kpA[i];
    const b = kpB[i];

    const wA = a.score;
    const wB = b.score;

    // If both bad → null
    if (wA < threshold && wB < threshold) {
      fused.push(null);
      continue;
    }

    // If A strong, B weak → use A
    if (wA >= threshold && wB < threshold) {
      fused.push({ ...a });
      continue;
    }

    // If B strong, A weak → use B
    if (wB >= threshold && wA < threshold) {
      fused.push({ ...b });
      continue;
    }

    // Both valid → confidence-weighted fusion
    const x = (a.x * wA + b.x * wB) / (wA + wB);
    const y = (a.y * wA + b.y * wB) / (wA + wB);

    fused.push({
      x,
      y,
      score: Math.max(wA, wB), // keep the stronger view's confidence
      name: a.name,
    });
  }

  return {
    mode: "fusion",
    fused,
  };
}

// --------------------------------------------------------------
// Main exported function that decides:
// - If only one view exists → return it
// - If two views exist → best-view OR fused depending on parameter
// --------------------------------------------------------------
export function chooseKeypoints(viewA, viewB, method = "best") {
  if (!viewA && !viewB) return null;
  if (viewA && !viewB) return { mode: "single", picked: viewA };
  if (!viewA && viewB) return { mode: "single", picked: viewB };

  // ---------------- BEST VIEW ----------------
  if (method === "best") {
    return pickBestView(viewA, viewB);
  }

  // ---------------- FUSION ----------------
  if (method === "fusion") {
    return fuseKeypoints(viewA, viewB);
  }

  return null;
}
import { chooseKeypoints } from "./js/bestview.js";

const result = chooseKeypoints(kpA, kpB, "best"); 
// or chooseKeypoints(kpA, kpB, "fusion")
let finalKP = result.picked || result.fused;

Skeleton Drawing Module

/js/draw.js

// --------------------------------------------------------------
// Step 7: Draw Skeleton on Canvas (MoveNet Thunder)
// Supports padded image, scaling, joints & bones
// --------------------------------------------------------------

// MoveNet Thunder 17-keypoint skeleton map
const EDGES = {
  0: [1, 2],     // Nose → Eyes
  1: [3],        // Left eye → Left ear
  2: [4],        // Right eye → Right ear
  5: [6],        // Shoulders
  5: [7],        // Left shoulder → left elbow
  7: [9],        // left elbow → left wrist
  6: [8],        // right shoulder → right elbow
  8: [10],       // right elbow → right wrist
  5: [11],       // left shoulder → left hip
  6: [12],       // right shoulder → right hip
  11: [13],      // left hip → left knee
  13: [15],      // left knee → left ankle
  12: [14],      // right hip → right knee
  14: [16]       // right knee → right ankle
};

// Colors
const JOINT_COLOR = "#43c176";
const BONE_COLOR  = "#2e73ff";

/**
 * Resize canvas to match parent container size
 */
export function fitCanvasToBox(canvas, parentBox) {
  const rect = parentBox.getBoundingClientRect();
  canvas.width  = rect.width;
  canvas.height = rect.height;
}

/**
 * Draw a padded image with skeleton lines & joints
 * @param {HTMLCanvasElement} canvas 
 * @param {HTMLImageElement | HTMLVideoElement} media
 * @param {Array} keypoints - MoveNet kp array
 */
export function drawSkeleton(canvas, media, keypoints) {
  if (!canvas || !media) return;
  const ctx = canvas.getContext("2d");

  // Clear previous frame
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const cw = canvas.width;
  const ch = canvas.height;
  const mw = media.videoWidth || media.naturalWidth;
  const mh = media.videoHeight || media.naturalHeight;

  // Compute padded scale
  const scale = Math.min(cw / mw, ch / mh);
  const w = mw * scale;
  const h = mh * scale;
  const offsetX = (cw - w) / 2;
  const offsetY = (ch - h) / 2;

  // Draw image
  ctx.drawImage(media, offsetX, offsetY, w, h);

  if (!keypoints) return;

  // ------------------ Draw Bones ------------------
  ctx.lineWidth = 3;
  ctx.strokeStyle = BONE_COLOR;
  ctx.globalAlpha = 0.9;

  for (let i = 0; i < keypoints.length; i++) {
    const from = keypoints[i];
    if (!from || from.score < 0.25) continue;

    const edges = EDGES[i];
    if (!edges) continue;

    for (const j of edges) {
      const to = keypoints[j];
      if (!to || to.score < 0.25) continue;

      ctx.beginPath();
      ctx.moveTo(offsetX + from.x * scale, offsetY + from.y * scale);
      ctx.lineTo(offsetX + to.x * scale, offsetY + to.y * scale);
      ctx.stroke();
    }
  }

  // ------------------ Draw Joints ------------------
  ctx.globalAlpha = 1.0;
  ctx.fillStyle = JOINT_COLOR;

  for (const kp of keypoints) {
    if (!kp || kp.score < 0.25) continue;

    const x = offsetX + kp.x * scale;
    const y = offsetY + kp.y * scale;

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

/**
 * Create a canvas inside a result box if missing
 */
export function ensureCanvas(resultBox) {
  let canvas = resultBox.querySelector("canvas");
  if (!canvas) {
    canvas = document.createElement("canvas");
    resultBox.innerHTML = "";
    resultBox.appendChild(canvas);
  }
  return canvas;
}
import { fitCanvasToBox, drawSkeleton, ensureCanvas } from "./js/draw.js";

const box = document.getElementById("normalResultBox");
const canvas = ensureCanvas(box);

fitCanvasToBox(canvas, box);
drawSkeleton(canvas, mediaElement, finalKeypoints);

CSV EXPORT SYSTEM

/js/exportCSV.js

// ------------------------------------------------------------
// STEP 8: CSV EXPORT
// Creates CSV with keypoints, angles, confidence, metadata
// ------------------------------------------------------------

/**
 * Convert array of row objects → CSV string
 */
function convertToCSV(rows) {
  if (!rows || rows.length === 0) return "";

  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];

  for (const row of rows) {
    const line = headers
      .map(h => (row[h] !== undefined ? row[h] : ""))
      .join(",");
    lines.push(line);
  }
  return lines.join("\n");
}

/**
 * Trigger browser download for CSV text
 */
function downloadCSV(filename, csvText) {
  const blob = new Blob([csvText], { type: "text/csv" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();

  URL.revokeObjectURL(url);
}

/**
 * Build rows for CSV from result frames
 * @param {Array} frames   ← Each item = { index, timestamp, keypoints, angles }
 * @param {String} mode    ← "normal" or "advance"
 * @returns CSV rows array
 */
export function buildCSVRows(frames, mode = "normal") {
  const rows = [];

  frames.forEach((frame, idx) => {
    const base = {
      frame_index: frame.index ?? idx,
      timestamp_ms: frame.timestamp ?? 0,
      mode: mode
    };

    // Append keypoints: kp0_x, kp0_y, kp0_score,...
    frame.keypoints.forEach((kp, i) => {
      base[`kp${i}_x`] = Math.round(kp.x);
      base[`kp${i}_y`] = Math.round(kp.y);
      base[`kp${i}_conf`] = kp.score.toFixed(3);
    });

    // Append angles: angle_elbow_left, etc.
    if (frame.angles) {
      for (const key in frame.angles) {
        base[key] = frame.angles[key];
      }
    }

    rows.push(base);
  });

  return rows;
}

/**
 * Export CSV API for app.js
 * @param {Array} frames 
 * @param {String} mode 
 */
export function exportCSV(frames, mode = "normal") {
  if (!frames || frames.length === 0) {
    alert("No results to export yet.");
    return;
  }

  const rows = buildCSVRows(frames, mode);
  const csv = convertToCSV(rows);

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const filename = `ergopose_${mode}_${stamp}.csv`;

  downloadCSV(filename, csv);
}
import { exportCSV } from "./js/exportCSV.js";
let resultFrames = [];  // global
resultFrames.push({
  index: frameIndex,
  timestamp: timestampMS,
  keypoints: finalKeypoints,   // after confidence best-view
  angles: angleResults         // from Step 6
});
document.getElementById("csvBtn").onclick = () => {
  exportCSV(resultFrames, currentMode);
};
<button id="csvBtn" class="btn-primary">Export CSV</button>

GENERATE REPORT (HTML → PDF)

report.js

// ------------------------------------------------------------
// STEP 9: REPORT GENERATOR (HTML → PDF)
// Uses html2canvas + jsPDF
// ------------------------------------------------------------

import { jsPDF } from "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
import html2canvas from "https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.esm.js";

/**
 * Convert an image element or canvas → base64 PNG
 */
async function elementToPNG(element) {
  const canvas = await html2canvas(element, {
    scale: 2,
    backgroundColor: "#ffffff"
  });
  return canvas.toDataURL("image/png");
}

/**
 * Build PDF pages
 * @param {Array} frames - full frame results with skeleton
 * @param {String} mode - "normal" or "advance"
 */
export async function generateReport(frames, mode = "normal") {
  if (!frames || frames.length === 0) {
    alert("Run analysis first.");
    return;
  }

  const pdf = new jsPDF({
    orientation: "portrait",
    unit: "mm",
    format: "a4"
  });

  const pageW = 210;
  const margin = 12;

  // ----------- COVER PAGE -----------
  pdf.setFontSize(21);
  pdf.text("Ergonomic Assessment Report", margin, 30);

  pdf.setFontSize(12);
  pdf.text(`Generated: ${new Date().toLocaleString()}`, margin, 42);
  pdf.text(`Mode: ${mode}`, margin, 50);
  pdf.text(`Frames analyzed: ${frames.length}`, margin, 58);

  pdf.addPage();

  // ----------- ANGLES SUMMARY PAGE -----------
  pdf.setFontSize(16);
  pdf.text("Angle Summary", margin, 20);

  pdf.setFontSize(11);
  let y = 32;

  const sampleAngles = frames[0].angles;
  for (const key in sampleAngles) {
    const avg = (
      frames.reduce((a, f) => a + f.angles[key], 0) / frames.length
    ).toFixed(1);

    pdf.text(`${key}: avg ${avg}°`, margin, y);
    y += 8;
    if (y > 270) {
      pdf.addPage();
      y = 20;
    }
  }

  pdf.addPage();

  // ----------- FRAME IMAGES -----------
  pdf.setFontSize(16);
  pdf.text("Skeleton Frames", margin, 20);

  let posY = 32;

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    const elem = document.querySelector(`#frameCanvas_${frame.index}`);
    if (!elem) continue;

    const png = await elementToPNG(elem);
    const imgW = pageW - margin * 2;
    const imgH = (elem.height / elem.width) * imgW;

    if (posY + imgH > 282) {
      pdf.addPage();
      posY = 20;
    }

    pdf.addImage(png, "PNG", margin, posY, imgW, imgH);
    posY += imgH + 10;
  }

  // ----------- SAVE FILE -----------
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  pdf.save(`ergopose_report_${mode}_${stamp}.pdf`);
}
<button id="reportBtn" class="btn-primary">Generate Report</button>
<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
import { generateReport } from "./js/report.js";
document.getElementById("reportBtn").onclick = () => {
  generateReport(resultFrames, currentMode);
};

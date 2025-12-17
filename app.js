/* ============================================================
   STEP 1 — Load MoveNet Thunder
============================================================ */

let movenetModel = null;

async function loadMoveNetThunder() {
  const status = document.getElementById("modelStatus");
  status.textContent = "Loading model...";

  movenetModel = await tf.loadGraphModel(
    "https://tfhub.dev/google/movenet/singlepose/thunder/4",
    { fromTFHub: true }
  );

  status.textContent = "Model: MoveNet Thunder loaded ✔";
  document.getElementById("processMediaBtn").disabled = false;
}

document.getElementById("loadModelBtn").onclick = loadMoveNetThunder;

/* ============================================================
   STEP 2 — Handle Image / Video Input
============================================================ */

let inputImage = null;
let inputVideo = null;

document.getElementById("imageInput").addEventListener("change", e => {
  inputImage = e.target.files[0] || null;
  inputVideo = null;
});

document.getElementById("videoInput").addEventListener("change", e => {
  inputVideo = e.target.files[0] || null;
  inputImage = null;
});

/* ============================================================
   STEP 3 — Preprocessing (padding, resizing)
============================================================ */

function preprocessImage(img) {
  const tensor = tf.browser.fromPixels(img);
  const [h, w] = tensor.shape.slice(0, 2);
  const size = Math.max(h, w);

  // Make square canvas around the image
  const padTop = (size - h) / 2;
  const padBottom = padTop;
  const padLeft = (size - w) / 2;
  const padRight = padLeft;

  const padded = tf.pad(tensor, [
    [padTop, padBottom],
    [padLeft, padRight],
    [0, 0]
  ]);

  const resized = tf.image.resizeBilinear(padded, [256, 256]);
  const normalized = resized.div(255).expandDims(0);

  return { padded, normalized, padLeft, padTop, size, origW: w, origH: h };
}

/* ============================================================
   STEP 4 — Run inference
============================================================ */

async function runInference(imageTensor) {
  const res = await movenetModel.executeAsync(imageTensor);
  const keypoints = res.arraySync()[0][0];
  tf.dispose(res);
  return keypoints;
}

/* ============================================================
   STEP 5 — Draw skeleton
============================================================ */

const EDGES = {
  0: 1, 1: 3, 3: 5,
  0: 2, 2: 4, 4: 6,
  5: 7, 7: 9,
  6: 8, 8: 10,
  5: 6,
  11: 12,
  5: 11, 6: 12,
  11: 13, 13: 15,
  12: 14, 14: 16
};

function drawSkeleton(ctx, kpts, scaleX, scaleY) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#0f0";
  ctx.fillStyle = "#f00";

  // draw bones
  for (const [a, b] of Object.entries(EDGES)) {
    const A = kpts[a];
    const B = kpts[b];
    if (A[2] > 0.3 && B[2] > 0.3) {
      ctx.beginPath();
      ctx.moveTo(A[0] * scaleX, A[1] * scaleY);
      ctx.lineTo(B[0] * scaleX, B[1] * scaleY);
      ctx.stroke();
    }
  }

  // draw keypoints
  kpts.forEach(k => {
    if (k[2] > 0.3) {
      ctx.beginPath();
      ctx.arc(k[0] * scaleX, k[1] * scaleY, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
}

/* ============================================================
   STEP 6 — Keypoints → Table (with integer output)
============================================================ */

function keypointsToTable(kpts) {
  return kpts.map(k => ({
    x: Math.round(k[0]),
    y: Math.round(k[1]),
    score: k[2].toFixed(3)
  }));
}

/* ============================================================
   STEP 7 — Angle Calculation
============================================================ */

function angle(pA, pB, pC) {
  const BA = { x: pA[0] - pB[0], y: pA[1] - pB[1] };
  const BC = { x: pC[0] - pB[0], y: pC[1] - pB[1] };

  const dot = BA.x * BC.x + BA.y * BC.y;
  const magBA = Math.hypot(BA.x, BA.y);
  const magBC = Math.hypot(BC.x, BC.y);

  if (magBA * magBC === 0) return 0;

  return Math.acos(dot / (magBA * magBC)) * (180 / Math.PI);
}

/* ============================================================
   STEP 8 — Best-view Confidence Selection
============================================================ */

function bestView(kptsFrames) {
  let bestIdx = 0;
  let bestScore = 0;

  kptsFrames.forEach((frame, idx) => {
    const score = frame.reduce((s, k) => s + k[2], 0);
    if (score > bestScore) {
      bestScore = score;
      bestIdx = idx;
    }
  });
  return bestIdx;
}

/* ============================================================
   STEP 9 — CSV Export
============================================================ */

function exportCSV(rows) {
  const headers = Object.keys(rows[0]);
  const csv =
    headers.join(",") +
    "\n" +
    rows.map(r => headers.map(h => r[h]).join(",")).join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "pose_results.csv";
  a.click();
}

/* ============================================================
   MAIN PROCESS HANDLER (Combining Everything)
============================================================ */

let cancelRequested = false;
document.getElementById("cancelBtn").onclick = () => (cancelRequested = true);

document.getElementById("processMediaBtn").onclick = async () => {
  if (!movenetModel) return alert("Load model first.");

  cancelRequested = false;
  document.getElementById("cancelBtn").disabled = false;

  const canvas = document.getElementById("displayCanvas");
  const ctx = canvas.getContext("2d");

  // ---------------- IMAGE MODE ----------------
  if (inputImage) {
    const img = await loadImage(URL.createObjectURL(inputImage));
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const prep = preprocessImage(img);
    const kpts = await runInference(prep.normalized);

    const scaleX = canvas.width / prep.size;
    const scaleY = canvas.height / prep.size;

    drawSkeleton(ctx, kpts, scaleX, scaleY);

    const table = keypointsToTable(kpts);
    lastCSV = table;
    document.getElementById("downloadCSVBtn").disabled = false;
  }

  // ---------------- VIDEO MODE ----------------
  else if (inputVideo) {
    const video = document.createElement("video");
    video.src = URL.createObjectURL(inputVideo);
    await video.play();

    const frameCount = Math.floor(video.duration * 6); // 6FPS sampling
    const kptsFrames = [];

    document.getElementById("progressContainer").style.display = "block";

    for (let i = 0; i < frameCount; i++) {
      if (cancelRequested) break;

      video.currentTime = i / 6;

      await new Promise(res => (video.onseeked = () => res()));

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const prep = preprocessImage(canvas);
      const kpts = await runInference(prep.normalized);
      kptsFrames.push(kpts);

      document.getElementById("progressBar").value =
        ((i + 1) / frameCount) * 100;
      document.getElementById("progressText").textContent = `${i + 1}/${frameCount}`;
    }

    const best = bestView(kptsFrames);

    ctx.drawImage(video, 0, 0);
    const prep = preprocessImage(canvas);
    drawSkeleton(ctx, kptsFrames[best], canvas.width / prep.size, canvas.height / prep.size);

    const table = keypointsToTable(kptsFrames[best]);
    lastCSV = table;
    document.getElementById("downloadCSVBtn").disabled = false;
  }
};

/* ============================================================
   CSV BUTTON
============================================================ */

let lastCSV = null;
document.getElementById("downloadCSVBtn").onclick = () => {
  if (lastCSV) exportCSV(lastCSV);
};

/* ============================================================
   Helper: Load image
============================================================ */

function loadImage(src) {
  return new Promise(res => {
    const img = new Image();
    img.onload = () => res(img);
    img.src = src;
  });
}

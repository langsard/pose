// app.js - MoveNet Thunder integration, padded input, draw skeleton on fixed boxes,
// integer coords table, basic elbow/knee angle calculations.

// KEYPOINT NAMES (MoveNet 17)
const KEYPOINT_NAMES = [
  "Nose","Left Eye","Right Eye","Left Ear","Right Ear",
  "Left Shoulder","Right Shoulder","Left Elbow","Right Elbow",
  "Left Wrist","Right Wrist","Left Hip","Right Hip",
  "Left Knee","Right Knee","Left Ankle","Right Ankle"
];

const SKELETON = [
  [0,1],[0,2],[1,3],[2,4],
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
];

// DOM
const frontInput = document.getElementById('frontInput');
const sideInput = document.getElementById('sideInput');
const frontPreviewImg = document.getElementById('frontPreviewImg');
const sidePreviewImg = document.getElementById('sidePreviewImg');

const runBtn = document.getElementById('runBtn');
const modelStatus = document.getElementById('modelStatus');

const frontResultBox = document.getElementById('frontResultBox');
const sideResultBox = document.getElementById('sideResultBox');
const resultsTableDiv = document.getElementById('resultsTable');
const anglesSummaryDiv = document.getElementById('anglesSummary');

let detector = null;

// Set preview image from file
function setPreviewFromFile(file, imgElement){
  if(!file) return;
  const url = URL.createObjectURL(file);
  imgElement.src = url;
}

// Update run button state
function updateRunButtonState(){
  const hasFile = (frontInput.files && frontInput.files.length) || (sideInput.files && sideInput.files.length);
  const hasPreview = (frontPreviewImg && frontPreviewImg.src) || (sidePreviewImg && sidePreviewImg.src);
  runBtn.disabled = !detector || !(hasFile || hasPreview);
}

// Pad image to square and return canvas + metadata
function padToSquare(imgElement){
  const w = imgElement.naturalWidth || imgElement.width;
  const h = imgElement.naturalHeight || imgElement.height;
  const size = Math.max(w, h);
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = 'white';
  ctx.fillRect(0,0,size,size);

  const offsetX = Math.round((size - w)/2);
  const offsetY = Math.round((size - h)/2);
  ctx.drawImage(imgElement, offsetX, offsetY, w, h);

  return { canvas, offsetX, offsetY, size, originalW: w, originalH: h };
}

// Normalize keypoints (if normalized 0..1 convert to pixels on pad canvas)
function normalizeKeypoints(kps, padSize){
  return kps.map(k => {
    let x = k.x, y = k.y;
    if(x <= 1.01 && y <= 1.01){
      x = x * padSize;
      y = y * padSize;
    }
    return { x, y, score: (k.score ?? 0) };
  });
}

// Render result into fixed-size result box; draw padded image scaled to fit and overlay skeleton
function renderResult(padCanvas, kpArray, resultBox){
  resultBox.innerHTML = '';

  const displayW = resultBox.clientWidth;
  const displayH = resultBox.clientHeight;

  const canvas = document.createElement('canvas');
  canvas.width = displayW;
  canvas.height = displayH;
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  const ctx = canvas.getContext('2d');

  const scale = Math.min(canvas.width / padCanvas.width, canvas.height / padCanvas.height);
  const drawW = padCanvas.width * scale;
  const drawH = padCanvas.height * scale;
  const dx = (canvas.width - drawW)/2;
  const dy = (canvas.height - drawH)/2;

  // draw padded image
  ctx.drawImage(padCanvas, 0,0, padCanvas.width, padCanvas.height, dx, dy, drawW, drawH);

  // transform keypoints to canvas coords
  const scaled = kpArray.map(k => ({ x: Math.round(dx + k.x * scale), y: Math.round(dy + k.y * scale), score: k.score }));

  // draw skeleton
  ctx.lineWidth = Math.max(2, Math.round(2 * scale));
  ctx.strokeStyle = 'lime';
  ctx.fillStyle = 'red';
  SKELETON.forEach(pair=>{
    const a = scaled[pair[0]]; const b = scaled[pair[1]];
    if(!a || !b) return;
    if((a.score||0) < 0.05 || (b.score||0) < 0.05) return;
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  });
  // draw keypoints
  scaled.forEach(p=>{
    if((p.score||0) < 0.05) return;
    ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fill();
  });

  resultBox.appendChild(canvas);
  // return integer coords in displayed canvas space
  return scaled.map(p => ({ x: Math.round(p.x), y: Math.round(p.y), score: p.score }));
}

// compute angle at B from A-B-C in degrees
function computeAngle(A,B,C){
  if(!A||!B||!C) return null;
  const ABx = A.x - B.x, ABy = A.y - B.y;
  const CBx = C.x - B.x, CBy = C.y - B.y;
  const dot = ABx*CBx + ABy*CBy;
  const mag1 = Math.hypot(ABx, ABy);
  const mag2 = Math.hypot(CBx, CBy);
  if(mag1 < 1e-6 || mag2 < 1e-6) return null;
  let cosv = dot / (mag1*mag2);
  cosv = Math.max(-1, Math.min(1, cosv));
  const rad = Math.acos(cosv);
  return (rad * 180 / Math.PI);
}

// build integer coords table
function buildResultsTable(frontCoords, sideCoords){
  let html = '<table><thead><tr><th>Keypoint</th><th>Front co-or</th><th>Side co-or</th></tr></thead><tbody>';
  for(let i=0;i<KEYPOINT_NAMES.length;i++){
    const name = KEYPOINT_NAMES[i];
    const f = frontCoords && frontCoords[i] ? `${frontCoords[i].x}, ${frontCoords[i].y}` : '-';
    const s = sideCoords && sideCoords[i] ? `${sideCoords[i].x}, ${sideCoords[i].y}` : '-';
    html += `<tr><td>${name}</td><td>${f}</td><td>${s}</td></tr>`;
  }
  html += '</tbody></table>';
  return html;
}

// compute elbow/knee angles (rounded integers) from coords array (17 elements)
function computeAnglesFromCoords(coords){
  const idx = {
    leftShoulder:5, rightShoulder:6,
    leftElbow:7, rightElbow:8,
    leftWrist:9, rightWrist:10,
    leftHip:11, rightHip:12,
    leftKnee:13, rightKnee:14,
    leftAnkle:15, rightAnkle:16
  };
  const res = {};
  const safe = i => coords && coords[i] ? coords[i] : null;
  const L_elbow = computeAngle(safe(idx.leftShoulder), safe(idx.leftElbow), safe(idx.leftWrist));
  const R_elbow = computeAngle(safe(idx.rightShoulder), safe(idx.rightElbow), safe(idx.rightWrist));
  const L_knee  = computeAngle(safe(idx.leftHip), safe(idx.leftKnee), safe(idx.leftAnkle));
  const R_knee  = computeAngle(safe(idx.rightHip), safe(idx.rightKnee), safe(idx.rightAnkle));
  if(L_elbow!=null) res.leftElbow = Math.round(L_elbow);
  if(R_elbow!=null) res.rightElbow = Math.round(R_elbow);
  if(L_knee!=null) res.leftKnee = Math.round(L_knee);
  if(R_knee!=null) res.rightKnee = Math.round(R_knee);
  return res;
}

function formatAngles(obj){
  if(!obj || Object.keys(obj).length===0) return '-';
  const parts = [];
  if(obj.leftElbow !== undefined) parts.push(`L elbow ${obj.leftElbow}°`);
  if(obj.rightElbow !== undefined) parts.push(`R elbow ${obj.rightElbow}°`);
  if(obj.leftKnee !== undefined) parts.push(`L knee ${obj.leftKnee}°`);
  if(obj.rightKnee !== undefined) parts.push(`R knee ${obj.rightKnee}°`);
  return parts.join(' · ');
}

// Wait for image decode helper
async function ensureImageDecoded(img){
  if(img.decode) {
    try { await img.decode(); } catch(e) { /* ignore */ }
  }
}

// ---------------- Model loading (MoveNet Thunder) ----------------
async function loadModel(){
  modelStatus.textContent = 'Loading MoveNet Thunder...';
  try {
    // try webgl backend first
    try { await tf.setBackend('webgl'); await tf.ready(); }
    catch(e){ console.warn('webgl backend failed, trying wasm backend', e); try { await tf.setBackend('wasm'); await tf.ready(); } catch(e2){ console.warn('wasm backend failed', e2); } }

    if(typeof poseDetection === 'undefined'){
      throw new Error('poseDetection library (tfjs-models/pose-detection) not found. Check imports.');
    }

    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER }
    );

    modelStatus.textContent = 'Model loaded (MoveNet Thunder)';
    updateRunButtonState();
  } catch (err) {
    console.error('Model load failed:', err);
    modelStatus.textContent = 'Model load failed — check console';
    detector = null;
    updateRunButtonState();
  }
}
loadModel();

// ---------------- Input wiring ----------------
frontInput.addEventListener('change', e => {
  const f = e.target.files[0];
  if(f) setPreviewFromFile(f, frontPreviewImg);
  updateRunButtonState();
});
sideInput.addEventListener('change', e => {
  const f = e.target.files[0];
  if(f) setPreviewFromFile(f, sidePreviewImg);
  updateRunButtonState();
});

frontPreviewImg.addEventListener('load', updateRunButtonState);
sidePreviewImg.addEventListener('load', updateRunButtonState);

// ---------------- Main run handler ----------------
runBtn.addEventListener('click', async () => {
  if(!detector){
    alert('Model not ready. See model status or console.');
    return;
  }
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';
  resultsTableDiv.innerHTML = '';
  anglesSummaryDiv.innerHTML = '';

  try {
    let frontCoords=null, sideCoords=null;

    // FRONT
    if(frontPreviewImg && frontPreviewImg.src){
      await ensureImageDecoded(frontPreviewImg);
      const pad = padToSquare(frontPreviewImg);
      const poses = await detector.estimatePoses(pad.canvas, { maxPoses: 1, flipHorizontal: false });
      if(poses && poses.length){
        const kps = normalizeKeypoints(poses[0].keypoints, pad.size);
        frontCoords = renderResult(pad.canvas, kps, frontResultBox);
      } else {
        frontResultBox.innerHTML = '<div style="color:#c33;padding:8px">No pose detected</div>';
      }
    }

    // SIDE
    if(sidePreviewImg && sidePreviewImg.src){
      await ensureImageDecoded(sidePreviewImg);
      const pad = padToSquare(sidePreviewImg);
      const poses = await detector.estimatePoses(pad.canvas, { maxPoses: 1, flipHorizontal: false });
      if(poses && poses.length){
        const kps = normalizeKeypoints(poses[0].keypoints, pad.size);
        sideCoords = renderResult(pad.canvas, kps, sideResultBox);
      } else {
        sideResultBox.innerHTML = '<div style="color:#c33;padding:8px">No pose detected</div>';
      }
    }

    // table and angles
    resultsTableDiv.innerHTML = buildResultsTable(frontCoords, sideCoords);
    let anglesHtml = '<strong>Angles (degrees):</strong><div style="margin-top:8px">';
    if(frontCoords) anglesHtml += `<div><em>Front:</em> ${formatAngles(computeAnglesFromCoords(mapToModelCoords(frontCoords)))}</div>`;
    if(sideCoords)  anglesHtml += `<div><em>Side:</em>  ${formatAngles(computeAnglesFromCoords(mapToModelCoords(sideCoords)))}</div>`;
    anglesHtml += '</div>';
    anglesSummaryDiv.innerHTML = anglesHtml;

  } catch(err){
    console.error('Detection failed:', err);
    alert('Detection error — see console.');
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Run';
  }
});

// map displayed coords array to model-indexed array for computeAnglesFromCoords
function mapToModelCoords(displayCoords){
  const arr = new Array(KEYPOINT_NAMES.length).fill(null);
  if(!displayCoords) return arr;
  for(let i=0;i<displayCoords.length && i<KEYPOINT_NAMES.length;i++){
    const p = displayCoords[i];
    if(p && typeof p.x !== 'undefined') arr[i] = { x: p.x, y: p.y };
  }
  return arr;
}

// keep run button state in sync
setInterval(()=>updateRunButtonState(), 700);

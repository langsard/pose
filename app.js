// app.js
// Loads MoveNet (TFJS pose-detection), pads images, runs detection,
// draws skeleton on fixed-size result canvas, prints integer coords table,
// computes a few joint angles (elbow, knee) from each available view.

// -------- CONFIG / KEYPOINTS ----------------
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

// -------- DOM elements ----------------
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

// -------- Utilities ----------------
function setPreviewFromFile(file, imgElement){
  if(!file) return;
  const url = URL.createObjectURL(file);
  imgElement.src = url;
}

function updateRunButtonState(){
  const hasPreview = (frontInput.files && frontInput.files.length) || (sideInput.files && sideInput.files.length) ||
    (frontPreviewImg && frontPreviewImg.src) || (sidePreviewImg && sidePreviewImg.src);
  runBtn.disabled = !detector || !hasPreview;
}

// create offscreen canvas padded to square (white background) and return metadata
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

// normalize keypoints: ensure x,y are pixel coordinates for padded canvas
function normalizeKeypoints(kps, padSize){
  // Many detectors return pixels when given canvas input; but if values are in [0,1] convert
  return kps.map(k => {
    let x = k.x, y = k.y;
    if(x <= 1.01 && y <= 1.01){
      x = x * padSize;
      y = y * padSize;
    }
    return { x, y, score: (k.score ?? 0) };
  });
}

// render result: draw padded canvas scaled to result box and overlay skeleton.
// returns array of integer coords in result canvas space or null if no detection.
function renderResult(padCanvas, kpArray, resultBox){
  // remove previous content
  resultBox.innerHTML = '';

  // create canvas the same CSS pixel size as the box
  const displayW = resultBox.clientWidth;
  const displayH = resultBox.clientHeight;

  const canvas = document.createElement('canvas');
  // choose internal resolution to preserve clarity (use CSS size)
  canvas.width = displayW;
  canvas.height = displayH;
  canvas.style.width = '100%';
  canvas.style.height = '100%';

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);

  // scale factor to draw padCanvas into canvas (fit contain)
  const scale = Math.min(canvas.width / padCanvas.width, canvas.height / padCanvas.height);
  const drawW = padCanvas.width * scale;
  const drawH = padCanvas.height * scale;
  const dx = (canvas.width - drawW)/2;
  const dy = (canvas.height - drawH)/2;

  ctx.drawImage(padCanvas, 0, 0, padCanvas.width, padCanvas.height, dx, dy, drawW, drawH);

  // transform keypoints to displayed canvas coordinates
  const scaled = kpArray.map(k => {
    return {
      x: Math.round(dx + k.x * scale),
      y: Math.round(dy + k.y * scale),
      score: k.score
    };
  });

  // draw skeleton lines
  ctx.lineWidth = Math.max(2, Math.round(2 * scale));
  ctx.strokeStyle = 'lime';
  ctx.fillStyle = 'red';

  SKELETON.forEach(pair => {
    const a = scaled[pair[0]];
    const b = scaled[pair[1]];
    if(!a || !b) return;
    // optional: skip low confidence points
    if((a.score||0) < 0.05 || (b.score||0) < 0.05) return;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  // draw keypoint dots
  scaled.forEach(pt=>{
    if((pt.score||0) < 0.05) return;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 4, 0, Math.PI*2);
    ctx.fill();
  });

  resultBox.appendChild(canvas);

  // return integer coords relative to the displayed canvas
  return scaled.map(p => ({ x: Math.round(p.x), y: Math.round(p.y), score: p.score }));
}

// angle at B from A-B-C (degrees)
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

// Build results table with integer coords, columns: Keypoint | Front co-or | Side co-or
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

// compute a small set of angles (we pick common ergonomic joints):
// left elbow (shoulder-elbow-wrist), right elbow, left knee (hip-knee-ankle), right knee
function computeAnglesFromCoords(coords){
  // coords: array of 17 points {x,y}
  const idx = {
    leftShoulder:5, rightShoulder:6,
    leftElbow:7, rightElbow:8,
    leftWrist:9, rightWrist:10,
    leftHip:11, rightHip:12,
    leftKnee:13, rightKnee:14,
    leftAnkle:15, rightAnkle:16
  };
  const results = {};
  function safe(i){ return coords && coords[i] ? coords[i] : null; }
  const L_elbow = computeAngle(safe(idx.leftShoulder), safe(idx.leftElbow), safe(idx.leftWrist));
  const R_elbow = computeAngle(safe(idx.rightShoulder), safe(idx.rightElbow), safe(idx.rightWrist));
  const L_knee  = computeAngle(safe(idx.leftHip), safe(idx.leftKnee), safe(idx.leftAnkle));
  const R_knee  = computeAngle(safe(idx.rightHip), safe(idx.rightKnee), safe(idx.rightAnkle));
  if(L_elbow!=null) results.leftElbow = Math.round(L_elbow);
  if(R_elbow!=null) results.rightElbow = Math.round(R_elbow);
  if(L_knee!=null) results.leftKnee = Math.round(L_knee);
  if(R_knee!=null) results.rightKnee = Math.round(R_knee);
  return results;
}

// -------- Model loading ----------------
async function loadModel(){
  modelStatus.textContent = 'Loading MoveNet (client)...';
  try{
    await tf.setBackend('webgl'); // use webgl if available
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );
    modelStatus.textContent = 'Model loaded';
    updateRunButtonState();
  }catch(err){
    console.error('Model load failed:', err);
    modelStatus.textContent = 'Model load failed (see console)';
  }
}

// -------- Wiring file inputs ------------
frontInput.addEventListener('change', (e)=>{
  const f = e.target.files[0];
  if(f) setPreviewFromFile(f, frontPreviewImg);
  updateRunButtonState();
});
sideInput.addEventListener('change', (e)=>{
  const f = e.target.files[0];
  if(f) setPreviewFromFile(f, sidePreviewImg);
  updateRunButtonState();
});

// re-check run button when previews change
frontPreviewImg.addEventListener('load', updateRunButtonState);
sidePreviewImg.addEventListener('load', updateRunButtonState);

// -------- Main run handler ----------
runBtn.addEventListener('click', async () => {
  if(!detector){
    alert('Model not ready yet.');
    return;
  }
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';
  resultsTableDiv.innerHTML = '';
  anglesSummaryDiv.innerHTML = '';

  try{
    let frontCoords=null, sideCoords=null;

    // FRONT processing - only if there's a preview image that is not empty
    if(frontPreviewImg && frontPreviewImg.src){
      // create pad canvas
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

    // SIDE processing
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

    // Build & show table
    resultsTableDiv.innerHTML = buildResultsTable(frontCoords, sideCoords);

    // compute angles: if both available compute and show both; else show from whichever view exists
    let anglesHtml = '<strong>Angles (degrees):</strong><div style="margin-top:8px">';
    if(frontCoords) {
      const aFront = computeAnglesFromCoords(mapToModelCoords(frontCoords));
      anglesHtml += `<div><em>Front:</em> ${formatAngles(aFront)}</div>`;
    }
    if(sideCoords) {
      const aSide = computeAnglesFromCoords(mapToModelCoords(sideCoords));
      anglesHtml += `<div><em>Side:</em> ${formatAngles(aSide)}</div>`;
    }
    anglesHtml += '</div>';
    anglesSummaryDiv.innerHTML = anglesHtml;

  }catch(err){
    console.error(err);
    alert('Error during detection — see console');
  }finally{
    runBtn.disabled = false;
    runBtn.textContent = 'Run';
  }
});

// helper: ensure image .src is decoded before use
async function ensureImageDecoded(img){
  if(img.decode) {
    try { await img.decode(); } catch(err) { /* ignore */ }
  }
}

// map displayed coords back to a reasonable model-space array (17 elements), filling missing with nulls
function mapToModelCoords(displayCoords){
  // displayCoords is an array of 17 objects {x,y,score} in canvas space; we can use them directly for angle math.
  // Build an array with indexes matching KEYPOINT_NAMES keeping x,y or null.
  const arr = new Array(KEYPOINT_NAMES.length).fill(null);
  if(!displayCoords) return arr;
  for(let i=0;i<displayCoords.length && i<KEYPOINT_NAMES.length;i++){
    const p = displayCoords[i];
    if(p && typeof p.x !== 'undefined') arr[i] = { x: p.x, y: p.y };
  }
  return arr;
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

// initial model load
loadModel();

// make sure run button reacts to model load / preview changes
const checkInterval = setInterval(()=>{ updateRunButtonState(); }, 700);
window.addEventListener('beforeunload', ()=>clearInterval(checkInterval));

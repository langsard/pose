// app.js (Full finished file)
// Ergonomic Assessment (MoveNet Thunder)
// Features:
//  - Normal (single) or Advance (front+side) mode
//  - Image / Video input (video sampling via prompt; default 60s)
//  - Pad-to-square without distortion; draw skeleton on padded output(s)
//  - Integer coords table (Keypoint / Front co-or / Side co-or)
//  - Angle calculations (elbows, knees), Best-view selection
//  - Pseudo-3D fusion (simple depth from side.x scaled to front width)
//  - Progress + Cancel + CSV export

/* ========== CONFIG ========== */
const CONF_THRESHOLD = 0.30; // minimum keypoint score to be considered
const DEFAULT_SAMPLING_SEC = 60; // seconds per sample for video if user doesn't change
/* ============================ */

/* ========== DOM ========== */
const normalModeBtn = document.getElementById('normalModeBtn');
const advanceModeBtn = document.getElementById('advanceModeBtn');

const normalUpload = document.getElementById('normalUpload');
const advanceUpload = document.getElementById('advanceUpload');

const normalInput = document.getElementById('normalInput');
const frontInput = document.getElementById('frontInput');
const sideInput = document.getElementById('sideInput');

const normalResultBox = document.getElementById('normalResultBox');
const frontResultBox = document.getElementById('frontResultBox');
const sideResultBox = document.getElementById('sideResultBox');

const runBtn = document.getElementById('runBtn');
const modelStatus = document.getElementById('modelStatus');
const resultsTableDiv = document.getElementById('resultsTable');
const anglesSummaryDiv = document.getElementById('anglesSummary');

/* ========== INTERNAL STATE ========== */
let currentMode = 'normal'; // 'normal' or 'advance'
let detector = null;
let modelReady = false;
let cancelRequested = false;

/* Hidden video elements (for frame capture) created lazily */
let hiddenVideoA = null;
let hiddenVideoB = null;

/* KEYPOINT NAMES & SKELETON (MoveNet 17) */
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

/* ========== UTILS ========== */
function logWarn(...args){ console.warn('[app]',...args); }
function logInfo(...args){ console.log('[app]',...args); }

/* create an element with attrs */
function el(tag, attrs={}) {
  const e = document.createElement(tag);
  for (const k in attrs) {
    if (k === 'text') e.textContent = attrs[k];
    else e.setAttribute(k, attrs[k]);
  }
  return e;
}

/* sleep */
const sleep = (ms) => new Promise(r=>setTimeout(r, ms));

/* prompt sampling seconds for videos (fallback to default) */
async function askSamplingSeconds(defaultSec=DEFAULT_SAMPLING_SEC){
  try {
    const raw = prompt(`Enter sampling interval in seconds for video frames (integer). Default ${defaultSec}s:`, `${defaultSec}`);
    if (!raw) return defaultSec;
    const n = parseInt(raw,10);
    if (isNaN(n) || n <= 0) return defaultSec;
    return n;
  } catch(e){ return defaultSec; }
}

/* convert image/video element to a canvas capturing current frame / full image */
function createCanvasFromImageEl(imgEl){
  const c = document.createElement('canvas');
  c.width = imgEl.naturalWidth || imgEl.width || 1;
  c.height = imgEl.naturalHeight || imgEl.height || 1;
  const ctx = c.getContext('2d');
  ctx.drawImage(imgEl,0,0,c.width,c.height);
  return c;
}
function createCanvasFromVideoEl(videoEl){
  const c = document.createElement('canvas');
  c.width = videoEl.videoWidth || 1;
  c.height = videoEl.videoHeight || 1;
  const ctx = c.getContext('2d');
  ctx.drawImage(videoEl,0,0,c.width,c.height);
  return c;
}

/* Seek video to time (seconds), resolves when seeked or rejects */
function seekTo(videoEl, timeSec, timeoutMs=7000){
  return new Promise((resolve, reject) => {
    function cleanup(){ videoEl.removeEventListener('seeked', onseek); videoEl.removeEventListener('error', onerr); clearTimeout(timer); }
    function onseek(){ cleanup(); resolve(); }
    function onerr(e){ cleanup(); reject(e || new Error('video error')); }
    const timer = setTimeout(()=>{ cleanup(); reject(new Error('seek timeout')); }, timeoutMs);
    videoEl.addEventListener('seeked', onseek);
    videoEl.addEventListener('error', onerr);
    // clamp time
    try { videoEl.currentTime = Math.min(Math.max(0, timeSec), Math.max(0, (videoEl.duration||0) - 0.001)); }
    catch(e){ cleanup(); reject(e); }
  });
}

/* Pad a canvas to square: returns {canvas, offsetX, offsetY, size, originalW, originalH} */
function padCanvasToSquare(sourceCanvas){
  const w = sourceCanvas.width;
  const h = sourceCanvas.height;
  const size = Math.max(w, h);
  const out = document.createElement('canvas');
  out.width = size; out.height = size;
  const ctx = out.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0,0,size,size);
  const offsetX = Math.round((size - w)/2);
  const offsetY = Math.round((size - h)/2);
  ctx.drawImage(sourceCanvas, offsetX, offsetY, w, h);
  return { canvas: out, offsetX, offsetY, size, originalW: w, originalH: h };
}

/* Draw skeleton and keypoints on a canvas (keypoints in pixel coords relative to canvas size) */
function drawSkeleton(canvas, keypoints, opts={}){
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.lineWidth = opts.lineWidth || Math.max(2, Math.round(Math.min(w,h)/200));
  ctx.strokeStyle = opts.lineColor || 'lime';
  ctx.fillStyle = opts.pointColor || 'red';
  // bones
  SKELETON.forEach(pair=>{
    const a = keypoints[pair[0]];
    const b = keypoints[pair[1]];
    if(!a || !b) return;
    if((a.score||0) < CONF_THRESHOLD || (b.score||0) < CONF_THRESHOLD) return;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  });
  // points
  keypoints.forEach(k=>{
    if(!k) return;
    if((k.score||0) < CONF_THRESHOLD) return;
    ctx.beginPath();
    const r = Math.max(3, Math.round(Math.min(w,h)/150));
    ctx.arc(k.x, k.y, r, 0, Math.PI*2);
    ctx.fill();
  });
}

/* Scale padded canvas into a result-box sized canvas and draw skeleton there (returns scaled coords) */
function renderPaddedToBox(padCanvas, keypoints, resultBox){
  // resultBox is the element with fixed CSS size (.image-box). We'll create a canvas sized to the displayed pixel size.
  resultBox.innerHTML = ''; // clear
  const displayW = resultBox.clientWidth || padCanvas.width;
  const displayH = resultBox.clientHeight || padCanvas.height;

  const out = document.createElement('canvas');
  out.width = displayW;
  out.height = displayH;
  out.style.width = '100%';
  out.style.height = '100%';
  const ctx = out.getContext('2d');

  // scale preserve aspect (padCanvas is square). draw centered
  const scale = Math.min(out.width / padCanvas.width, out.height / padCanvas.height);
  const drawW = padCanvas.width * scale;
  const drawH = padCanvas.height * scale;
  const dx = Math.round((out.width - drawW) / 2);
  const dy = Math.round((out.height - drawH) / 2);
  ctx.drawImage(padCanvas, 0,0, padCanvas.width, padCanvas.height, dx, dy, drawW, drawH);

  // transform keypoints from padCanvas pixel coords to out canvas coords
  const scaled = keypoints.map(k => {
    if(!k) return null;
    return {
      x: Math.round(dx + k.x * scale),
      y: Math.round(dy + k.y * scale),
      score: k.score
    };
  });

  // draw skeleton on out canvas using scaled coords
  // But preserve drawing style consistent
  ctx.lineWidth = Math.max(2, Math.round(2 * scale));
  ctx.strokeStyle = 'lime';
  ctx.fillStyle = 'red';
  SKELETON.forEach(pair=>{
    const a = scaled[pair[0]]; const b = scaled[pair[1]];
    if(!a || !b) return;
    if((a.score||0) < CONF_THRESHOLD || (b.score||0) < CONF_THRESHOLD) return;
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  });
  scaled.forEach(p=>{
    if(!p) return;
    if((p.score||0) < CONF_THRESHOLD) return;
    ctx.beginPath(); ctx.arc(p.x,p.y, Math.max(3, Math.round(2*scale)), 0, Math.PI*2); ctx.fill();
  });

  resultBox.appendChild(out);

  // Return integer coords in displayed canvas coordinate system (these are what user sees)
  return scaled.map(p => p ? { x: Math.round(p.x), y: Math.round(p.y), score: p.score } : null);
}

/* compute angle at B formed by A-B-C (2D or 3D coords); returns degrees or null */
function computeAngle(A,B,C){
  if(!A||!B||!C) return null;
  const ABx = A.x - B.x, ABy = A.y - B.y, ABz = ('z' in A || 'z' in B) ? ((A.z||0) - (B.z||0)) : 0;
  const CBx = C.x - B.x, CBy = C.y - B.y, CBz = ('z' in C || 'z' in B) ? ((C.z||0) - (B.z||0)) : 0;
  const dot = ABx*CBx + ABy*CBy + ABz*CBz;
  const mag1 = Math.hypot(ABx, ABy, ABz);
  const mag2 = Math.hypot(CBx, CBy, CBz);
  if(mag1 < 1e-6 || mag2 < 1e-6) return null;
  let cosv = dot / (mag1 * mag2);
  cosv = Math.max(-1, Math.min(1, cosv));
  const rad = Math.acos(cosv);
  return (rad * 180 / Math.PI);
}

/* Simple fused 3D approximation from frontKP (x,y) and sideKP (x,y): returns fused array of { name, x,y,z, scoreFront, scoreSide } */
function fuseKeypoints(frontKP, sideKP){
  if(!frontKP && !sideKP) return null;
  // frontKP and sideKP are arrays of keypoint objects with x,y,score in pixel coords relative to padded canvas
  const frontW = frontKP ? Math.max(...frontKP.map(k=>k.x)) - Math.min(...frontKP.map(k=>k.x)) : 1;
  const sideW  = sideKP  ? Math.max(...sideKP.map(k=>k.x))  - Math.min(...sideKP.map(k=>k.x))  : 1;

  const fused = [];
  for(let i=0;i<KEYPOINT_NAMES.length;i++){
    const f = frontKP && frontKP[i] ? frontKP[i] : null;
    const s = sideKP  && sideKP[i]  ? sideKP[i]  : null;
    const yVals = [];
    if(f && (f.score||0) >= CONF_THRESHOLD) yVals.push(f.y);
    if(s && (s.score||0) >= CONF_THRESHOLD) yVals.push(s.y);
    const y = yVals.length ? (yVals.reduce((a,b)=>a+b,0)/yVals.length) : null;

    const x = f && (f.score||0) >= CONF_THRESHOLD ? f.x : (s && (s.score||0) >= CONF_THRESHOLD ? s.x : null);

    let z = null;
    if(s && (s.score||0) >= CONF_THRESHOLD && sideW > 0){
      // center-normalize side.x and scale to front width to produce relative depth
      const minSide = Math.min(...sideKP.map(k=>k.x));
      const maxSide = Math.max(...sideKP.map(k=>k.x));
      const centerSide = (minSide + maxSide)/2;
      z = (s.x - centerSide) * (frontW / Math.max(sideW,1));
    }

    fused.push({
      name: KEYPOINT_NAMES[i],
      x: x !== null ? Math.round(x) : null,
      y: y !== null ? Math.round(y) : null,
      z: z !== null ? Math.round(z) : null,
      scoreFront: f ? (f.score||0) : 0,
      scoreSide:  s ? (s.score||0) : 0
    });
  }
  return fused;
}

/* Build HTML results table (integer coords) with columns: Keypoint | Front co-or | Side co-or */
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

/* Compute elbow/knee angles (works with 2D fused or single-view coords) - return object */
function computeAnglesFromCoords(coordsArray){
  // coordsArray: array indexed by keypoint index, each {x,y,z?,score?} or null
  const idx = {
    leftShoulder:5, rightShoulder:6,
    leftElbow:7, rightElbow:8,
    leftWrist:9, rightWrist:10,
    leftHip:11, rightHip:12,
    leftKnee:13, rightKnee:14,
    leftAnkle:15, rightAnkle:16
  };
  const get = i => coordsArray && coordsArray[i] ? coordsArray[i] : null;
  const res = {};
  const L_el = computeAngle(get(idx.leftShoulder), get(idx.leftElbow), get(idx.leftWrist));
  const R_el = computeAngle(get(idx.rightShoulder), get(idx.rightElbow), get(idx.rightWrist));
  const L_kn = computeAngle(get(idx.leftHip), get(idx.leftKnee), get(idx.leftAnkle));
  const R_kn = computeAngle(get(idx.rightHip), get(idx.rightKnee), get(idx.rightAnkle));
  if(L_el!=null) res.leftElbow = Math.round(L_el);
  if(R_el!=null) res.rightElbow = Math.round(R_el);
  if(L_kn!=null) res.leftKnee = Math.round(L_kn);
  if(R_kn!=null) res.rightKnee = Math.round(R_kn);
  return res;
}

/* Format angles object to a short text */
function formatAngles(obj){
  if(!obj || Object.keys(obj).length===0) return '-';
  const parts=[];
  if(obj.leftElbow!==undefined) parts.push(`L elbow ${obj.leftElbow}°`);
  if(obj.rightElbow!==undefined) parts.push(`R elbow ${obj.rightElbow}°`);
  if(obj.leftKnee!==undefined) parts.push(`L knee ${obj.leftKnee}°`);
  if(obj.rightKnee!==undefined) parts.push(`R knee ${obj.rightKnee}°`);
  return parts.join(' · ');
}

/* Create downloadable CSV from results array (samples) */
function createCSVDownload(results){
  // results: array of { t, fused (array), angles3D (object) }
  const rows = [];
  // header
  const header = ['sample_index','time_s','keypoint','x','y','z','scoreFront','scoreSide','angles'];
  rows.push(header.join(','));
  results.forEach((r, idx)=>{
    const angparts = [];
    if(r.angles3D.leftElbow!==undefined) angparts.push(`Lel:${r.angles3D.leftElbow}`);
    if(r.angles3D.rightElbow!==undefined) angparts.push(`Rel:${r.angles3D.rightElbow}`);
    if(r.angles3D.leftKnee!==undefined) angparts.push(`Lkn:${r.angles3D.leftKnee}`);
    if(r.angles3D.rightKnee!==undefined) angparts.push(`Rkn:${r.angles3D.rightKnee}`);
    const angStr = angparts.join(';') || '';
    if(r.fused && r.fused.length){
      r.fused.forEach(k=>{
        rows.push(`${idx},${Math.round(r.t)},${k.name},${k.x===null?'':k.x},${k.y===null?'':k.y},${k.z===null?'':k.z},${k.scoreFront||0},${k.scoreSide||0},"${angStr}"`);
      });
    } else {
      rows.push(`${idx},${Math.round(r.t)},-,,-,-,0,0,"${angStr}"`);
    }
  });
  const csv = rows.join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ergonomics_results_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ========== MODEL LOADING ========== */
async function loadModel(){
  modelStatus.textContent = 'Loading MoveNet Thunder...';
  try {
    // try preferred tf backend (webgl) -- tf.js may choose
    if (typeof tf !== 'undefined'){
      try { await tf.setBackend('webgl'); await tf.ready(); } catch(e){ try{ await tf.setBackend('wasm'); await tf.ready(); } catch(e2){} }
    }
    if (typeof poseDetection === 'undefined'){
      throw new Error('poseDetection library not found. Check HTML script imports.');
    }
    // create detector
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER, runtime:'tfjs' }
    );
    modelReady = true;
    modelStatus.textContent = 'Model loaded (MoveNet Thunder)';
    runBtn.disabled = false; // allow run if inputs present
  } catch(err){
    console.error('Model load failed', err);
    modelStatus.textContent = 'Model load failed — check console';
    modelReady = false;
    runBtn.disabled = true;
  }
}
loadModel();

/* ========== UI: Mode toggle & preview wiring ========== */
function activateNormalMode(){
  currentMode = 'normal';
  normalUpload.style.display = 'flex';
  advanceUpload.style.display = 'none';
  document.body.classList.add('normal-mode'); document.body.classList.remove('advance-mode');
  normalModeBtn.style.background = 'var(--accent)'; normalModeBtn.style.color = '#fff';
  advanceModeBtn.style.background = '#fff'; advanceModeBtn.style.color = '#000';
  // disable run until file chosen (HTML already has run button logic; we'll re-run check)
  checkRunEnable();
}
function activateAdvanceMode(){
  currentMode = 'advance';
  normalUpload.style.display = 'none';
  advanceUpload.style.display = 'flex';
  document.body.classList.add('advance-mode'); document.body.classList.remove('normal-mode');
  advanceModeBtn.style.background = 'var(--accent)'; advanceModeBtn.style.color = '#fff';
  normalModeBtn.style.background = '#fff'; normalModeBtn.style.color = '#000';
  checkRunEnable();
}

normalModeBtn.addEventListener('click', activateNormalMode);
advanceModeBtn.addEventListener('click', activateAdvanceMode);

/* preview update helpers */
function setPreviewFromFileInput(fileInput, previewImgEl){
  if(!fileInput || !fileInput.files || fileInput.files.length === 0) return;
  const f = fileInput.files[0];
  if(!f) return;
  const url = URL.createObjectURL(f);
  previewImgEl.src = url;
}

/* Check whether run button should be enabled (simple: require uploaded files) */
function checkRunEnable(){
  if(!modelReady){ runBtn.disabled = true; return; }
  if(currentMode === 'normal'){
    runBtn.disabled = !(normalInput.files && normalInput.files.length > 0);
  } else {
    runBtn.disabled = !(
      frontInput.files && frontInput.files.length > 0 &&
      sideInput.files && sideInput.files.length > 0
    );
  }
}
normalInput.addEventListener('change', ()=>{ setPreviewFromFileInput(normalInput, document.getElementById('normalPreviewImg')); checkRunEnable(); });
frontInput.addEventListener('change', ()=>{ setPreviewFromFileInput(frontInput, document.getElementById('frontPreviewImg')); checkRunEnable(); });
sideInput.addEventListener('change', ()=>{ setPreviewFromFileInput(sideInput, document.getElementById('sidePreviewImg')); checkRunEnable(); });

activateNormalMode(); // initial

/* ========== Processing / Run handler ========== */
runBtn.addEventListener('click', async () => {
  if(!modelReady){ alert('Model not ready. Check model status.'); return; }

  // create Cancel + CSV buttons in UI area
  cancelRequested = false;
  runBtn.disabled = true;
  const controlsRow = runBtn.parentElement;
  // create cancel if not exists
  let cancelBtn = document.getElementById('cancelProcessBtn');
  if(!cancelBtn){
    cancelBtn = el('button', { id:'cancelProcessBtn' });
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.marginLeft = '8px';
    cancelBtn.className = 'btn-primary';
    cancelBtn.style.background = '#e23';
    cancelBtn.onclick = ()=>{ cancelRequested = true; cancelBtn.disabled = true; cancelBtn.textContent = 'Cancelling...'; };
    controlsRow.appendChild(cancelBtn);
  } else {
    cancelBtn.disabled = false; cancelBtn.textContent = 'Cancel';
  }

  // CSV export button
  let csvBtn = document.getElementById('exportCsvBtn');
  if(!csvBtn){
    csvBtn = el('button', { id:'exportCsvBtn' });
    csvBtn.textContent = 'Export CSV';
    csvBtn.style.marginLeft = '8px';
    csvBtn.className = 'btn-primary';
    csvBtn.style.background = '#333';
    csvBtn.onclick = ()=>{ if(latestResults && latestResults.length) createCSVDownload(latestResults); else alert('No results to export'); };
    controlsRow.appendChild(csvBtn);
  }

  // UI feedback
  modelStatus.textContent = 'Running detection...';
  resultsTableDiv.innerHTML = ''; anglesSummaryDiv.innerHTML = '';
  normalResultBox.innerHTML = ''; frontResultBox.innerHTML = ''; sideResultBox.innerHTML = '';

  // Determine inputs based on mode
  const adv = (currentMode === 'advance');

  // Prepare input descriptors: { kind:'image'|'video', file, el: preview image/video element }
  async function prepareInputFromFileInput(fileInput, previewImgId){
    if(!fileInput || !fileInput.files || fileInput.files.length === 0) return null;
    const file = fileInput.files[0];
    const url = URL.createObjectURL(file);
    if(file.type.startsWith('image/')){
      const img = new Image();
      img.src = url;
      await new Promise((res, rej)=>{ img.onload = res; img.onerror = rej; });
      return { kind:'image', file, el: img, url };
    } else if(file.type.startsWith('video/')){
      // use or create hidden video element to avoid UI playback
      const v = document.createElement('video');
      v.src = url;
      v.muted = true; v.playsInline = true; v.preload = 'metadata';
      // wait for metadata
      try {
        await new Promise((res, rej)=>{
          const onmd = ()=>{ v.removeEventListener('loadedmetadata', onmd); res(); };
          v.addEventListener('loadedmetadata', onmd);
          setTimeout(()=>rej(new Error('video metadata timeout')), 7000);
        });
      } catch(e){
        // continue even if metadata not loaded; user has to ensure file is ok
        logWarn('video metadata error', e);
      }
      return { kind:'video', file, el: v, url };
    } else {
      return null;
    }
  }

  // gather inputs
  let inputA = null, inputB = null;
  try {
    if(!adv){
      inputA = await prepareInputFromFileInput(normalInput, 'normalPreviewImg');
      if(!inputA){ alert('Please choose a valid input file'); return; }
    } else {
      inputA = await prepareInputFromFileInput(frontInput, 'frontPreviewImg');
      inputB = await prepareInputFromFileInput(sideInput, 'sidePreviewImg');
      if(!inputA || !inputB){ alert('Both front and side inputs required in Advance mode'); return; }
    }
  } catch(e){
    console.error('prepareInput error', e);
    alert('Failed to prepare inputs. See console.');
    runBtn.disabled = false;
    cancelBtn.remove();
    return;
  }

  // determine timestamps to sample
  let samplingSec = DEFAULT_SAMPLING_SEC;
  // If any involved input is video, ask user for sampling seconds
  const anyVideo = (inputA && inputA.kind === 'video') || (inputB && inputB.kind === 'video');
  if(anyVideo){
    samplingSec = await askSamplingSeconds(DEFAULT_SAMPLING_SEC);
    if(!samplingSec || typeof samplingSec !== 'number' || samplingSec <= 0) samplingSec = DEFAULT_SAMPLING_SEC;
  }

  // compute timestamps
  let timestamps = [];
  if(!adv){
    if(inputA.kind === 'video'){
      const dur = inputA.el.duration || 0;
      if(dur <= 0){ timestamps = [0]; }
      else {
        for(let t=0; t < dur; t += samplingSec) timestamps.push(Math.min(t, dur-0.001));
        if(timestamps.length === 0) timestamps.push(0);
      }
    } else {
      timestamps = [0];
    }
  } else {
    // advance: if both videos, use min duration; if one video, use that duration
    let duration = 0;
    if(inputA.kind === 'video' && inputB.kind === 'video'){
      duration = Math.min(inputA.el.duration || 0, inputB.el.duration || 0);
    } else if(inputA.kind === 'video') duration = inputA.el.duration || 0;
    else if(inputB.kind === 'video') duration = inputB.el.duration || 0;
    if(duration > 0){
      for(let t=0; t < duration; t += samplingSec) timestamps.push(Math.min(t, duration-0.001));
      if(timestamps.length === 0) timestamps.push(0);
    } else timestamps = [0];
  }

  // results array for CSV/export and rendering
  const results = [];
  latestResults = results; // allow CSV button access

  // sequential sample loop
  for(let si=0; si<timestamps.length; si++){
    if(cancelRequested) break;
    const t = timestamps[si];
    modelStatus.textContent = `Processing sample ${si+1}/${timestamps.length} (t=${Math.round(t)}s)...`;

    // prepare front and side canvas (padded)
    let frontCanvasPad = null, sideCanvasPad = null;
    // front = inputA (normal or front)
    try {
      let frontCanvasRaw = null;
      if(inputA.kind === 'image'){
        frontCanvasRaw = createCanvasFromImageEl(inputA.el);
      } else { // video
        try { await seekTo(inputA.el, t); } catch(e){ logWarn('seek front failed', e); }
        // small delay to let frame render
        await sleep(50);
        frontCanvasRaw = createCanvasFromVideoEl(inputA.el);
      }
      frontCanvasPad = padCanvasToSquare(frontCanvasRaw).canvas;
    } catch(e){
      console.error('front capture error', e);
    }

    if(adv && inputB){
      try {
        let sideCanvasRaw = null;
        if(inputB.kind === 'image'){
          sideCanvasRaw = createCanvasFromImageEl(inputB.el);
        } else {
          try { await seekTo(inputB.el, t); } catch(e){ logWarn('seek side failed', e); }
          await sleep(50);
          sideCanvasRaw = createCanvasFromVideoEl(inputB.el);
        }
        sideCanvasPad = padCanvasToSquare(sideCanvasRaw).canvas;
      } catch(e){
        console.error('side capture error', e);
      }
    }

    // Run MoveNet detection
    let poseFront = null, poseSide = null;
    try {
      if(frontCanvasPad){
        const pf = await detector.estimatePoses(frontCanvasPad, { maxPoses:1, flipHorizontal: false });
        poseFront = (pf && pf.length) ? pf[0] : null;
      }
    } catch(e){ console.error('detect front error', e); }

    if(sideCanvasPad){
      try {
        const ps = await detector.estimatePoses(sideCanvasPad, { maxPoses:1, flipHorizontal: false });
        poseSide = (ps && ps.length) ? ps[0] : null;
      } catch(e){ console.error('detect side error', e); }
    }

    // Draw skeletons onto copy canvases at original padded resolution
    let frontOut = null, sideOut = null;
    if(frontCanvasPad){
      frontOut = document.createElement('canvas');
      frontOut.width = frontCanvasPad.width;
      frontOut.height = frontCanvasPad.height;
      const fctx = frontOut.getContext('2d'); fctx.drawImage(frontCanvasPad, 0,0);
      if(poseFront && poseFront.keypoints){
        // MoveNet returns keypoints with x,y in pixels relative to input (canvas), so safe to draw directly
        drawSkeleton(frontOut, poseFront.keypoints, {});
      }
    }

    if(sideCanvasPad){
      sideOut = document.createElement('canvas');
      sideOut.width = sideCanvasPad.width;
      sideOut.height = sideCanvasPad.height;
      const sctx = sideOut.getContext('2d'); sctx.drawImage(sideCanvasPad, 0,0);
      if(poseSide && poseSide.keypoints){
        drawSkeleton(sideOut, poseSide.keypoints, {});
      }
    }

    // render small scaled view(s) into result box(es) for the user
    // If normal mode and multiple timestamps, append canvases sequentially inside normalResultBox
    let renderedFrontDisplayCoords = null;
    let renderedSideDisplayCoords = null;

    if(!adv){
      // normal mode: show each sample canvas appended inside normalResultBox
      const wrapper = document.createElement('div');
      wrapper.style.display = 'inline-block';
      wrapper.style.marginRight = '10px';
      wrapper.style.width = `${normalResultBox.clientWidth}px`;
      wrapper.style.height = `${normalResultBox.clientHeight}px`;
      // We'll render padded frontOut (if available) scaled into a canvas sized to the box
      if(frontOut){
        // temporarily draw skeleton into frontOut if not already done
        renderedFrontDisplayCoords = renderPaddedToBox(frontOut, poseFront && poseFront.keypoints ? poseFront.keypoints : new Array(KEYPOINT_NAMES.length).fill(null), wrapper);
      } else {
        const div = el('div'); div.textContent = 'No frame';
        wrapper.appendChild(div);
      }
      normalResultBox.appendChild(wrapper);
    } else {
      // advance mode: frontResultBox and sideResultBox each should show ONE appended canvas per sample (we append)
      const frontWrap = document.createElement('div'); frontWrap.style.display='inline-block'; frontWrap.style.marginRight='10px';
      const sideWrap  = document.createElement('div'); sideWrap.style.display='inline-block'; sideWrap.style.marginRight='10px';
      if(frontOut){
        renderedFrontDisplayCoords = renderPaddedToBox(frontOut, poseFront && poseFront.keypoints ? poseFront.keypoints : new Array(KEYPOINT_NAMES.length).fill(null), frontWrap);
      } else { frontWrap.appendChild(el('div',{ text: 'No frame' })); }
      if(sideOut){
        renderedSideDisplayCoords = renderPaddedToBox(sideOut, poseSide && poseSide.keypoints ? poseSide.keypoints : new Array(KEYPOINT_NAMES.length).fill(null), sideWrap);
      } else { sideWrap.appendChild(el('div',{ text: 'No frame' })); }
      frontResultBox.appendChild(frontWrap);
      sideResultBox.appendChild(sideWrap);
    }

    // Build coords arrays in model-indexed order from returned keypoints
    // PoseDetection keypoint structure: {name, x, y, score}
    function buildCoordsFromPose(pose){
      if(!pose || !pose.keypoints) return null;
      const arr = new Array(KEYPOINT_NAMES.length).fill(null);
      for(let i=0;i<pose.keypoints.length && i<KEYPOINT_NAMES.length;i++){
        const kp = pose.keypoints[i];
        // Note: x,y here are pixel coords relative to the padded canvas
        arr[i] = { x: Math.round(kp.x), y: Math.round(kp.y), score: kp.score || 0 };
      }
      return arr;
    }
    const coordsFront = poseFront ? buildCoordsFromPose(poseFront) : null;
    const coordsSide  = poseSide ? buildCoordsFromPose(poseSide) : null;

    // For display table we want integer display coordinates relative to displayed canvas.
    // The earlier 'renderedFrontDisplayCoords' / 'renderedSideDisplayCoords' are in displayed space.
    // We'll use those for table display.
    let displayFront = renderedFrontDisplayCoords;
    let displaySide = renderedSideDisplayCoords;
    // If null, fill with nulls of length 17
    if(!displayFront && coordsFront){
      // If we didn't render display (rare), map padded coords directly to integers (rounded)
      displayFront = coordsFront.map(k => k ? { x: k.x, y: k.y, score: k.score } : null);
    }
    if(!displaySide && coordsSide){
      displaySide = coordsSide.map(k => k ? { x: k.x, y: k.y, score: k.score } : null);
    }

    // fused pseudo-3D
    const fused = fuseKeypoints(coordsFront, coordsSide);

    // compute 3D angles from fused
    const angles3D = computeAnglesFromCoords(fused ? fused.map(k=>k ? { x:k.x, y:k.y, z:k.z } : null) : null);

    // compute 2D angles per view as well for extra info
    const anglesFront2D = computeAnglesFromCoords(coordsFront);
    const anglesSide2D  = computeAnglesFromCoords(coordsSide);

    // push sample result
    results.push({
      t,
      coordsFront: coordsFront ? coordsFront : null,
      coordsSide: coordsSide ? coordsSide : null,
      displayFront: displayFront ? displayFront : null,
      displaySide: displaySide ? displaySide : null,
      fused,
      angles3D,
      anglesFront2D,
      anglesSide2D
    });

    // quick pause to keep UI responsive
    await sleep(40);
  } // end for timestamps

  // finished or cancelled
  if(cancelRequested) modelStatus.textContent = 'Cancelled';
  else modelStatus.textContent = `Done: ${results.length} sample(s) processed.`;

  // render final table and angles summary for first sample (or best sample)
  if(results.length > 0){
    // Choose sample to summarise: pick highest-avg-confidence sample across front+side
    let bestIdx = 0;
    let bestScore = -1;
    results.forEach((r, idx)=>{
      let s = 0, cnt = 0;
      if(r.coordsFront) { r.coordsFront.forEach(k=>{ if(k) { s += (k.score||0); cnt++; } }); }
      if(r.coordsSide)  { r.coordsSide.forEach(k=>{ if(k) { s += (k.score||0); cnt++; } }); }
      const avg = cnt>0 ? s/cnt : 0;
      if(avg > bestScore){ bestScore = avg; bestIdx = idx; }
    });

    const chosen = results[bestIdx];

    // Build results table using displayed coords (displayFront/displaySide) if present
    resultsTableDiv.innerHTML = buildResultsTable(chosen.displayFront, chosen.displaySide);
    // angles summary combine
    let summaryHtml = `<strong>Angles (chosen sample t=${Math.round(chosen.t)}s):</strong><div style="margin-top:8px">`;
    summaryHtml += `<div><em>Front 2D:</em> ${formatAngles(chosen.anglesFront2D)}</div>`;
    summaryHtml += `<div><em>Side 2D:</em> ${formatAngles(chosen.anglesSide2D)}</div>`;
    summaryHtml += `<div><em>Fused 3D:</em> ${formatAngles(chosen.angles3D)}</div>`;
    summaryHtml += `</div>`;
    anglesSummaryDiv.innerHTML = summaryHtml;
  } else {
    resultsTableDiv.innerHTML = '<div style="color:#c33">No results</div>';
  }

  // attach results to csv button closure
  latestResults = results;

  // finalize UI
  runBtn.disabled = false;
  if(cancelRequested){
    const cb = document.getElementById('cancelProcessBtn');
    if(cb){ cb.textContent = 'Cancelled'; cb.disabled = true; setTimeout(()=>cb.remove(),2000); }
  } else {
    const cb = document.getElementById('cancelProcessBtn');
    if(cb){ cb.textContent = 'Done'; cb.disabled = true; setTimeout(()=>cb.remove(),2000); }
  }
  modelStatus.textContent = 'Idle.';
});

/* allow CSV export access */
let latestResults = [];

/* End of app.js */

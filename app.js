// app.js
// MoveNet web UI logic: padding, detection, scaled render, integer coords table

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

let detector = null;
const runBtn = document.getElementById('runBtn');
const modelStatus = document.getElementById('modelStatus');

const frontInput = document.getElementById('frontInput');
const sideInput = document.getElementById('sideInput');
const frontPreviewImg = document.getElementById('frontPreviewImg');
const sidePreviewImg = document.getElementById('sidePreviewImg');

const frontResultBox = document.getElementById('frontResultBox');
const sideResultBox  = document.getElementById('sideResultBox');
const resultsTableDiv = document.getElementById('resultsTable');

// helper: read file to dataURL and set preview
function setPreviewFromFile(file, imgElement){
  if(!file) return;
  const url = URL.createObjectURL(file);
  imgElement.src = url;
}

// wire file inputs: replace preview image when user selects file
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

function updateRunButtonState(){
  const any = frontInput.files.length || sideInput.files.length || frontPreviewImg.src.includes('examples/') || sidePreviewImg.src.includes('examples/');
  // only enable if model loaded and at least one image present
  runBtn.disabled = !detector || (!frontInput.files.length && !sideInput.files.length && !frontPreviewImg.src && !sidePreviewImg.src && !sidePreviewImg.src);
}

// ------------------------------------------------------------------
// load MoveNet model (client-side TFJS pose-detection)
// ------------------------------------------------------------------
async function loadModel(){
  modelStatus.textContent = 'Loading model...';
  try{
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );
    modelStatus.textContent = 'Model loaded';
    runBtn.disabled = false;
  }catch(err){
    console.error('model load error', err);
    modelStatus.textContent = 'Model load failed (check console)';
  }
}
loadModel();

// ------------------------------------------------------------------
// pad image to square (offscreen) and return canvas + offsets
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// ensure keypoint x,y are pixels (if normalized 0..1 convert to pixels)
// returns array of {x,y,score}
// ------------------------------------------------------------------
function normalizeKeypoints(kps, padSize){
  return kps.map(k => {
    // some models return x,y in pixels already; others normalized (0..1)
    let x = k.x, y = k.y;
    if(x <= 1.01 && y <= 1.01){ x = x * padSize; y = y * padSize; }
    return { x: x, y: y, score: (k.score ?? 0) };
  });
}

// ------------------------------------------------------------------
// compute display scale and render scaled canvas inside result box
// we render the padded image scaled to fit the fixed result-box CSS dimensions
// returns scaled keypoints as integers
// ------------------------------------------------------------------
function renderResult(padCanvas, rawKps, resultBox){
  // calculate target display size (box CSS width / height)
  const style = getComputedStyle(resultBox);
  // note: CSS sets width/height in px via var; resultBox.clientWidth/Height reflect actual size
  const displayW = resultBox.clientWidth;
  const displayH = resultBox.clientHeight;

  // create canvas to insert
  const canvas = document.createElement('canvas');
  // set internal resolution to padded size scaled down to fit (preserve aspect)
  const scale = Math.min(displayW / padCanvas.width, displayH / padCanvas.height, 1.0);
  const outW = Math.max(1, Math.round(padCanvas.width * scale));
  const outH = Math.max(1, Math.round(padCanvas.height * scale));
  canvas.width = outW;
  canvas.height = outH;
  // CSS display will fill the box due to styles; but internal resolution is outW/outH
  canvas.style.width = '100%';
  canvas.style.height = '100%';

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,outW,outH);
  ctx.drawImage(padCanvas, 0, 0, outW, outH);

  // draw skeleton on top (use scaled coordinates)
  ctx.lineWidth = Math.max(2, Math.round(2 * scale));
  ctx.strokeStyle = 'lime';
  ctx.fillStyle = 'red';

  // convert rawKps to scaled coords
  const scaled = rawKps.map(k => ({
    x: Math.round(k.x * scale),
    y: Math.round(k.y * scale),
    score: k.score
  }));

  // draw keypoints (small circles) and skeleton
  scaled.forEach(pt => {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, Math.max(3, Math.round(4*scale)), 0, Math.PI*2);
    ctx.fill();
  });

  SKELETON.forEach(pair => {
    const a = scaled[pair[0]];
    const b = scaled[pair[1]];
    if(!a || !b) return;
    if((a.score || 0) < 0.05 || (b.score || 0) < 0.05) return;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  // attach canvas to resultBox (clear previous)
  resultBox.innerHTML = '';
  resultBox.appendChild(canvas);

  // return scaled integer coords with keypoint names
  return scaled.map((p,i) => ({ name: KEYPOINT_NAMES[i], x: Math.round(p.x), y: Math.round(p.y), score: p.score }));
}

// ------------------------------------------------------------------
// compute angle at B formed by A-B-C. returns number with 2 decimals (but we will not display decimals)
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

// ------------------------------------------------------------------
// table with integers - columns: Keypoint | Front co-or | Side co-or
// frontCoords and sideCoords arrays contain objects {name,x,y,score} or null
// ------------------------------------------------------------------
function buildResultsTable(frontCoords, sideCoords){
  let html = '<table><thead><tr><th>Keypoint</th><th>Front co-or</th><th>Side co-or</th></tr></thead><tbody>';
  for(let i=0;i<17;i++){
    const name = KEYPOINT_NAMES[i];
    const f = frontCoords && frontCoords[i] ? `${frontCoords[i].x}, ${frontCoords[i].y}` : '-';
    const s = sideCoords && sideCoords[i] ? `${sideCoords[i].x}, ${sideCoords[i].y}` : '-';
    html += `<tr><td>${name}</td><td>${f}</td><td>${s}</td></tr>`;
  }
  html += '</tbody></table>';
  return html;
}

// ------------------------------------------------------------------
// main run handler
// ------------------------------------------------------------------
runBtn.addEventListener('click', async () => {
  try{
    runBtn.disabled = true;
    runBtn.textContent = 'Running...';
    resultsTableDiv.innerHTML = '';

    // gather image elements (prefer uploaded file preview; if none, default example)
    const frontHas = frontInput.files && frontInput.files[0];
    const sideHas  = sideInput.files && sideInput.files[0];

    if(!frontHas && !sideHas){
      alert('Upload at least one image (front or side).');
      runBtn.disabled = false;
      runBtn.textContent = 'Run';
      return;
    }

    let frontCoords = null, sideCoords = null;

    // ---------- FRONT ----------
    if(frontHas){
      // create an image element and wait for decode
      const file = frontInput.files[0];
      const img = new Image();
      img.src = URL.createObjectURL(file);
      await img.decode();

      const pad = padToSquare(img);
      // run detection on pad.canvas (full res)
      const poses = await detector.estimatePoses(pad.canvas);
      if(!poses || poses.length === 0) {
        alert('No pose detected on front image.');
      } else {
        const raw = normalizeKeypoints(poses[0].keypoints, pad.size);
        // render result into frontResultBox
        frontCoords = renderResult(pad.canvas, raw, frontResultBox);
      }
    } else {
      // user didn't upload front; but maybe preview contains default sample - still allow detection using preview image element
      // We'll run only if preview image isn't blank and the user wants
      const imgEl = frontPreviewImg;
      if(imgEl && imgEl.src){
        // if default image path (examples) we still process
        const pad = padToSquare(imgEl);
        const poses = await detector.estimatePoses(pad.canvas);
        if(poses && poses.length) {
          const raw = normalizeKeypoints(poses[0].keypoints, pad.size);
          frontCoords = renderResult(pad.canvas, raw, frontResultBox);
        }
      }
    }

    // ---------- SIDE ----------
    if(sideHas){
      const file = sideInput.files[0];
      const img = new Image();
      img.src = URL.createObjectURL(file);
      await img.decode();

      const pad = padToSquare(img);
      const poses = await detector.estimatePoses(pad.canvas);
      if(!poses || poses.length === 0) {
        alert('No pose detected on side image.');
      } else {
        const raw = normalizeKeypoints(poses[0].keypoints, pad.size);
        sideCoords = renderResult(pad.canvas, raw, sideResultBox);
      }
    } else {
      const imgEl = sidePreviewImg;
      if(imgEl && imgEl.src){
        const pad = padToSquare(imgEl);
        const poses = await detector.estimatePoses(pad.canvas);
        if(poses && poses.length) {
          const raw = normalizeKeypoints(poses[0].keypoints, pad.size);
          sideCoords = renderResult(pad.canvas, raw, sideResultBox);
        }
      }
    }

    // build table (integer coords)
    resultsTableDiv.innerHTML = buildResultsTable(frontCoords, sideCoords);

  }catch(err){
    console.error(err);
    alert('Error during detection — see console.');
  }finally{
    runBtn.disabled = false;
    runBtn.textContent = 'Run';
  }
});

// enable run when model loaded and at least one preview exists
const observer = new MutationObserver(()=>updateRunState());
function updateRunState(){
  const hasFile = (frontInput.files && frontInput.files.length) || (sideInput.files && sideInput.files.length);
  const hasPreview = (frontPreviewImg && frontPreviewImg.src) || (sidePreviewImg && sidePreviewImg.src);
  runBtn.disabled = !(detector && (hasFile || hasPreview));
}
function isModelReady(){ return !!detector; }
setInterval(()=>{ if(detector) updateRunState(); }, 800);

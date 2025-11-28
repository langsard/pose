// app.js - Dual-view MoveNet Posture Analyzer

// Config
const CANVAS_SIZE = 512;
const KP_NAMES = [
  "nose","left_eye","right_eye","left_ear","right_ear",
  "left_shoulder","right_shoulder","left_elbow","right_elbow",
  "left_wrist","right_wrist","left_hip","right_hip",
  "left_knee","right_knee","left_ankle","right_ankle"
];

// Angle definitions: [A, B (vertex), C]
const ANGLE_DEFS = {
  "Left Elbow": ["left_shoulder","left_elbow","left_wrist"],
  "Right Elbow": ["right_shoulder","right_elbow","right_wrist"],
  "Left Shoulder": ["left_elbow","left_shoulder","left_hip"],
  "Right Shoulder": ["right_elbow","right_shoulder","right_hip"],
  "Left Knee": ["left_hip","left_knee","left_ankle"],
  "Right Knee": ["right_hip","right_knee","right_ankle"],
  "Left Hip": ["left_shoulder","left_hip","left_knee"],
  "Right Hip": ["right_shoulder","right_hip","right_knee"]
};

// UI refs
const frontFileEl = document.getElementById('frontFile');
const sideFileEl  = document.getElementById('sideFile');
const runBtn = document.getElementById('runBtn');
const confRange = document.getElementById('confRange');
const confLabel = document.getElementById('confLabel');

const frontCanvas = document.getElementById('frontCanvas');
const sideCanvas  = document.getElementById('sideCanvas');
const frontKPbox  = document.getElementById('frontKP');
const sideKPbox   = document.getElementById('sideKP');
const reportDiv   = document.getElementById('report');

// TF Pose detector
let detector = null;
async function loadModel(){
  setStatus('loading model...');
  detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, { modelType: poseDetection.movenet.modelType.THUNDER });
  setStatus('model loaded');
}
loadModel();

function setStatus(s){ console.log(s); }

// Update conf label
confRange.addEventListener('input', ()=> {
  confLabel.textContent = (Number(confRange.value)/100).toFixed(2);
});

// Read file to Image
function fileToImage(file){
  return new Promise((res, rej)=>{
    if (!file) return rej(new Error('no file'));
    const img = new Image();
    img.onload = ()=> res(img);
    img.onerror = e => rej(e);
    img.src = URL.createObjectURL(file);
  });
}

// Pad image to square canvas (no squeeze) and draw it into provided canvas.
// Returns pad object: { scale, offsetX, offsetY, srcWidth, srcHeight }
function drawImagePaddedToCanvas(img, canvas){
  const ctx = canvas.getContext('2d');
  // size to place original image into a square canvas WITHOUT distortion
  const srcW = img.naturalWidth, srcH = img.naturalHeight;
  const size = Math.max(srcW, srcH);
  const scale = CANVAS_SIZE / size;
  const destW = Math.round(srcW * scale), destH = Math.round(srcH * scale);
  const offsetX = Math.round((CANVAS_SIZE - destW)/2);
  const offsetY = Math.round((CANVAS_SIZE - destH)/2);

  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;

  // optional black background to show padding clearly
  ctx.fillStyle = '#111';
  ctx.fillRect(0,0,CANVAS_SIZE,CANVAS_SIZE);

  ctx.drawImage(img, offsetX, offsetY, destW, destH);
  return { scale, offsetX, offsetY, srcW, srcH, destW, destH };
}

// If MoveNet returns normalized coords (<=1), convert to pixel coords, else leave as-is
function ensurePixelCoords(kp, canvas){
  const w = canvas.width, h = canvas.height;
  if (kp.x <= 1 && kp.y <= 1) {
    return { x: kp.x * w, y: kp.y * h, score: kp.score, name: kp.name ?? '' };
  } else {
    return { x: kp.x, y: kp.y, score: kp.score, name: kp.name ?? '' };
  }
}

// Draw skeleton and keypoints ON SAME canvas
function drawPoseOnCanvas(keypoints, canvas, confThresh){
  const ctx = canvas.getContext('2d');
  // draw transparent overlay text etc
  const adj = keypoints.map(kp => ensurePixelCoords(kp, canvas));
  // Edges using utility adjacency pairs
  const edges = poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet);

  ctx.lineWidth = 3;
  ctx.strokeStyle = 'lime';
  ctx.fillStyle = 'yellow';

  // Draw edges
  edges.forEach(([a,b])=>{
    const p = adj[a], q = adj[b];
    if (!p || !q) return;
    if ((p.score||0) >= confThresh && (q.score||0) >= confThresh){
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
    }
  });

  // Draw keypoint dots and indices
  ctx.font = '10px monospace';
  adj.forEach((p, i)=>{
    ctx.beginPath();
    ctx.fillStyle = (p.score||0) >= confThresh ? 'cyan' : 'gray';
    ctx.arc(p.x, p.y, 4, 0, Math.PI*2);
    ctx.fill();
    ctx.fillStyle = 'white';
    ctx.fillText(String(i), p.x+6, p.y-6);
  });

  // return adj pixel coords for downstream
  return adj;
}

// angle helper (A-B-C angle at B)
function angleAtB(A,B,C){
  const ABx = A.x - B.x, ABy = A.y - B.y;
  const CBx = C.x - B.x, CBy = C.y - B.y;
  const mag1 = Math.hypot(ABx,ABy), mag2 = Math.hypot(CBx,CBy);
  if (mag1 === 0 || mag2 === 0) return null;
  let cos = (ABx*CBx + ABy*CBy) / (mag1*mag2);
  cos = Math.max(-1, Math.min(1, cos));
  return Math.acos(cos) * 180 / Math.PI;
}

// find index by name
function idx(name){
  return KP_NAMES.indexOf(name);
}

// compute angles for a kp array (pixel coords)
function computeAnglesFromKPs(kps){
  // kps expected as array with x,y,score,name
  const results = {};
  for (const [label, triple] of Object.entries(ANGLE_DEFS)){
    const A = kps[idx(triple[0])], B = kps[idx(triple[1])], C = kps[idx(triple[2])];
    if (!A || !B || !C) { results[label] = { angle:null, conf:0 }; continue; }
    const a = angleAtB(A,B,C);
    const conf = Math.min(A.score||0, B.score||0, C.score||0);
    results[label] = { angle: a===null ? null : Number(a.toFixed(4)), conf: Number(conf.toFixed(4)) };
  }
  return results;
}

// compute simple proportions (px)
function computeProportions(kps){
  const leftShoulder = kps[idx('left_shoulder')], rightShoulder = kps[idx('right_shoulder')];
  const leftHip = kps[idx('left_hip')], rightHip = kps[idx('right_hip')];
  const leftWrist = kps[idx('left_wrist')], rightWrist = kps[idx('right_wrist')];
  const leftAnkle = kps[idx('left_ankle')], rightAnkle = kps[idx('right_ankle')];

  function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }

  const arm = ((dist(leftShoulder,leftWrist) + dist(rightShoulder,rightWrist))/2) || null;
  const leg = ((dist(leftHip,leftAnkle) + dist(rightHip,rightAnkle))/2) || null;
  const torso = (dist({x:(leftShoulder.x+rightShoulder.x)/2, y:(leftShoulder.y+rightShoulder.y)/2},
                      {x:(leftHip.x+rightHip.x)/2, y:(leftHip.y+rightHip.y)/2})) || null;

  return { arm: arm ? Number(arm.toFixed(4)) : null, leg: leg ? Number(leg.toFixed(4)) : null, torso: torso ? Number(torso.toFixed(4)) : null };
}

// normalize pose (center and scale) — returns nx,ny values
function normalizePose(kps){
  const xs = kps.map(k => k.x), ys = kps.map(k => k.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const cx = (minX + maxX)/2, cy = (minY + maxY)/2;
  const scale = Math.max(maxX-minX, maxY-minY) || 1;
  return kps.map(k => ({ name: k.name, nx: Number(((k.x - cx)/scale).toPrecision(4)), ny: Number(((k.y - cy)/scale).toPrecision(4)), score: Number((k.score||0).toPrecision(4)) }));
}


// choose best view per angle (based on triple min confidence)
function chooseBestPerAngle(frontAngles, sideAngles){
  const chosen = {};
  for (const key of Object.keys(ANGLE_DEFS)){
    const f = frontAngles[key], s = sideAngles[key];
    const fConf = f ? f.conf : 0, sConf = s ? s.conf : 0;
    if (fConf >= sConf) chosen[key] = { view: 'front', angle: f.angle, conf: f.conf };
    else chosen[key] = { view: 'side', angle: s.angle, conf: s.conf };
  }
  return chosen;
}

// pretty number: 4 significant digits (or "-" if null)
function fmt4(x){
  if (x === null || x === undefined || Number.isNaN(x)) return '-';
  return Number(x).toPrecision(4);
}

// Build HTML report table
function buildReportHTML(frontAng, sideAng, chosen, proportions, normFront, normSide){
  let html = '<h3>Angles & Best-view selection</h3>';
  html += '<table><thead><tr><th>Joint</th><th>Front (deg/conf)</th><th>Side (deg/conf)</th><th>Chosen</th></tr></thead><tbody>';
  for (const joint of Object.keys(ANGLE_DEFS)){
    const f = frontAng[joint], s = sideAng[joint], c = chosen[joint];
    html += `<tr><td>${joint}</td>
      <td>${f.angle===null?'-':fmt4(f.angle)} / ${fmt4(f.conf)}</td>
      <td>${s.angle===null?'-':fmt4(s.angle)} / ${fmt4(s.conf)}</td>
      <td>${c.view} ${c.angle===null?'-':fmt4(c.angle)} / ${fmt4(c.conf)}</td></tr>`;
  }
  html += `</tbody></table>`;

  html += '<h3>Proportions (px)</h3>';
  html += '<table><tr><th>Arm</th><th>Leg</th><th>Torso</th></tr>';
  html += `<tr><td>${fmt4(proportions.arm)}</td><td>${fmt4(proportions.leg)}</td><td>${fmt4(proportions.torso)}</td></tr>`;
  html += '</table>';

  html += '<h3>Normalized keypoints (front / side) — nx, ny</h3>';
  html += '<div style="display:flex;gap:12px">';
  html += '<div style="flex:1"><strong>Front</strong><pre style="white-space:pre-wrap">';
  html += normFront.map(k=>`${k.name.padEnd(16)} nx:${k.nx.toPrecision(4)} ny:${k.ny.toPrecision(4)} sc:${k.score.toPrecision(4)}`).join('\n');
  html += '</pre></div>';
  html += '<div style="flex:1"><strong>Side</strong><pre style="white-space:pre-wrap">';
  html += normSide.map(k=>`${k.name.padEnd(16)} nx:${k.nx.toPrecision(4)} ny:${k.ny.toPrecision(4)} sc:${k.score.toPrecision(4)}`).join('\n');
  html += '</pre></div></div>';

  return html;
}


// PROCESS pipeline for one file
async function processFileToCanvas(file, canvas){
  const img = await fileToImage(file);
  const pad = drawImagePaddedToCanvas(img, canvas);

  // run MoveNet on the padded canvas (so keypoints are pixel coords on this canvas)
  const poses = await detector.estimatePoses(canvas, { flipHorizontal: false });
  if (!poses || poses.length === 0) return null;

  const rawKps = poses[0].keypoints.map((kp, i) => {
    // ensure name present, and convert to pixel coords if necessary
    const kpName = KP_NAMES[i] ?? (kp.name ?? `kp${i}`);
    let x = kp.x, y = kp.y;
    // sometimes model returns normalized coords (<=1) or pixel coords. Convert if normalized.
    if (x <= 1 && y <= 1) { x = x * canvas.width; y = y * canvas.height; }
    return { name: kpName, x: Number(x), y: Number(y), score: Number((kp.score ?? kp.score ?? 0).toFixed(4)) };
  });

  // draw skeleton on same canvas using current confidence threshold
  const confThresh = Number(confRange.value)/100;
  drawPoseOnCanvas(rawKps, canvas, confThresh);

  return { kps: rawKps, pad };
}


// MAIN: run button
runBtn.addEventListener('click', async ()=>{
  try {
    if (!detector) { alert('Model not loaded yet — wait a moment'); return; }
    if (!frontFileEl.files[0] || !sideFileEl.files[0]) { alert('Upload BOTH front and side images'); return; }

    setStatus('processing front...');
    const frontRes = await processFileToCanvas(frontFileEl.files[0], frontCanvas);
    setStatus('processing side...');
    const sideRes  = await processFileToCanvas(sideFileEl.files[0], sideCanvas);

    if (!frontRes || !sideRes){ alert('Pose not detected in at least one view'); return; }

    // show readable keypoint tables
    frontKPbox.textContent = frontRes.kps.map(k => `${k.name.padEnd(16)} x:${k.x.toFixed(1).padStart(7)} y:${k.y.toFixed(1).padStart(7)} sc:${k.score.toPrecision(4)}`).join('\n');
    sideKPbox.textContent  = sideRes.kps.map(k => `${k.name.padEnd(16)} x:${k.x.toFixed(1).padStart(7)} y:${k.y.toFixed(1).padStart(7)} sc:${k.score.toPrecision(4)}`).join('\n');

    // compute angles & proportions
    const frontAngles = computeAnglesFromKPs(frontRes.kps);
    const sideAngles  = computeAnglesFromKPs(sideRes.kps);

    // choose best view per angle
    const chosen = chooseBestPerAngle(frontAngles, sideAngles);

    // Merge per-joint best keypoints for aggregated proportions/angles (simple)
    // For proportions and normalization we choose the view with higher overall mean confidence
    const meanConf = arr => arr.reduce((s,k)=>s+(k.score||0),0)/arr.length;
    const preferFront = meanConf(frontRes.kps) >= meanConf(sideRes.kps);
    const mergedKps = preferFront ? frontRes.kps : sideRes.kps;

    const proportions = computeProportions(mergedKps);
    const normFront = normalizePose(frontRes.kps);
    const normSide  = normalizePose(sideRes.kps);

    // Build report HTML with 4 significant digits
    reportDiv.innerHTML = buildReportHTML(frontAngles, sideAngles, chosen, proportions, normFront, normSide);
    setStatus('done');
  } catch (err) {
    console.error(err);
    alert('Error: ' + (err.message || err));
  }
});

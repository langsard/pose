// app.js - improved: runs detection at full res, displays scaled canvases to fit screen

// ---------------------------
// Config / Globals
// ---------------------------
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

// UI refs
const frontInput = document.getElementById("frontInput");
const sideInput  = document.getElementById("sideInput");
const runBtn     = document.getElementById("detectBtn");
const frontCanvasElem = document.getElementById("frontCanvas");
const sideCanvasElem  = document.getElementById("sideCanvas");
const resultsDiv = document.getElementById("results");

// Disable button until model loads
runBtn.disabled = true;

// ---------------------------
// Load MoveNet model
// ---------------------------
async function loadModel(){
  try {
    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );
    console.log("MoveNet loaded");
    runBtn.disabled = false;
  } catch (e){
    console.error("Failed to load model:", e);
    alert("Model failed to load. Check console.");
  }
}
loadModel();

// ---------------------------
// pad image to square (offscreen canvas)
// returns {padCanvas, offsetX, offsetY, size, originalW, originalH}
// ---------------------------
function padToSquareImage(img){
  const originalW = img.naturalWidth || img.width;
  const originalH = img.naturalHeight || img.height;
  const size = Math.max(originalW, originalH);
  const padCanvas = document.createElement('canvas');
  padCanvas.width = size;
  padCanvas.height = size;
  const ctx = padCanvas.getContext('2d');

  // white bg
  ctx.fillStyle = 'white';
  ctx.fillRect(0,0,size,size);

  const offsetX = Math.round((size - originalW) / 2);
  const offsetY = Math.round((size - originalH) / 2);

  ctx.drawImage(img, offsetX, offsetY, originalW, originalH);
  return { padCanvas, offsetX, offsetY, size, originalW, originalH };
}

// ---------------------------
// calculate display scale so canvas fits viewport nicely
// keeps aspect (square)
// returns scale (<=1)
// ---------------------------
function calcDisplayScale(originalSize){
  // target constraints
  const maxWidthPerCanvas = Math.max(240, Math.floor(window.innerWidth * 0.45));  // at least 240px
  const maxHeightPerCanvas = Math.floor(window.innerHeight * 0.45); // e.g. 45vh

  const scaleW = maxWidthPerCanvas / originalSize;
  const scaleH = maxHeightPerCanvas / originalSize;
  const scale = Math.min(scaleW, scaleH, 1.0);
  return Math.max(scale, 0.15); // avoid extremely tiny; 15% minimum
}

// ---------------------------
// draw scaled padCanvas into a visible canvas and draw skeleton scaled
// rawKps: array of keypoints in padCanvas pixel coordinates (x,y,score)
// visibleCanvasElem: DOM canvas element to render into
// returns scaledKeypoints (coords relative to visible canvas)
// ---------------------------
function renderScaled(padCanvas, rawKps, visibleCanvasElem, displayScale){
  const visW = Math.round(padCanvas.width * displayScale);
  const visH = Math.round(padCanvas.height * displayScale);

  // set visible canvas internal resolution to scaled size
  visibleCanvasElem.width = visW;
  visibleCanvasElem.height = visH;

  // also make sure CSS width is 100% of parent for responsiveness (styles.css should handle)
  visibleCanvasElem.style.width = '100%'; // let CSS control visual sizing within layout
  visibleCanvasElem.style.height = 'auto';

  const ctx = visibleCanvasElem.getContext('2d');
  // draw scaled image
  ctx.clearRect(0,0,visW,visH);
  ctx.drawImage(padCanvas, 0, 0, visW, visH);

  // scale keypoints
  const scaledKps = rawKps.map((kp, i) => ({
    index: i,
    name: KEYPOINT_NAMES[i] ?? (`kp${i}`),
    x: kp.x * displayScale,
    y: kp.y * displayScale,
    score: kp.score
  }));

  // draw skeleton and points
  ctx.lineWidth = Math.max(2, Math.round(2 * displayScale));
  ctx.strokeStyle = 'lime';
  ctx.fillStyle = 'red';
  scaledKps.forEach(p=>{
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(3, 4 * displayScale), 0, Math.PI*2);
    ctx.fill();
  });

  SKELETON.forEach(([a,b])=>{
    const A = scaledKps[a], B = scaledKps[b];
    if (!A || !B) return;
    if ((A.score||0) > 0.15 && (B.score||0) > 0.15){
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(B.x, B.y);
      ctx.stroke();
    }
  });

  return scaledKps;
}

// ---------------------------
// angle helpers
// returns number as string with 2 decimals or '-' if invalid
// ---------------------------
function angleAtB(A,B,C){
  if (!A||!B||!C) return '-';
  const ABx = A.x - B.x, ABy = A.y - B.y;
  const CBx = C.x - B.x, CBy = C.y - B.y;
  const mag1 = Math.hypot(ABx, ABy), mag2 = Math.hypot(CBx, CBy);
  if (mag1 < 1e-6 || mag2 < 1e-6) return '-';
  let cosv = (ABx*CBx + ABy*CBy) / (mag1*mag2);
  cosv = Math.max(-1, Math.min(1, cosv));
  const ang = Math.acos(cosv) * 180 / Math.PI;
  return ang.toFixed(2);
}

// ---------------------------
// core pipeline for one input file => returns object with scaled keypoints and original keypoints
// ---------------------------
async function processFileToScaled(file, visibleCanvasElem){
  // load image element
  const img = new Image();
  img.src = URL.createObjectURL(file);
  await img.decode();

  // create padded offscreen canvas
  const padInfo = padToSquareImage(img);
  const padCanvas = padInfo.padCanvas;

  // run detector on padCanvas (full-res)
  const poses = await detector.estimatePoses(padCanvas, { flipHorizontal: false });
  if (!poses || poses.length === 0) {
    throw new Error('No pose detected');
  }

  // MoveNet v0 returns keypoints array with x,y in pixels relative to input image (padCanvas)
  const raw = poses[0].keypoints.map((kp,i) => ({
    x: kp.x, y: kp.y, score: (kp.score ?? 0)
  }));

  // compute display scale
  const displayScale = calcDisplayScale(padCanvas.width);

  // render scaled image + skeleton and get scaled keypoints
  const scaledKps = renderScaled(padCanvas, raw, visibleCanvasElem, displayScale);

  // We return both scaled and raw (raw useful for measurements in pixel-space if needed)
  return { scaledKps, rawKps: raw, padSize: padCanvas.width, displayScale };
}

// ---------------------------
// prepare table HTML (2-decimals)
function makeResultsTable(frontResult, sideResult){
  // build keypoint rows
  const rows = [];
  for (let i=0;i<17;i++){
    const name = KEYPOINT_NAMES[i] ?? `kp${i}`;

    let frontXY = '-';
    let sideXY = '-';
    let fconf = '-';
    let sconf = '-';

    if (frontResult && frontResult.scaledKps[i]){
      const p = frontResult.scaledKps[i];
      frontXY = `${p.x.toFixed(2)}, ${p.y.toFixed(2)}`;
      fconf = p.score.toFixed(2);
    }
    if (sideResult && sideResult.scaledKps[i]){
      const p = sideResult.scaledKps[i];
      sideXY = `${p.x.toFixed(2)}, ${p.y.toFixed(2)}`;
      sconf = p.score.toFixed(2);
    }

    rows.push(`<tr>
      <td>${name}</td>
      <td>${frontXY} <div class="muted">(${fconf})</div></td>
      <td>${sideXY} <div class="muted">(${sconf})</div></td>
    </tr>`);
  }

  // compute angles (choose best view: elbows from front, knees from side as simple heuristic)
  function computeAnglesFromScaled(result){
    if (!result) return null;
    const k = result.scaledKps;
    return {
      leftElbow: angleAtB(k[5], k[7], k[9]),
      rightElbow: angleAtB(k[6], k[8], k[10]),
      leftKnee: angleAtB(k[11], k[13], k[15]),
      rightKnee: angleAtB(k[12], k[14], k[16])
    };
  }

  const anglesF = computeAnglesFromScaled(frontResult);
  const anglesS = computeAnglesFromScaled(sideResult);

  // Choose best: prefer whichever has higher average confidence for involved triple
  function pick(angleName, tripleIndices){
    const [a,b,c] = tripleIndices;
    let fVal = '-', sVal = '-', chosen='-';
    if (anglesF) fVal = anglesF[angleName];
    if (anglesS) sVal = anglesS[angleName];
    // pick by presence
    if (fVal !== '-' && sVal !== '-'){
      // use mean score as tiebreak
      const fmean = (frontResult.scaledKps[a].score + frontResult.scaledKps[b].score + frontResult.scaledKps[c].score)/3;
      const smean = (sideResult.scaledKps[a].score + sideResult.scaledKps[b].score + sideResult.scaledKps[c].score)/3;
      chosen = (fmean >= smean) ? `front (${fVal})` : `side (${sVal})`;
    } else if (fVal !== '-') chosen = `front (${fVal})`;
    else if (sVal !== '-') chosen = `side (${sVal})`;
    else chosen = '-';
    return chosen;
  }

  const anglesSummary = {
    Left_Elbow: pick('leftElbow', [5,7,9]),
    Right_Elbow: pick('rightElbow', [6,8,10]),
    Left_Knee: pick('leftKnee', [11,13,15]),
    Right_Knee: pick('rightKnee', [12,14,16])
  };

  const html = `
    <div style="overflow:auto">
      <table style="width:100%; border-collapse:collapse">
        <thead><tr><th>Keypoint</th><th>Front (x,y) [conf]</th><th>Side (x,y) [conf]</th></tr></thead>
        <tbody>${rows.join('')}</tbody>
      </table>
      <h3>Selected Angles (2-decimals)</h3>
      <pre>${JSON.stringify(anglesSummary, null, 2)}</pre>
    </div>
  `;
  return html;
}

// ---------------------------
// MAIN button handler
// ---------------------------
runBtn.addEventListener('click', async () => {
  try {
    runBtn.disabled = true;
    runBtn.textContent = 'Running...';
    resultsDiv.innerHTML = '';

    if (!detector) throw new Error('Model not loaded yet.');

    const frontFile = frontInput.files[0];
    const sideFile  = sideInput.files[0];
    if (!frontFile || !sideFile) throw new Error('Upload both front and side images.');

    // process both files (they run detection at full res, then render scaled)
    const frontRes = await processFileToScaled(frontFile, frontCanvasElem);
    const sideRes  = await processFileToScaled(sideFile, sideCanvasElem);

    // compose table and angles
    const html = makeResultsTable(frontRes, sideRes);
    resultsDiv.innerHTML = html;

    runBtn.disabled = false;
    runBtn.textContent = 'Run Detection';
  } catch (err) {
    console.error(err);
    alert('Error: ' + (err.message || err));
    runBtn.disabled = false;
    runBtn.textContent = 'Run Detection';
  }
});

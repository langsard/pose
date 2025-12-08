/* app.js
   Single input (image/video) + Advanced 2-input (front+side) mode.
   Sampling dropdown: 5 / 10 / 30 / 60 sec.
   Uses TFJS pose-detection MoveNet (Thunder) for better accuracy (sparse sampling).
*/

// ---- DOM ----
const fileA = document.getElementById('fileA');
const fileB = document.getElementById('fileB');
const advToggle = document.getElementById('advToggle');
const advancedArea = document.getElementById('advancedArea');
const viewRole = document.getElementById('viewRole');
const samplingSel = document.getElementById('sampling');
const runBtn = document.getElementById('runBtn');
const status = document.getElementById('status');

const videoA = document.getElementById('videoA');
const imgA = document.getElementById('imgA');
const videoB = document.getElementById('videoB');
const imgB = document.getElementById('imgB');
const boxB = document.getElementById('boxB');

const samplesGrid = document.getElementById('samplesGrid');

let detector = null;
let inputA = null; // {type:'image'|'video', file, url}
let inputB = null;
let modelLoaded = false;

// keypoints indices for MoveNet (17)
const KEYPOINTS = [
  "nose","left_eye","right_eye","left_ear","right_ear",
  "left_shoulder","right_shoulder","left_elbow","right_elbow",
  "left_wrist","right_wrist","left_hip","right_hip",
  "left_knee","right_knee","left_ankle","right_ankle"
];
const SKELETON = [
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
];

// ---------------- load movenet model ----------------
async function loadModel() {
  try {
    status.innerText = 'Loading MoveNet...';
    if (typeof poseDetection === 'undefined') throw new Error('poseDetection not found');

    detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER, runtime: 'tfjs' }
    );

    modelLoaded = true;
    status.innerText = 'Model loaded ✓';
    runBtn.disabled = false;
  } catch (err) {
    console.error('Model error', err);
    status.innerText = 'Model load failed — check console';
    runBtn.disabled = true;
  }
}
loadModel();


// ---------------- UI wiring ----------------
advToggle.addEventListener('change', () => {
  if (advToggle.checked) {
    advancedArea.classList.remove('hidden');
    boxB.classList.remove('hidden');
  } else {
    advancedArea.classList.add('hidden');
    boxB.classList.add('hidden');
    fileB.value = '';
    videoB.src = '';
    imgB.src = '';
    inputB = null;
  }
});

fileA.addEventListener('change', async (e) => {
  inputA = await handleFileSelection(e.target.files && e.target.files[0], 'A');
});
fileB && fileB.addEventListener('change', async (e) => {
  inputB = await handleFileSelection(e.target.files && e.target.files[0], 'B');
});

// preview helpers
async function handleFileSelection(file, which) {
  if (!file) return null;
  const type = file.type.startsWith('image/') ? 'image' :
               file.type.startsWith('video/') ? 'video' : null;
  if (!type) {
    alert('Unsupported file type');
    return null;
  }
  const url = URL.createObjectURL(file);

  // show preview
  if (which === 'A') {
    if (type === 'image') {
      imgA.src = url; imgA.classList.remove('hidden'); videoA.classList.add('hidden'); videoA.pause();
    } else {
      videoA.src = url; videoA.classList.remove('hidden'); imgA.classList.add('hidden');
    }
  } else {
    if (type === 'image') {
      imgB.src = url; imgB.classList.remove('hidden'); videoB.classList.add('hidden'); videoB.pause();
    } else {
      videoB.src = url; videoB.classList.remove('hidden'); imgB.classList.add('hidden');
    }
  }

  return { file, type, url };
}

// ---------------- sampling utilities ----------------
function createCanvasFromVideoEl(videoEl) {
  const c = document.createElement('canvas');
  c.width = videoEl.videoWidth;
  c.height = videoEl.videoHeight;
  const ctx = c.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, c.width, c.height);
  return c;
}
function createCanvasFromImageEl(imgEl) {
  const c = document.createElement('canvas');
  c.width = imgEl.naturalWidth || imgEl.width;
  c.height = imgEl.naturalHeight || imgEl.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(imgEl, 0, 0, c.width, c.height);
  return c;
}

function seekTo(videoEl, time) {
  return new Promise((resolve, reject) => {
    function onSeeked() {
      videoEl.removeEventListener('seeked', onSeeked);
      resolve();
    }
    if (!videoEl.duration || isNaN(videoEl.duration)) {
      reject(new Error('metadata not loaded'));
      return;
    }
    videoEl.addEventListener('seeked', onSeeked);
    // clamp
    videoEl.currentTime = Math.min(time, Math.max(0, videoEl.duration - 0.001));
  });
}

// small sleep
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------------- angle math ----------------
// compute 2D/3D angle at B from A-B-C (vectors BA and BC)
function angleBetweenVectors3D(A,B,C) {
  if (!A || !B || !C) return null;
  const AB = [A.x - B.x, A.y - B.y, (A.z||0) - (B.z||0)];
  const CB = [C.x - B.x, C.y - B.y, (C.z||0) - (B.z||0)];
  const dot = AB[0]*CB[0] + AB[1]*CB[1] + AB[2]*CB[2];
  const magA = Math.hypot(AB[0],AB[1],AB[2]);
  const magC = Math.hypot(CB[0],CB[1],CB[2]);
  if (magA < 1e-6 || magC < 1e-6) return null;
  let cosv = dot/(magA*magC);
  cosv = Math.max(-1, Math.min(1, cosv));
  const rad = Math.acos(cosv);
  return rad * 180/Math.PI;
}

// ---------------- pseudo-3D fusion ----------------
function fuseKeypoints(frontKP, sideKP) {
  // Both arrays of keypoints (objects with x,y,score)
  // We'll produce an array of {x,y,z,scoreFront,scoreSide}
  // Normalize side x to same scale as front by using widths (we store raw pixels so we assume similar scale)
  // Better: normalize by shoulder-hip distance per image — we'll use front width as primary X scale
  if (!frontKP && !sideKP) return null;
  // Prepare sizes
  const frontW = frontKP ? Math.max(...frontKP.map(k=>k.x)) - Math.min(...frontKP.map(k=>k.x)) : 1;
  const sideW  = sideKP  ? Math.max(...sideKP.map(k=>k.x))  - Math.min(...sideKP.map(k=>k.x))  : 1;

  const fused = [];
  for (let i=0;i<KEYPOINTS.length;i++){
    const f = frontKP && frontKP[i] ? frontKP[i] : null;
    const s = sideKP  && sideKP[i]  ? sideKP[i]  : null;
    // choose Y as avg of available Y
    const yVals = [];
    if (f && f.score>0) yVals.push(f.y);
    if (s && s.score>0) yVals.push(s.y);
    const y = yVals.length ? (yVals.reduce((a,b)=>a+b,0)/yVals.length) : null;

    // x from front preferred
    const x = f && f.score>0 ? f.x : (s && s.score>0 ? s.x : null);

    // z from side.x scaled to front width: z = (side.x - centerSide) * (frontW/sideW)
    let z = null;
    if (s && s.score>0 && sideW>0) {
      const centerSide = (Math.max(...sideKP.map(k=>k.x)) + Math.min(...sideKP.map(k=>k.x)))/2;
      const centerFront = frontKP && frontKP.length ? (Math.max(...frontKP.map(k=>k.x)) + Math.min(...frontKP.map(k=>k.x)))/2 : 0;
      z = (s.x - centerSide) * (frontW / Math.max(sideW,1)); // pseudo depth
      // we could offset by centerFront but depth is relative so it's okay
    }

    fused.push({
      name: KEYPOINTS[i],
      x: x !== null ? Math.round(x) : null,
      y: y !== null ? Math.round(y) : null,
      z: z !== null ? Math.round(z) : null,
      scoreFront: f ? f.score : 0,
      scoreSide:  s ? s.score  : 0
    });
  }
  return fused;
}

// ---------------- draw skeleton helper ----------------
function drawSkeletonToCanvas(canvas, keypoints, opts={}) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  // background is already drawn
  ctx.fillStyle = opts.pointColor || 'red';
  ctx.strokeStyle = opts.lineColor || 'lime';
  ctx.lineWidth = Math.max(2, Math.round(Math.min(w,h)/200));

  // draw bones
  SKELETON.forEach(([a,b]) => {
    const pA = keypoints[a], pB = keypoints[b];
    if (!pA || !pB) return;
    if ((pA.score||0) < 0.05 || (pB.score||0) < 0.05) return;
    ctx.beginPath();
    ctx.moveTo(pA.x, pA.y);
    ctx.lineTo(pB.x, pB.y);
    ctx.stroke();
  });
  // points
  keypoints.forEach(p => {
    if (!p) return;
    if ((p.score||0) < 0.05) return;
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(3, Math.round(Math.min(w,h)/150)), 0, Math.PI*2);
    ctx.fill();
  });
}

// ---------------- main RUN logic ----------------
runBtn.addEventListener('click', async () => {
  if (!modelLoaded) { alert('Model not loaded'); return; }
  samplesGrid.innerHTML = '';
  status.innerText = 'Preparing...';

  // validate inputA
  if (!inputA) { alert('Please select input A'); return; }

  const samplingSec = parseInt(samplingSel.value,10) || 30;
  const adv = advToggle.checked;

  // prepare sources: for each input produce an object with type and element to draw from
  // for images: we will ensure the image element is loaded
  async function ensureReady(inp, which) {
    if (!inp) return null;
    if (inp.type === 'image') {
      // ensure loaded
      const el = which==='A' ? imgA : imgB;
      if (!el.complete) {
        await new Promise((r) => el.onload = r);
      }
      return { kind:'image', el };
    } else {
      // video
      const el = which==='A' ? videoA : videoB;
      // ensure metadata loaded
      if (!el.duration || isNaN(el.duration)) {
        await new Promise((resolve, reject) => {
          el.addEventListener('loadedmetadata', resolve, { once:true });
          setTimeout(()=>reject(new Error('video metadata timeout')), 5000);
        }).catch((e)=>{ /* ignore */ });
      }
      return { kind:'video', el };
    }
  }

  // ensure B is present if adv
  if (adv && !inputB) { alert('Advanced mode selected but second input missing'); return; }

  // ready objects
  let srcA = await ensureReady(inputA,'A');
  let srcB = adv ? await ensureReady(inputB,'B') : null;

  // determine which source is front or side based on viewRole and which file is A/B
  let frontSrc, sideSrc;
  if (adv) {
    if (viewRole.value === 'front') {
      frontSrc = srcA; sideSrc = srcB;
    } else {
      frontSrc = srcB; sideSrc = srcA;
    }
  } else {
    // single input: treat as 'front' only; sideSrc null
    frontSrc = srcA; sideSrc = null;
  }

  // determine sampling timestamps:
  // if both video: use duration = min(frontDur, sideDur) to avoid out-of-range
  let duration = 0;
  if (frontSrc.kind === 'video' && sideSrc && sideSrc.kind === 'video') {
    duration = Math.min(frontSrc.el.duration || 0, sideSrc.el.duration || 0);
  } else if (frontSrc.kind === 'video') {
    duration = frontSrc.el.duration || 0;
  } else if (sideSrc && sideSrc.kind === 'video') {
    duration = sideSrc.el.duration || 0;
  } else {
    duration = 0;
  }

  let timestamps = [];
  if (duration > 0) {
    // sample at 0, samplingSec, 2*samplingSec ... up to duration
    for (let t=0; t < duration; t += samplingSec) timestamps.push(Math.min(t, duration-0.001));
    // ensure at least one sample
    if (timestamps.length === 0) timestamps.push(Math.min(0, duration-0.001));
  } else {
    // no video present - single photo scenario
    timestamps = [0];
  }

  status.innerText = `Processing ${timestamps.length} sample(s)... (this may take a bit)`;
  const results = [];

  // sequential loop over timestamps
  for (let i=0;i<timestamps.length;i++){
    const t = timestamps[i];
    status.innerText = `Sample ${i+1}/${timestamps.length} — t=${Math.round(t)}s`;

    // produce canvases (frontCanvas, sideCanvas) for this timestamp
    let frontCanvas = null, sideCanvas = null;
    // front
    if (frontSrc.kind === 'image') {
      frontCanvas = createCanvasFromImageEl(frontSrc.el);
    } else {
      // video: seek and capture
      try {
        await seekTo(frontSrc.el, t);
      } catch(e){ console.warn('seek front failed',e); }
      // tiny wait for some browsers
      await sleep(80);
      frontCanvas = createCanvasFromVideoEl(frontSrc.el);
    }
    // side (if exists)
    if (sideSrc) {
      if (sideSrc.kind === 'image') {
        sideCanvas = createCanvasFromImageEl(sideSrc.el);
      } else {
        try {
          await seekTo(sideSrc.el, t);
        } catch(e){ console.warn('seek side failed',e); }
        await sleep(80);
        sideCanvas = createCanvasFromVideoEl(sideSrc.el);
      }
    }

    // Run MoveNet on canvases (front then side)
    let poseFront = null, poseSide = null;
    try {
      const pf = await detector.estimatePoses(frontCanvas, { maxPoses: 1, flipHorizontal: false });
      poseFront = (pf && pf.length) ? pf[0] : null;
    } catch (e) { console.error('front detect err',e); }
    if (sideCanvas) {
      try {
        const ps = await detector.estimatePoses(sideCanvas, { maxPoses: 1, flipHorizontal: false });
        poseSide = (ps && ps.length) ? ps[0] : null;
      } catch (e) { console.error('side detect err',e); }
    }

    // draw skeletons onto copies of canvases at their original resolution
    const frontOut = document.createElement('canvas'); frontOut.width = frontCanvas.width; frontOut.height = frontCanvas.height;
    const fctx = frontOut.getContext('2d'); fctx.drawImage(frontCanvas,0,0);
    if (poseFront && poseFront.keypoints) drawSkeletonToCanvas(frontOut, poseFront.keypoints);

    let sideOut = null;
    if (sideCanvas) {
      sideOut = document.createElement('canvas'); sideOut.width = sideCanvas.width; sideOut.height = sideCanvas.height;
      const sctx = sideOut.getContext('2d'); sctx.drawImage(sideCanvas,0,0);
      if (poseSide && poseSide.keypoints) drawSkeletonToCanvas(sideOut, poseSide.keypoints);
    }

    // fuse KPs into pseudo-3D
    const fused = fuseKeypoints(
      poseFront && poseFront.keypoints ? poseFront.keypoints : null,
      poseSide && poseSide.keypoints ? poseSide.keypoints : null
    );

    // compute 3D angles for elbows and knees
    // get helper by name -> fused index
    function idx(name){ return KEYPOINTS.indexOf(name); }
    const angles3D = {};
    if (fused) {
      // left elbow: shoulder - elbow - wrist
      const L_sh = fused[idx('left_shoulder')];
      const L_el = fused[idx('left_elbow')];
      const L_wr = fused[idx('left_wrist')];
      const R_sh = fused[idx('right_shoulder')];
      const R_el = fused[idx('right_elbow')];
      const R_wr = fused[idx('right_wrist')];
      const L_hip = fused[idx('left_hip')], L_knee = fused[idx('left_knee')], L_ank = fused[idx('left_ankle')];
      const R_hip = fused[idx('right_hip')], R_knee = fused[idx('right_knee')], R_ank = fused[idx('right_ankle')];

      function to3(p){ if(!p) return null; return {x:(p.x===null?null:p.x), y:(p.y===null?null:p.y), z:(p.z===null?0:p.z)}; }

      const L_el_ang = angleBetweenVectors3D(to3(L_sh), to3(L_el), to3(L_wr));
      const R_el_ang = angleBetweenVectors3D(to3(R_sh), to3(R_el), to3(R_wr));
      const L_kn_ang = angleBetweenVectors3D(to3(L_hip), to3(L_knee), to3(L_ank));
      const R_kn_ang = angleBetweenVectors3D(to3(R_hip), to3(R_knee), to3(R_ank));

      if (L_el_ang!==null) angles3D.leftElbow = Math.round(L_el_ang);
      if (R_el_ang!==null) angles3D.rightElbow = Math.round(R_el_ang);
      if (L_kn_ang!==null) angles3D.leftKnee  = Math.round(L_kn_ang);
      if (R_kn_ang!==null) angles3D.rightKnee = Math.round(R_kn_ang);
    }

    // push result
    results.push({
      t,
      frontOut,
      sideOut,
      fused,
      angles3D
    });

    // brief pause
    await sleep(60);
  } // end timestamps loop

  status.innerText = `Done. Rendering ${results.length} samples...`;
  // render cards
  samplesGrid.innerHTML = '';
  results.forEach((r, i) => {
    const card = document.createElement('div'); card.className = 'sample-card';
    const header = document.createElement('div'); header.className='meta-row';
    header.innerHTML = `<div><strong>Sample ${i+1}</strong> — t=${Math.round(r.t)}s</div>`;
    card.appendChild(header);

    const viewRow = document.createElement('div'); viewRow.className='view-row';
    const leftBox = document.createElement('div'); leftBox.className='preview-small';
    const leftCanvas = document.createElement('canvas'); leftCanvas.width = r.frontOut.width; leftCanvas.height = r.frontOut.height;
    leftCanvas.getContext('2d').drawImage(r.frontOut,0,0);
    leftBox.appendChild(leftCanvas);
    viewRow.appendChild(leftBox);

    if (r.sideOut) {
      const rightBox = document.createElement('div'); rightBox.className='preview-small';
      const rightCanvas = document.createElement('canvas'); rightCanvas.width = r.sideOut.width; rightCanvas.height = r.sideOut.height;
      rightCanvas.getContext('2d').drawImage(r.sideOut,0,0);
      rightBox.appendChild(rightCanvas);
      viewRow.appendChild(rightBox);
    }
    card.appendChild(viewRow);

    // angles + fused table
    const info = document.createElement('div');
    info.style.fontSize='13px';
    // angles
    const angParts = [];
    if (r.angles3D.leftElbow!==undefined) angParts.push(`L elbow ${r.angles3D.leftElbow}°`);
    if (r.angles3D.rightElbow!==undefined) angParts.push(`R elbow ${r.angles3D.rightElbow}°`);
    if (r.angles3D.leftKnee!==undefined)  angParts.push(`L knee ${r.angles3D.leftKnee}°`);
    if (r.angles3D.rightKnee!==undefined) angParts.push(`R knee ${r.angles3D.rightKnee}°`);
    const angDiv = document.createElement('div'); angDiv.innerHTML = `<strong>3D Angles:</strong> ${angParts.length?angParts.join(' · '): 'No angles'}`;
    info.appendChild(angDiv);

    // fused coords table (integer coords)
    if (r.fused && r.fused.length) {
      const tbl = document.createElement('table');
      tbl.style.width='100%';
      tbl.style.borderCollapse='collapse';
      tbl.style.fontSize='12px';
      tbl.innerHTML = `<thead><tr><th style="text-align:left;padding:6px">Keypoint</th><th style="padding:6px">X</th><th style="padding:6px">Y</th><th style="padding:6px">Z</th></tr></thead>`;
      const body = document.createElement('tbody');
      r.fused.forEach(k => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td style="padding:6px;border-top:1px solid #eee">${k.name}</td>
                        <td style="padding:6px;border-top:1px solid #eee">${k.x===null?'-':k.x}</td>
                        <td style="padding:6px;border-top:1px solid #eee">${k.y===null?'-':k.y}</td>
                        <td style="padding:6px;border-top:1px solid #eee">${k.z===null?'-':k.z}</td>`;
        body.appendChild(tr);
      });
      tbl.appendChild(body);
      info.appendChild(tbl);
    } else {
      const p = document.createElement('div'); p.innerText = 'No fused keypoints.';
      info.appendChild(p);
    }

    card.appendChild(info);

    // open full-size button
    const btnRow = document.createElement('div'); btnRow.style.display='flex'; btnRow.style.justifyContent='flex-end';
    const openBtn = document.createElement('button'); openBtn.textContent = 'Open Full';
    openBtn.onclick = () => {
      const w = window.open('');
      if (!w) { alert('Popup blocked'); return; }
      w.document.body.style.margin = '0';
      w.document.title = `Sample ${i+1}`;
      const c = document.createElement('div');
      c.style.display='flex'; c.style.gap='12px';
      const img1 = new Image(); img1.src = r.frontOut.toDataURL();
      c.appendChild(img1);
      if (r.sideOut) {
        const img2 = new Image(); img2.src = r.sideOut.toDataURL();
        c.appendChild(img2);
      }
      w.document.body.appendChild(c);
    };
    btnRow.appendChild(openBtn);
    card.appendChild(btnRow);

    samplesGrid.appendChild(card);
  });

  status.innerText = `Completed ${results.length} sample(s).`;
});

// ---------------- end of file ----------------

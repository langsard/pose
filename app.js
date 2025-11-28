// --------------------------------------------------
// GLOBALS
// --------------------------------------------------
let detector;

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

// --------------------------------------------------
// LOAD MOVENET MODEL
// --------------------------------------------------
async function loadModel() {
  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );
  console.log("✅ MoveNet model loaded");
}
loadModel();

// --------------------------------------------------
// IMAGE PADDING
// --------------------------------------------------
function padToSquare(img) {
  const maxSide = Math.max(img.width, img.height);
  const canvas = document.createElement("canvas");
  canvas.width = maxSide;
  canvas.height = maxSide;

  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, maxSide, maxSide);

  const offsetX = (maxSide - img.width) / 2;
  const offsetY = (maxSide - img.height) / 2;

  ctx.drawImage(img, offsetX, offsetY);

  return {
    canvas,
    offsetX,
    offsetY,
    originalW: img.width,
    originalH: img.height
  };
}

// --------------------------------------------------
// CONVERT KEYPOINT COORDS
// --------------------------------------------------
function convertCoords(rawPoints, padInfo) {
  return rawPoints.map((kp, i) => ({
    name: KEYPOINT_NAMES[i],
    x: kp.x * padInfo.originalW + padInfo.offsetX,
    y: kp.y * padInfo.originalH + padInfo.offsetY,
    score: kp.score
  }));
}

// --------------------------------------------------
// DRAW SKELETON
// --------------------------------------------------
function drawPose(ctx, keypoints) {
  ctx.fillStyle = "red";
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 3;

  // keypoints
  keypoints.forEach(kp => {
    ctx.beginPath();
    ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  // skeleton
  SKELETON.forEach(([a, b]) => {
    const pA = keypoints[a];
    const pB = keypoints[b];
    ctx.beginPath();
    ctx.moveTo(pA.x, pA.y);
    ctx.lineTo(pB.x, pB.y);
    ctx.stroke();
  });
}

// --------------------------------------------------
// ANGLE CALCULATION
// --------------------------------------------------
function angle3(a, b, c) {
  const ab = [a.x - b.x, a.y - b.y];
  const cb = [c.x - b.x, c.y - b.y];

  const dot = ab[0]*cb[0] + ab[1]*cb[1];
  const mag1 = Math.hypot(ab[0], ab[1]);
  const mag2 = Math.hypot(cb[0], cb[1]);
  const angle = Math.acos(dot / (mag1 * mag2 + 1e-6));

  return (angle * 180 / Math.PI).toFixed(2);
}

function computeAngles(kp) {
  return {
    leftElbow: angle3(kp[5], kp[7], kp[9]),
    rightElbow: angle3(kp[6], kp[8], kp[10]),
    leftKnee: angle3(kp[11], kp[13], kp[15]),
    rightKnee: angle3(kp[12], kp[14], kp[16])
  };
}

// --------------------------------------------------
// MAIN DETECTION
// --------------------------------------------------
document.getElementById("detectBtn").onclick = async () => {
  if (!detector) return alert("Model not loaded yet!");

  const frontFile = document.getElementById("frontInput").files[0];
  const sideFile = document.getElementById("sideInput").files[0];

  if (!frontFile || !sideFile) return alert("Upload both images first.");

  const frontImg = new Image();
  const sideImg = new Image();

  frontImg.src = URL.createObjectURL(frontFile);
  sideImg.src = URL.createObjectURL(sideFile);

  frontImg.onload = async () => {
    sideImg.onload = async () => {

      // ---------------- FRONT ----------------
      const padF = padToSquare(frontImg);
      const detF = await detector.estimatePoses(padF.canvas);
      let kf = convertCoords(detF[0].keypoints, padF);

      const frontCanvas = document.getElementById("frontCanvas");
      frontCanvas.width = padF.canvas.width;
      frontCanvas.height = padF.canvas.height;

      const ctxF = frontCanvas.getContext("2d");
      ctxF.drawImage(padF.canvas, 0, 0);
      drawPose(ctxF, kf);


      // ---------------- SIDE ----------------
      const padS = padToSquare(sideImg);
      const detS = await detector.estimatePoses(padS.canvas);
      let ks = convertCoords(detS[0].keypoints, padS);

      const sideCanvas = document.getElementById("sideCanvas");
      sideCanvas.width = padS.canvas.width;
      sideCanvas.height = padS.canvas.height;

      const ctxS = sideCanvas.getContext("2d");
      ctxS.drawImage(padS.canvas, 0, 0);
      drawPose(ctxS, ks);


      // ---------------- ANGLES ----------------
      const anglesF = computeAngles(kf);
      const anglesS = computeAngles(ks);

      const finalAngles = {
        leftElbow: anglesF.leftElbow,
        rightElbow: anglesF.rightElbow,
        leftKnee: anglesS.leftKnee,
        rightKnee: anglesS.rightKnee
      };


      // ---------------- TABLE ----------------
      let html = `<table>
        <tr><th>Keypoint</th><th>Front (x,y)</th><th>Side (x,y)</th></tr>`;

      for (let i = 0; i < 17; i++) {
        html += `
        <tr>
          <td>${KEYPOINT_NAMES[i]}</td>
          <td>${kf[i].x.toFixed(2)}, ${kf[i].y.toFixed(2)}</td>
          <td>${ks[i].x.toFixed(2)}, ${ks[i].y.toFixed(2)}</td>
        </tr>`;
      }
      html += "</table>";

      html += `<h3>Joint Angles</h3><pre>${JSON.stringify(finalAngles, null, 2)}</pre>`;

      document.getElementById("results").innerHTML = html;
    };
  };
};

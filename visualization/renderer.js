// MoveNet skeleton connections
const skeleton = [
  [5, 7], [7, 9],        // left arm
  [6, 8], [8, 10],       // right arm
  [5, 6],                // shoulders
  [5, 11], [6, 12],      // torso upper
  [11, 12],              // hips
  [11, 13], [13, 15],    // left leg
  [12, 14], [14, 16]     // right leg
];

// Risk color mapping
function riskColor(angle) {
  if (!angle) return "gray";

  if (angle > 150 || angle < 30) return "red";
  if (angle > 120 || angle < 60) return "yellow";
  return "green";
}

// Draw skeleton + colored joints
function drawSkeleton(ctx, keypoints, angles) {

  // Draw lines
  ctx.lineWidth = 3;
  ctx.strokeStyle = "white";

  skeleton.forEach(([a, b]) => {
    if (keypoints[a].score > 0.4 && keypoints[b].score > 0.4) {
      ctx.beginPath();
      ctx.moveTo(keypoints[a].x, keypoints[a].y);
      ctx.lineTo(keypoints[b].x, keypoints[b].y);
      ctx.stroke();
    }
  });

  // Draw joints
  keypoints.forEach((kp, index) => {
    if (kp.score > 0.4) {

      let color = "green";

      // Map joint to related angle
      if (index === 7 && angles.leftElbow) color = riskColor(angles.leftElbow);
      if (index === 8 && angles.rightElbow) color = riskColor(angles.rightElbow);
      if (index === 5 && angles.leftShoulder) color = riskColor(angles.leftShoulder);
      if (index === 6 && angles.rightShoulder) color = riskColor(angles.rightShoulder);
      if (index === 11 && angles.leftHip) color = riskColor(angles.leftHip);
      if (index === 12 && angles.rightHip) color = riskColor(angles.rightHip);
      if (index === 13 && angles.leftKnee) color = riskColor(angles.leftKnee);
      if (index === 14 && angles.rightKnee) color = riskColor(angles.rightKnee);

      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    }
  });
}

export function renderPreview(source, keypoints) {

  const canvas = document.getElementById("outputCanvas");
  const ctx = canvas.getContext("2d");

  canvas.width = source.width;
  canvas.height = source.height;

  ctx.drawImage(source, 0, 0);

  drawSkeleton(ctx, keypoints, {}); // preview without risk
}

export function renderResultModal(source, keypoints, angles) {

  const modal = document.getElementById("resultModal");
  const canvas = document.getElementById("resultCanvas");
  const ctx = canvas.getContext("2d");
  const table = document.getElementById("angleTable");
  const closeBtn = document.getElementById("closeResult");

  modal.classList.remove("hidden");

  canvas.width = source.width;
  canvas.height = source.height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(source, 0, 0);

  drawSkeleton(ctx, keypoints, angles);

  // ==========================
  // SIMPLE OWAS CLASSIFICATION
  // ==========================

  let back = 1;
  let arms = 1;
  let legs = 1;
  let load = 1;

  // ---------- BACK ----------

  const trunkAngle =
    ((angles.leftHip || 180) +
     (angles.rightHip || 180)) / 2;

  if (trunkAngle < 150)
    back = 2;

  // ---------- ARMS ----------

  const leftRaised =
    keypoints[9]?.y < keypoints[5]?.y;

  const rightRaised =
    keypoints[10]?.y < keypoints[6]?.y;

  if (leftRaised || rightRaised)
    arms = 2;

  if (leftRaised && rightRaised)
    arms = 3;

  // ---------- LEGS ----------

  const leftBent =
    (angles.leftKnee || 180) < 150;

  const rightBent =
    (angles.rightKnee || 180) < 150;

  if (leftBent && rightBent)
    legs = 4;

  // ---------- OWAS CODE ----------

  const owasCode =
    `${back}${arms}${legs}${load}`;

  table.innerHTML = `
    <h3>OWAS Assessment</h3>

    <div><b>Back:</b> ${back}</div>
    <div><b>Arms:</b> ${arms}</div>
    <div><b>Legs:</b> ${legs}</div>
    <div><b>Load:</b> ${load}</div>

    <div style="margin-top:10px;">
      <b>OWAS Code: ${owasCode}</b>
    </div>

    <hr>

    <h3>Joint Angles</h3>

    ${Object.entries(angles)
      .map(([k,v]) =>
        `<div>${k}: ${v.toFixed(1)}°</div>`
      )
      .join("")}
  `;

  closeBtn.onclick = () => {
    modal.classList.add("hidden");
  };
}

export function renderGallery(results) {

  const modal = document.getElementById("resultModal");
  const canvas = document.getElementById("resultCanvas");
  const ctx = canvas.getContext("2d");
  const table = document.getElementById("angleTable");
  const closeBtn = document.getElementById("closeResult");

  modal.classList.remove("hidden");

  let index = 0;

  function showFrame(i) {

    const frame = results[i];
    const img = new Image();

    img.onload = () => {

      canvas.width = img.width;
      canvas.height = img.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // REDRAW skeleton and risk colors
      drawSkeleton(ctx, frame.keypoints, frame.angles);

      table.innerHTML = `// ==========================
// SIMPLE OWAS CLASSIFICATION
// ==========================

let back = 1;
let arms = 1;
let legs = 1;
let load = 1;

const trunkAngle =
  ((frame.angles.leftHip || 180) +
   (frame.angles.rightHip || 180)) / 2;

if (trunkAngle < 150)
  back = 2;

const leftRaised =
  frame.keypoints[9]?.y <
  frame.keypoints[5]?.y;

const rightRaised =
  frame.keypoints[10]?.y <
  frame.keypoints[6]?.y;

if (leftRaised || rightRaised)
  arms = 2;

if (leftRaised && rightRaised)
  arms = 3;

const leftBent =
  (frame.angles.leftKnee || 180) < 150;

const rightBent =
  (frame.angles.rightKnee || 180) < 150;

if (leftBent && rightBent)
  legs = 4;

const owasCode =
  `${back}${arms}${legs}${load}`;

table.innerHTML = `
  <h3>Frame ${i+1} / ${results.length}</h3>

  <div><b>OWAS Code:</b> ${owasCode}</div>
  <div>Back: ${back}</div>
  <div>Arms: ${arms}</div>
  <div>Legs: ${legs}</div>
  <div>Load: ${load}</div>

  <hr>

  ${Object.entries(frame.angles)
    .map(([k,v]) =>
      `<div>${k}: ${v.toFixed(1)}°</div>`
    )
    .join("")}

  <br><br>

  <button id="prevFrame">Previous</button>
  <button id="nextFrame">Next</button>
`;
      document.getElementById("prevFrame").onclick = () => {
        if (index > 0) {
          index--;
          showFrame(index);
        }
      };

      document.getElementById("nextFrame").onclick = () => {
        if (index < results.length - 1) {
          index++;
          showFrame(index);
        }
      };
    };

    img.src = frame.image;
  }

  closeBtn.onclick = () => {
    modal.classList.add("hidden");
  };

  showFrame(index);
}

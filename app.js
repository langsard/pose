let detector;
const kpNames = [
  "nose","left_eye","right_eye","left_ear","right_ear",
  "left_shoulder","right_shoulder","left_elbow","right_elbow",
  "left_wrist","right_wrist","left_hip","right_hip",
  "left_knee","right_knee","left_ankle","right_ankle"
];

// -------------------------------
// Load MoveNet Thunder
// -------------------------------
async function loadModel() {
  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );
  console.log("MoveNet Loaded.");
}
loadModel();

// -------------------------------
// Image to padded canvas
// -------------------------------
function drawPadded(img, canvas) {
  const ctx = canvas.getContext("2d");

  const maxSize = 512;
  const ratio = Math.min(maxSize / img.width, maxSize / img.height);
  const w = img.width * ratio;
  const h = img.height * ratio;

  canvas.width = maxSize;
  canvas.height = maxSize;

  ctx.fillStyle = "black";
  ctx.fillRect(0,0,maxSize,maxSize);

  ctx.drawImage(img, (maxSize-w)/2, (maxSize-h)/2, w, h);

  return {offsetX: (maxSize-w)/2, offsetY: (maxSize-h)/2, scale: ratio};
}

// -------------------------------
// Draw skeleton ON padded canvas
// -------------------------------
function drawSkeleton(canvas, keypoints) {
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 3;

  const edges = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"]
  ];

  function kp(name) {
    return keypoints[kpNames.indexOf(name)];
  }

  edges.forEach(([a,b]) => {
    const p1 = kp(a);
    const p2 = kp(b);
    if (p1.score>0.3 && p2.score>0.3) {
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  });
}

// -------------------------------
// Main run function
// -------------------------------
document.getElementById("runBtn").onclick = async () => {
  if (!detector) {
    alert("Model not loaded yet.");
    return;
  }

  const frontFile = document.getElementById("frontInput").files[0];
  const sideFile = document.getElementById("sideInput").files[0];
  if (!frontFile || !sideFile) {
    alert("Please upload both images.");
    return;
  }

  const frontImg = new Image();
  const sideImg = new Image();

  frontImg.src = URL.createObjectURL(frontFile);
  sideImg.src = URL.createObjectURL(sideFile);

  frontImg.onload = async () => {
    sideImg.onload = async () => {
      // Draw padded
      const frontCanvas = document.getElementById("frontCanvas");
      const sideCanvas = document.getElementById("sideCanvas");

      const fp = drawPadded(frontImg, frontCanvas);
      const sp = drawPadded(sideImg, sideCanvas);

      // Detect
      const frontPose = await detector.estimatePoses(frontCanvas);
      const sidePose = await detector.estimatePoses(sideCanvas);

      const frontKP = frontPose[0].keypoints;
      const sideKP = sidePose[0].keypoints;

      // Draw skeleton ON padded
      drawSkeleton(frontCanvas, frontKP);
      drawSkeleton(sideCanvas, sideKP);

      // Print coordinates
      let txt = "FRONT VIEW\n";
      frontKP.forEach((kp,i)=>{
        txt += `${kpNames[i]}: (${kp.x.toFixed(1)}, ${kp.y.toFixed(1)})  score=${kp.score.toFixed(2)}\n`;
      });

      txt += "\nSIDE VIEW\n";
      sideKP.forEach((kp,i)=>{
        txt += `${kpNames[i]}: (${kp.x.toFixed(1)}, ${kp.y.toFixed(1)})  score=${kp.score.toFixed(2)}\n`;
      });

      document.getElementById("kpOutput").textContent = txt;
    };
  };
};

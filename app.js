let detector = null;
let inputImage = null;
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");

// -----------------------------
//  LOAD MOVENET
// -----------------------------
async function loadModel() {
    try {
        if (typeof poseDetection === "undefined") {
            throw new Error("poseDetection library not loaded");
        }

        console.log("Loading MoveNet...");

        detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            {
                modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
                runtime: "tfjs"
            }
        );

        document.getElementById("modelStatus").innerText = "Model Loaded ✓";
        console.log("MoveNet Loaded Successfully");
    } catch (err) {
        console.error("Model load failed:", err);
        document.getElementById("modelStatus").innerText =
            "Model Load Failed: " + err.message;
    }
}

window.onload = () => {
    loadModel();
};


// -----------------------------
//  IMAGE INPUT
// -----------------------------
document.getElementById("fileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    const img = document.getElementById("inputImg");
    img.src = url;

    img.onload = () => {
        inputImage = img;
        canvas.width = img.width;
        canvas.height = img.height;
    };
});


// -----------------------------
//  ANALYZE BUTTON
// -----------------------------
document.getElementById("analyzeBtn").addEventListener("click", async () => {
    if (!detector) {
        alert("Model not loaded yet!");
        return;
    }
    if (!inputImage) {
        alert("Upload an image first!");
        return;
    }

    const poses = await detector.estimatePoses(inputImage);
    drawPose(poses);
    summarizePosture(poses);
});


// -----------------------------
//  DRAW POSE LANDMARKS
// -----------------------------
function drawPose(poses) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!poses || poses.length === 0) return;

    const keypoints = poses[0].keypoints;

    ctx.fillStyle = "red";
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 3;

    keypoints.forEach((kp) => {
        if (kp.score > 0.3) {
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}


// -----------------------------
//  BASIC ERGONOMIC FEEDBACK
// -----------------------------
function summarizePosture(poses) {
    if (!poses || poses.length === 0) {
        document.getElementById("resultText").innerText =
            "No person detected.";
        return;
    }

    const kp = poses[0].keypoints;

    const leftShoulder = kp.find(p => p.name === "left_shoulder");
    const rightShoulder = kp.find(p => p.name === "right_shoulder");

    let message = "Posture Check: ";

    if (leftShoulder && rightShoulder) {
        const diff = Math.abs(leftShoulder.y - rightShoulder.y);

        if (diff < 20) {
            message += "Shoulders look level.";
        } else {
            message += "Your shoulders appear tilted. Adjust sitting height.";
        }
    }

    document.getElementById("resultText").innerText = message;
}

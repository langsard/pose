// HTML elements
const frontInput = document.getElementById("frontInput");
const sideInput  = document.getElementById("sideInput");
const runBtn     = document.getElementById("runBtn");
const statusBox  = document.getElementById("status");

const frontCanvas = document.getElementById("frontCanvas");
const sideCanvas  = document.getElementById("sideCanvas");

let detector = null;

// -------------------------------
// Status update helper
// -------------------------------
function setStatus(msg) {
    statusBox.textContent = msg;
}

// -------------------------------
// Load MoveNet Thunder
// -------------------------------
async function loadModel() {
    setStatus("Loading MoveNet Thunder...");
    
    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.THUNDER }
    );

    setStatus("Model loaded ✔");
    runBtn.disabled = false;
}
loadModel();

// -------------------------------
// Image → padded canvas
// -------------------------------
function drawPaddedImage(img, canvas) {
    return new Promise(resolve => {
        const ctx = canvas.getContext("2d");

        // Clear
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Compute padding
        let scale = Math.min(canvas.width / img.width, canvas.height / img.height);
        let newW = img.width * scale;
        let newH = img.height * scale;

        let dx = (canvas.width - newW) / 2;
        let dy = (canvas.height - newH) / 2;

        ctx.drawImage(img, dx, dy, newW, newH);

        resolve({ dx, dy, scale });
    });
}

// -------------------------------
// Draw skeleton on SAME canvas (correct placement)
// -------------------------------
function drawSkeleton(canvas, keypoints) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "red";
    ctx.strokeStyle = "cyan";
    ctx.lineWidth = 3;

    // Keypoint list
    const kp = keypoints;

    // Draw keypoints
    kp.forEach(pt => {
        if (pt.score > 0.3) {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    // MoveNet edges
    const edges = [
        [0,1],[1,3],[0,2],[2,4],
        [5,7],[7,9],[6,8],[8,10],
        [5,6],[5,11],[6,12],
        [11,13],[13,15],[12,14],[14,16]
    ];

    edges.forEach(([a,b]) => {
        if (kp[a].score > 0.3 && kp[b].score > 0.3) {
            ctx.beginPath();
            ctx.moveTo(kp[a].x, kp[a].y);
            ctx.lineTo(kp[b].x, kp[b].y);
            ctx.stroke();
        }
    });
}

// -------------------------------
// Run MoveNet on ONE canvas
// -------------------------------
async function processView(file, canvas) {
    if (!file) return null;

    const img = new Image();
    img.src = URL.createObjectURL(file);

    await img.decode();

    let {dx, dy, scale} = await drawPaddedImage(img, canvas);

    const poses = await detector.estimatePoses(canvas);
    if (poses.length === 0) return null;

    let raw = poses[0].keypoints;

    // Convert normalized keypoints back to padded canvas coordinates
    let mapped = raw.map(pt => ({
        x: pt.x * scale + dx,
        y: pt.y * scale + dy,
        score: pt.score
    }));

    drawSkeleton(canvas, mapped);
    return mapped;
}

// -------------------------------
// Angle helper
// -------------------------------
function angle(a, b, c) {
    const ab = {x: a.x - b.x, y: a.y - b.y};
    const cb = {x: c.x - b.x, y: c.y - b.y};
    const dot = ab.x * cb.x + ab.y * cb.y;
    const m1 = Math.sqrt(ab.x**2 + ab.y**2);
    const m2 = Math.sqrt(cb.x**2 + cb.y**2);
    return Math.acos(dot / (m1*m2)) * 180 / Math.PI;
}

// -------------------------------
// Fill table
// -------------------------------
function fillTable(frontKP, sideKP) {
    const table = document.getElementById("results-table");
    table.innerHTML = "";

    const row = (name, f, s) =>
        `<tr><td>${name}</td><td>${f}</td><td>${s}</td></tr>`;

    table.innerHTML += `<tr><th>Point</th><th>Front</th><th>Side</th></tr>`;

    for (let i = 0; i < 17; i++) {
        let f = frontKP ? `${frontKP[i].x.toFixed(4)}, ${frontKP[i].y.toFixed(4)}` : "-";
        let s = sideKP  ? `${sideKP[i].x.toFixed(4)}, ${sideKP[i].y.toFixed(4)}` : "-";
        table.innerHTML += row(`KP ${i}`, f, s);
    }
}

// -------------------------------
// Run Button
// -------------------------------
runBtn.onclick = async () => {
    setStatus("Running detection…");

    const frontKP = await processView(frontInput.files[0], frontCanvas);
    const sideKP  = await processView(sideInput.files[0],  sideCanvas);

    if (!frontKP && !sideKP) {
        setStatus("No person detected.");
        return;
    }

    fillTable(frontKP, sideKP);

    setStatus("Done ✔");
};

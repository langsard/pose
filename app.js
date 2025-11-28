let detector;

// LOAD MOVENET
async function loadModel() {
    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );
    console.log("MoveNet loaded.");
}
loadModel();


// IMAGE → PADDED CANVAS
function drawPadded(img, canvas) {
    return new Promise(resolve => {
        const ctx = canvas.getContext("2d");

        const target = 350;  
        let scale = Math.min(target / img.width, target / img.height);
        let newW = img.width * scale;
        let newH = img.height * scale;

        canvas.width = target;
        canvas.height = target;
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, target, target);

        let x = (target - newW) / 2;
        let y = (target - newH) / 2;
        ctx.drawImage(img, x, y, newW, newH);

        resolve({ scale, offsetX: x, offsetY: y });
    });
}


// DRAW SKELETON
function drawSkeleton(keypoints, canvas) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "red";
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 3;

    // Keypoints
    keypoints.forEach(kp => {
        if (kp.score > 0.3) {
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    // Connections
    const edges = poseDetection.util.getAdjacentPairs(
        poseDetection.SupportedModels.MoveNet
    );

    edges.forEach(([a, b]) => {
        const kp1 = keypoints[a];
        const kp2 = keypoints[b];
        if (kp1.score > 0.3 && kp2.score > 0.3) {
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.stroke();
        }
    });
}


// NORMALIZE BACK TO PADDED COORDS
function transformKeypoints(kps, scale, x0, y0) {
    return kps.map(kp => ({
        ...kp,
        x: kp.x * scale + x0,
        y: kp.y * scale + y0
    }));
}


// SIMPLE ANGLE BETWEEN 3 POINTS
function angle(a, b, c) {
    const ab = { x: a.x - b.x, y: a.y - b.y };
    const cb = { x: c.x - b.x, y: c.y - b.y };

    let dot = ab.x * cb.x + ab.y * cb.y;
    let mag1 = Math.sqrt(ab.x**2 + ab.y**2);
    let mag2 = Math.sqrt(cb.x**2 + cb.y**2);

    return Math.acos(dot / (mag1 * mag2)) * 180 / Math.PI;
}


// BEST VIEW SELECTION
function pickBetter(front, side) {
    const f = front.reduce((a, b) => a + b.score, 0);
    const s = side.reduce((a, b) => a + b.score, 0);
    return f >= s ? "front" : "side";
}


// PROCESS IMAGE
async function process(imgFile, canvas, outputPre) {
    if (!imgFile) return null;

    const img = new Image();
    img.src = URL.createObjectURL(imgFile);

    await new Promise(r => img.onload = r);

    const pad = await drawPadded(img, canvas);

    const input = tf.browser.fromPixels(canvas);
    const poses = await detector.estimatePoses(input);
    input.dispose();

    if (poses.length === 0) return null;

    let kps = poses[0].keypoints;
    kps = transformKeypoints(kps, 1, 0, 0);

    drawSkeleton(kps, canvas);

    outputPre.textContent = JSON.stringify(kps, null, 2);

    return kps;
}


// RUN BUTTON
document.getElementById("runBtn").addEventListener("click", async () => {
    const frontFile = document.getElementById("frontInput").files[0];
    const sideFile = document.getElementById("sideInput").files[0];

    const frontKP = await process(
        frontFile,
        document.getElementById("frontCanvas"),
        document.getElementById("frontKeypoints")
    );

    const sideKP = await process(
        sideFile,
        document.getElementById("sideCanvas"),
        document.getElementById("sideKeypoints")
    );

    console.log("Front:", frontKP);
    console.log("Side:", sideKP);

    if (frontKP && sideKP) {
        console.log("Best:", pickBetter(frontKP, sideKP));
    }
});

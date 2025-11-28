let detector;

// =========================================================
// LOAD MOVENET
// =========================================================
async function loadModel() {
    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );
    console.log("MoveNet loaded.");
}
loadModel();


// =========================================================
// UTILS
// =========================================================

// Draw padded image inside square 350×350
function drawPadded(img, canvas) {
    return new Promise(resolve => {
        const target = 350;
        const ctx = canvas.getContext("2d");

        let scale = Math.min(target / img.width, target / img.height);
        let w = img.width * scale;
        let h = img.height * scale;

        let x = (target - w) / 2;
        let y = (target - h) / 2;

        canvas.width = target;
        canvas.height = target;
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, target, target);
        ctx.drawImage(img, x, y, w, h);

        resolve({ scale, offsetX: x, offsetY: y });
    });
}

// Transform MoveNet normalized coords to canvas coords
function transformKeypoints(kps, pad) {
    return kps.map(kp => ({
        name: kp.name,
        score: kp.score,
        x: kp.x * pad.scale + pad.offsetX,
        y: kp.y * pad.scale + pad.offsetY
    }));
}


// Draw skeleton on canvas
function drawSkeleton(kps, canvas) {
    const ctx = canvas.getContext("2d");
    ctx.strokeStyle = "lime";
    ctx.fillStyle = "red";
    ctx.lineWidth = 3;

    const edges = poseDetection.util.getAdjacentPairs(
        poseDetection.SupportedModels.MoveNet
    );

    // dots
    kps.forEach(p => {
        if (p.score > 0.3) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
            ctx.fill();
        }
    });

    // edges
    edges.forEach(([a, b]) => {
        const p1 = kps[a];
        const p2 = kps[b];
        if (p1.score > 0.3 && p2.score > 0.3) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }
    });
}


// angle between 3 points
function angle(A, B, C) {
    const AB = { x: A.x - B.x, y: A.y - B.y };
    const CB = { x: C.x - B.x, y: C.y - B.y };
    let dot = AB.x * CB.x + AB.y * CB.y;
    let mag = Math.sqrt(AB.x**2 + AB.y**2) * Math.sqrt(CB.x**2 + CB.y**2);
    return Math.acos(dot / mag) * 180 / Math.PI;
}


// normalized pose (scale + centered)
function normalize(kps) {
    const xs = kps.map(p => p.x);
    const ys = kps.map(p => p.y);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;

    const scale = Math.max(
        Math.max(...xs) - Math.min(...xs),
        Math.max(...ys) - Math.min(...ys)
    );

    return kps.map(p => ({
        ...p,
        nx: (p.x - cx) / scale,
        ny: (p.y - cy) / scale
    }));
}


// segment length
function dist(a, b) {
    return Math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2);
}


// =========================================================
// PROCESS ONE IMAGE
// =========================================================
async function process(imgFile, canvas, keypointBox) {
    if (!imgFile) return null;

    const img = new Image();
    img.src = URL.createObjectURL(imgFile);
    await new Promise(r => img.onload = r);

    // padded
    const pad = await drawPadded(img, canvas);

    // detect
    const input = tf.browser.fromPixels(canvas);
    const poses = await detector.estimatePoses(input);
    input.dispose();

    if (poses.length === 0) return null;

    let kps = poses[0].keypoints;
    kps = transformKeypoints(kps, pad);
    drawSkeleton(kps, canvas);

    keypointBox.textContent =
        "KEYPOINTS:\n" + JSON.stringify(kps, null, 2);

    return kps;
}


// =========================================================
// COMBINE FRONT + SIDE
// =========================================================
function pickBestByJoint(front, side) {
    const merged = [];
    for (let i = 0; i < front.length; i++) {
        merged.push(
            front[i].score >= side[i].score ? front[i] : side[i]
        );
    }
    return merged;
}


// joint angles
function computeAngles(kps) {
    const idx = name => kps.findIndex(p => p.name === name);

    function safe(a, b, c) {
        if (!a || !b || !c) return null;
        if (a.score < 0.3 || b.score < 0.3 || c.score < 0.3) return null;
        return angle(a, b, c).toFixed(1);
    }

    return {
        leftElbow: safe(
            kps[idx("left_shoulder")],
            kps[idx("left_elbow")],
            kps[idx("left_wrist")]
        ),
        rightElbow: safe(
            kps[idx("right_shoulder")],
            kps[idx("right_elbow")],
            kps[idx("right_wrist")]
        ),
        leftKnee: safe(
            kps[idx("left_hip")],
            kps[idx("left_knee")],
            kps[idx("left_ankle")]
        ),
        rightKnee: safe(
            kps[idx("right_hip")],
            kps[idx("right_knee")],
            kps[idx("right_ankle")]
        ),
        pelvis: safe(
            kps[idx("left_shoulder")],
            kps[idx("left_hip")],
            kps[idx("right_hip")]
        )
    };
}


// body proportion ratios
function proportions(kps) {
    const idx = name => kps.findIndex(p => p.name === name);

    let lArm = dist(kps[idx("left_shoulder")], kps[idx("left_wrist")]);
    let rArm = dist(kps[idx("right_shoulder")], kps[idx("right_wrist")]);
    let arm = ((lArm + rArm) / 2).toFixed(1);

    let lLeg = dist(kps[idx("left_hip")], kps[idx("left_ankle")]);
    let rLeg = dist(kps[idx("right_hip")], kps[idx("right_ankle")]);
    let leg = ((lLeg + rLeg) / 2).toFixed(1);

    let torso = dist(kps[idx("left_shoulder")], kps[idx("left_hip")]).toFixed(1);

    return { arm, leg, torso };
}


// =========================================================
// MAIN BUTTON HANDLER
// =========================================================
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

    if (!frontKP || !sideKP) {
        document.getElementById("report").textContent =
            "❌ Missing one of the views.";
        return;
    }

    // per-joint best view
    const merged = pickBestByJoint(frontKP, sideKP);

    // angles
    const ang = computeAngles(merged);

    // proportions
    const prop = proportions(merged);

    // normalized pose
    const norm = normalize(merged);

    // output report
    document.getElementById("report").textContent =
        "=== BEST-VIEW MERGED POSE ===\n" +
        JSON.stringify(merged, null, 2) +
        "\n\n=== ANGLES ===\n" +
        JSON.stringify(ang, null, 2) +
        "\n\n=== BODY PROPORTIONS ===\n" +
        JSON.stringify(prop, null, 2) +
        "\n\n=== NORMALIZED POSE (0–1) ===\n" +
        JSON.stringify(norm, null, 2);

});

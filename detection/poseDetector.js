let detector;

export async function initDetector() {

  await tf.setBackend("webgl");

  const model = poseDetection.SupportedModels.MoveNet;

  detector = await poseDetection.createDetector(model, {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER
  });
}

export async function detectPose(source) {

  const poses = await detector.estimatePoses(source);

  if (!poses || poses.length === 0) return [];

  return poses[0].keypoints;
}

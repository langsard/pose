function calculateAngle(a, b, c) {

  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dot = ab.x * cb.x + ab.y * cb.y;

  const magAB = Math.sqrt(ab.x**2 + ab.y**2);
  const magCB = Math.sqrt(cb.x**2 + cb.y**2);

  return Math.acos(dot / (magAB * magCB)) * (180 / Math.PI);
}

export function calculateAngles(keypoints) {

  const leftShoulder = keypoints[5];
  const leftElbow = keypoints[7];
  const leftWrist = keypoints[9];

  const rightShoulder = keypoints[6];
  const rightElbow = keypoints[8];
  const rightWrist = keypoints[10];

  return {
    leftElbowAngle: calculateAngle(leftShoulder, leftElbow, leftWrist),
    rightElbowAngle: calculateAngle(rightShoulder, rightElbow, rightWrist)
  };
}

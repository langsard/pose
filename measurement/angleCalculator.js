function angle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.hypot(ab.x, ab.y);
  const magCB = Math.hypot(cb.x, cb.y);

  return Math.acos(dot / (magAB * magCB)) * (180 / Math.PI);
}

export function calculateAngles(kp) {

  const midShoulder = {
    x: (kp[5].x + kp[6].x) / 2,
    y: (kp[5].y + kp[6].y) / 2
  };

  const midHip = {
    x: (kp[11].x + kp[12].x) / 2,
    y: (kp[11].y + kp[12].y) / 2
  };

  return {

    leftElbow: angle(kp[5], kp[7], kp[9]),
    rightElbow: angle(kp[6], kp[8], kp[10]),

    leftShoulder: angle(kp[7], kp[5], kp[11]),
    rightShoulder: angle(kp[8], kp[6], kp[12]),

    leftHip: angle(kp[5], kp[11], kp[13]),
    rightHip: angle(kp[6], kp[12], kp[14]),

    leftKnee: angle(kp[11], kp[13], kp[15]),
    rightKnee: angle(kp[12], kp[14], kp[16]),

    trunk: angle(midShoulder, midHip, kp[0]),
    neck: angle(kp[5], kp[0], kp[6])
  };
}

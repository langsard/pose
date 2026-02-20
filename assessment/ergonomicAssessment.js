export function assessPosture(angles) {

  return {
    leftElbow: angles.leftElbowAngle > 120 ? "Low Risk" : "High Risk",
    rightElbow: angles.rightElbowAngle > 120 ? "Low Risk" : "High Risk"
  };
}

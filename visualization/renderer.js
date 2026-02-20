export function renderPreview(source, keypoints) {

  const canvas = document.getElementById("outputCanvas");
  const ctx = canvas.getContext("2d");

  canvas.width = source.width;
  canvas.height = source.height;

  ctx.drawImage(source, 0, 0);

  keypoints.forEach(kp => {
    if (kp.score > 0.4) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  });
}

export function renderResultModal(source, keypoints, angles, assessment) {

  const modal = document.getElementById("resultModal");
  const canvas = document.getElementById("resultCanvas");
  const ctx = canvas.getContext("2d");
  const table = document.getElementById("angleTable");
  const closeBtn = document.getElementById("closeResult");

  modal.classList.remove("hidden");

  canvas.width = source.width;
  canvas.height = source.height;

  ctx.drawImage(source, 0, 0);

  keypoints.forEach(kp => {
    if (kp.score > 0.4) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "blue";
      ctx.fill();
    }
  });

  table.innerHTML = `
    <h3>Joint Angles</h3>
    <ul>
      <li>Left Elbow: ${angles.leftElbowAngle.toFixed(1)}° (${assessment.leftElbow})</li>
      <li>Right Elbow: ${angles.rightElbowAngle.toFixed(1)}° (${assessment.rightElbow})</li>
    </ul>
  `;

  closeBtn.onclick = () => {
    modal.classList.add("hidden");
  };
}

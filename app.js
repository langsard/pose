const frontInput = document.getElementById('frontInput');
const sideInput = document.getElementById('sideInput');
const frontPreview = document.getElementById('frontPreview');
const sidePreview = document.getElementById('sidePreview');
const runBtn = document.getElementById('runBtn');
const outputSection = document.getElementById('output-section');

const frontCanvas = document.getElementById('frontCanvas');
const sideCanvas = document.getElementById('sideCanvas');
const tableBody = document.querySelector('#resultTable tbody');

// Preview Handlers
frontInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) frontPreview.src = URL.createObjectURL(file);
});

sideInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) sidePreview.src = URL.createObjectURL(file);
});

runBtn.addEventListener('click', () => {
  outputSection.style.display = 'block';

  // Example: fill table with dummy integer data
  const keypoints = ["Head", "Shoulder", "Elbow", "Hip", "Knee", "Ankle"];

  tableBody.innerHTML = "";
  keypoints.forEach(kp => {
    const row = document.createElement('tr');

    const fX = Math.floor(Math.random() * 300);
    const fY = Math.floor(Math.random() * 300);
    const sX = Math.floor(Math.random() * 300);
    const sY = Math.floor(Math.random() * 300);

    row.innerHTML = `
      <td>${kp}</td>
      <td>${fX}, ${fY}</td>
      <td>${sX}, ${sY}</td>
    `;
    tableBody.appendChild(row);
  });

  drawPlaceholder(frontCanvas);
  drawPlaceholder(sideCanvas);
});

function drawPlaceholder(canvas) {
  const ctx = canvas.getContext('2d');
  canvas.width = 320;
  canvas.height = 260;

  ctx.fillStyle = '#f0f0f0';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = '#000';
  ctx.beginPath();
  ctx.moveTo(50, 50);
  ctx.lineTo(270, 210);
  ctx.stroke();
}

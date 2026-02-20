export function handleInput(previewCallback, runCallback) {

  const fileInput = document.getElementById("fileInput");
  const chooseBtn = document.getElementById("chooseBtn");
  const runBtn = document.getElementById("runBtn");

  const modal = document.getElementById("intervalModal");
  const confirmBtn = document.getElementById("confirmInterval");
  const intervalInput = document.getElementById("frameIntervalInput");

  let selectedFile = null;
  let previewMedia = null;

  chooseBtn.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", (event) => {

    selectedFile = event.target.files[0];
    if (!selectedFile) return;

    if (selectedFile.type.startsWith("image")) {

      const img = new Image();
      img.src = URL.createObjectURL(selectedFile);

      img.onload = () => {
        previewMedia = { type: "image", data: img };
        previewCallback(previewMedia);
      };
    }

    if (selectedFile.type.startsWith("video")) {

      const video = document.createElement("video");
      video.src = URL.createObjectURL(selectedFile);

      video.onloadeddata = () => {
        previewMedia = { type: "video", data: video };
        previewCallback(previewMedia);
      };
    }
  });

  runBtn.addEventListener("click", () => {

    if (!selectedFile) return;

    if (selectedFile.type.startsWith("image")) {
      runCallback(previewMedia);
    }

    if (selectedFile.type.startsWith("video")) {

      modal.classList.remove("hidden");

      confirmBtn.onclick = () => {
        modal.classList.add("hidden");
        const fps = parseInt(intervalInput.value);
        runCallback({ ...previewMedia, fps });
      };
    }
  });
}

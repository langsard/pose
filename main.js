import { handleInput } from "./input/inputHandler.js";
import { processVideo } from "./input/videoProcessor.js";
import { initDetector, detectPose } from "./detection/poseDetector.js";
import { calculateAngles } from "./measurement/angleCalculator.js";
import { renderPreview, renderResultModal, renderGallery } from "./visualization/renderer.js";

await initDetector();

// Load default preview image
async function loadDefaultImage() {

  const canvas = document.getElementById("outputCanvas");
  const defaultPath = canvas.dataset.default;

  if (!defaultPath) return;

  const img = new Image();
  img.src = defaultPath;

  img.onload = async () => {

    const keypoints = await detectPose(img);
    renderPreview(img, keypoints);
  };
}

async function previewPipeline(media) {

  let source = media.data;

  if (media.type === "video") {
    const canvas = document.createElement("canvas");
    canvas.width = source.videoWidth;
    canvas.height = source.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(source, 0, 0);
    source = canvas;
  }

  const keypoints = await detectPose(source);
  renderPreview(source, keypoints);
}

async function runPipeline(media) {

  if (media.type === "image") {

    const keypoints = await detectPose(media.data);
    const angles = calculateAngles(keypoints);

    renderResultModal(media.data, keypoints, angles);
  }

  if (media.type === "video") {

    const galleryResults = [];

    await processVideo(media.data, media.fps, async (frameCanvas) => {

      const keypoints = await detectPose(frameCanvas);
      const angles = calculateAngles(keypoints);

      galleryResults.push({
        image: frameCanvas.toDataURL(),
        keypoints,
        angles
      });
    });

    renderGallery(galleryResults);
  }
}

handleInput(previewPipeline, runPipeline);
loadDefaultImage();

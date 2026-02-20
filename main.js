import { handleInput } from "./input/inputHandler.js";
import { processVideo } from "./input/videoProcessor.js";
import { initDetector, detectPose } from "./detection/poseDetector.js";
import { calculateAngles } from "./measurement/angleCalculator.js";
import { assessPosture } from "./assessment/ergonomicAssessment.js";
import { renderPreview, renderResultModal } from "./visualization/renderer.js";

await initDetector();

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
    const assessment = assessPosture(angles);

    renderResultModal(media.data, keypoints, angles, assessment);
  }

  if (media.type === "video") {

    await processVideo(media.data, media.fps, async (frameCanvas) => {

      const keypoints = await detectPose(frameCanvas);
      const angles = calculateAngles(keypoints);
      const assessment = assessPosture(angles);

      renderResultModal(frameCanvas, keypoints, angles, assessment);
    });
  }
}

handleInput(previewPipeline, runPipeline);

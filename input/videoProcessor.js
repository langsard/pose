export async function processVideo(video, interval, callback) {

  const duration = video.duration;

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");

  for (let time = 0; time < duration; time += interval) {

    video.currentTime = time;
    await new Promise(resolve => video.onseeked = resolve);

    ctx.drawImage(video, 0, 0);

    await callback(canvas);
  }
}

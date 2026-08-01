"""
Media I/O helpers — data URLs, images, audio, and video for multimodal prompts.
"""
import base64
import mimetypes
import os
import tempfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from ray_hive.errors import MediaError


def load_bytes_from_url(url: str) -> bytes:
    """Fetch bytes from an http(s) URL or ``data:`` URL."""
    if url.startswith("data:"):
        # data:[<mediatype>][;base64],<data>
        _, _, payload = url.partition(",")
        if ";base64" in url[: url.find(",")]:
            return base64.b64decode(payload)
        return payload.encode("utf-8")
    with urlopen(url, timeout=60) as resp:
        return resp.read()


def file_to_data_url(path: str | Path, mime: str | None = None) -> str:
    """Encode a local file as a ``data:<mime>;base64,...`` URL."""
    path = Path(path)
    if mime is None:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def pil_from_url(url: str):
    """Load an image URL/data-URL as an RGB PIL Image."""
    from PIL import Image

    return Image.open(BytesIO(load_bytes_from_url(url))).convert("RGB")


def audio_array_from_bytes(data: bytes, target_sr: int = 16000):
    """
    Decode PCM wav bytes → ``(float32 mono array, sample_rate)`` for vLLM.

    Resamples to ``target_sr`` (default 16 kHz) — required by Whisper-family
    encoders used by Qwen2-Audio and similar models.
    """
    import wave

    import numpy as np

    with wave.open(BytesIO(data), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    arr = np.asarray(arr, dtype=np.float32)
    if sr != target_sr and len(arr) > 0:
        new_len = max(1, int(round(len(arr) * target_sr / sr)))
        arr = np.interp(
            np.linspace(0, len(arr), new_len, endpoint=False),
            np.arange(len(arr), dtype=np.float64),
            arr,
        ).astype(np.float32)
        sr = target_sr
    return arr, int(sr)


def audio_from_url(url: str):
    """Load audio from URL/data-URL as ``(array, sample_rate)``."""
    return audio_array_from_bytes(load_bytes_from_url(url))


def audio_from_b64(data: str):
    """Load audio from raw base64 payload as ``(array, sample_rate)``."""
    return audio_array_from_bytes(base64.b64decode(data))


def video_frames_from_url(url: str):
    """
    Decode video URL/data-URL for vLLM multimodal input.

    Returns ``(frames, metadata)`` where frames is RGB uint8 ``(T, H, W, C)``
    and metadata has ``fps``, ``duration``, ``total_num_frames``, ``frames_indices``
    (required by Gemma 4 / models with ``video_needs_metadata``).
    """
    import numpy as np

    data = load_bytes_from_url(url)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    try:
        os.write(fd, data)
        os.close(fd)
        frames = []
        fps = 2.0
        try:
            import cv2

            cap = cv2.VideoCapture(path)
            raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            if raw_fps > 0:
                fps = raw_fps
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        except ImportError:
            import imageio.v3 as iio

            meta = iio.immeta(path, exclude_applied=False) or {}
            if meta.get("fps"):
                fps = float(meta["fps"])
            arr = iio.imread(path)
            if arr.ndim == 3:
                frames = [arr]
            else:
                frames = list(arr)
        if not frames:
            raise MediaError("could not decode video")
        stacked = np.stack(frames)
        n = int(stacked.shape[0])
        metadata = {
            "fps": float(fps),
            "total_num_frames": n,
            "duration": float(n / fps) if fps > 0 else float(n),
            "frames_indices": list(range(n)),
        }
        return stacked, metadata
    finally:
        if os.path.exists(path):
            os.unlink(path)

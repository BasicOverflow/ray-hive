"""Download one image/audio fixture from HF; build one short mp4 from the image."""
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parent

IMAGE_DS = "AI-Lab-Makerere/beans"
AUDIO_DS = "PolyAI/minds14"
AUDIO_CONFIG = "en-US"


def _clear_old():
    for pattern in ("sample.*", "image_*.png", "image_*.jpg", "audio_*.wav", "video_*.mp4"):
        for p in ROOT.glob(pattern):
            p.unlink()


def fetch_image():
    ds = load_dataset(IMAGE_DS, split="train", streaming=True)
    row = next(iter(ds))
    img = row["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    img.convert("RGB").save(ROOT / "image_00.png")
    print(f"wrote image_00.png from {IMAGE_DS}")


def fetch_audio():
    import soundfile as sf

    ds = load_dataset(AUDIO_DS, AUDIO_CONFIG, split="train", streaming=True)
    # Qwen2-Audio / Whisper-family encoders expect 16 kHz.
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    row = next(iter(ds))
    audio = row["audio"]
    sf.write(
        ROOT / "audio_00.wav",
        audio["array"],
        int(audio["sampling_rate"]),
        subtype="PCM_16",
    )
    print(f"wrote audio_00.wav @16kHz from {AUDIO_DS}:{AUDIO_CONFIG}")


def build_video():
    """Short mp4 from image_00 (real decodable video)."""
    import imageio.v3 as iio

    img_path = ROOT / "image_00.png"
    frame = np.asarray(Image.open(img_path).convert("RGB").resize((256, 256)))
    frames = [frame] * 8
    iio.imwrite(ROOT / "video_00.mp4", frames, fps=2, codec="libx264")
    print("wrote video_00.mp4 from image_00.png")


if __name__ == "__main__":
    _clear_old()
    fetch_image()
    fetch_audio()
    build_video()
    print("done:", sorted(
        p.name for p in ROOT.iterdir() if p.suffix in {".png", ".wav", ".mp4"}
    ))

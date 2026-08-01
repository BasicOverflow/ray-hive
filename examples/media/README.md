# Example media fixtures

One image, one audio clip, one short mp4 for the multimodal examples.

| File | Source |
|------|--------|
| `image_00.png` | `AI-Lab-Makerere/beans` |
| `audio_00.wav` | `PolyAI/minds14` (`en-US`, 16 kHz PCM) |
| `video_00.mp4` | built from `image_00.png` |

Regenerate:

```bash
pip install -r examples/requirements.txt
python examples/media/generate_fixtures.py
```

Respect upstream dataset licenses when redistributing. Fixtures are for local demos only.

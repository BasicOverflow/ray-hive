"""H — media helpers + MM factory."""
import base64

from ray_hive.core.model_specs.factory import is_multimodal_hf, select_vram_classes
from ray_hive.core.model_specs.mm_attention import MultimodalAttentionSpecs
from ray_hive.core.model_specs.mm_vram_reqs import MultimodalVramReqs
from ray_hive.core.model_specs.planner import build_vram_reqs, effective_input_len
from ray_hive.core.ray_utils.media import file_to_data_url, load_bytes_from_url, pil_from_url
from ray_hive.core.ray_utils import mm_helpers


def test_file_to_data_url_and_pil(tmp_path, tiny_png_bytes):
    p = tmp_path / "x.png"
    p.write_bytes(tiny_png_bytes)
    url = file_to_data_url(p)
    assert url.startswith("data:image/png;base64,")
    img = pil_from_url(url)
    assert img.size == (1, 1)


def test_load_data_url_raw():
    raw = b"hello"
    b64 = base64.b64encode(raw).decode()
    assert load_bytes_from_url(f"data:text/plain;base64,{b64}") == raw


def test_mm_factory(tiny_hf_mm, tiny_hf_dense):
    assert is_multimodal_hf(tiny_hf_mm)
    assert not is_multimodal_hf(tiny_hf_dense)
    attn, vram = select_vram_classes(tiny_hf_mm)
    assert attn is MultimodalAttentionSpecs
    assert vram is MultimodalVramReqs


def test_mm_effective_input_ge_text(tiny_hf_mm, tiny_hf_dense):
    mm = build_vram_reqs(tiny_hf_mm)
    text = build_vram_reqs(tiny_hf_dense)
    assert effective_input_len(mm, 64) >= effective_input_len(text, 64)


def test_mm_count_helper():
    assert mm_helpers.mm_count({"image": 2}, "image") == 2
    assert mm_helpers.mm_count(None, "image") == 0

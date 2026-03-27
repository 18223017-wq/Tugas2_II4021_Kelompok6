# tests/test_mp4.py
import os
import sys
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.stego_video_mp4 import embed_message, extract_message
from src.stego_lsb_utils import LSB_METHOD_332, LSB_METHOD_111, LSB_METHOD_444

COVER_PATH = "samples/blackpink.mp4"
PESAN_PATH = "samples/pesan.txt"
A51_KEY    = 0x123456789ABCDEF0
STEGO_KEY  = 42

@pytest.fixture(autouse=True)
def setup():
    os.makedirs("tests_output", exist_ok=True)

@pytest.fixture
def pesan():
    if not os.path.exists(PESAN_PATH):
        pytest.skip(f"pesan.txt tidak ada: {PESAN_PATH}")
    with open(PESAN_PATH, 'rb') as f:
        return f.read()

def _skip_if_no_cover():
    if not os.path.exists(COVER_PATH):
        pytest.skip(f"Cover video tidak ada: {COVER_PATH}")


def _run_roundtrip(pesan, lsb_method, use_encrypt=True, use_random=True, label=""):
    """Helper: embed → extract → verify."""
    _skip_if_no_cover()
    output_path = f"tests_output/stego_mp4_{label}.mp4"

    result = embed_message(
        cover_path=COVER_PATH, output_path=output_path,
        message=pesan, is_text=True, extension=".txt", filename="pesan.txt",
        use_encryption=use_encrypt,
        a51_key=A51_KEY if use_encrypt else None,
        use_random=use_random,
        stego_key=STEGO_KEY if use_random else None,
        mp4_crf=0, lsb_method=lsb_method
    )
    print(f"\n    [{label}] Embed OK | MSE={result['mse_avg']:.4f} PSNR={result['psnr_avg']:.2f} dB")

    out = extract_message(
        stego_path=output_path,
        a51_key=A51_KEY if use_encrypt else None,
        stego_key=STEGO_KEY if use_random else None,
    )
    extracted = out["message"]

    orig_md5 = hashlib.md5(pesan).hexdigest()
    extr_md5 = hashlib.md5(extracted).hexdigest()
    print(f"    MD5 orig={orig_md5}  extr={extr_md5}")

    assert extracted == pesan, f"Extract gagal untuk {label}"
    assert out["lsb_method_label"] is not None
    print(f"    ✓ Roundtrip {label} OK (LSB: {out['lsb_method_label']})")
    return result


# ─── 3-3-2 ────────────────────────────────────────────────────────────────────

def test_mp4_332_encrypted_random(pesan):
    _run_roundtrip(pesan, LSB_METHOD_332, use_encrypt=True, use_random=True, label="332_enc_rnd")

def test_mp4_332_plain_sequential(pesan):
    _run_roundtrip(pesan, LSB_METHOD_332, use_encrypt=False, use_random=False, label="332_plain_seq")

# ─── 1-1-1 ────────────────────────────────────────────────────────────────────

def test_mp4_111_encrypted_random(pesan):
    _run_roundtrip(pesan, LSB_METHOD_111, use_encrypt=True, use_random=True, label="111_enc_rnd")

def test_mp4_111_plain_sequential(pesan):
    _run_roundtrip(pesan, LSB_METHOD_111, use_encrypt=False, use_random=False, label="111_plain_seq")

# ─── 4-4-4 ────────────────────────────────────────────────────────────────────

def test_mp4_444_encrypted_random(pesan):
    _run_roundtrip(pesan, LSB_METHOD_444, use_encrypt=True, use_random=True, label="444_enc_rnd")

def test_mp4_444_plain_sequential(pesan):
    _run_roundtrip(pesan, LSB_METHOD_444, use_encrypt=False, use_random=False, label="444_plain_seq")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

# tests/test_stego_video.py
import os
import sys
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.stego_video import embed_message, extract_message, total_capacity_bytes
from src.stego_lsb_utils import LSB_METHOD_332, LSB_METHOD_111, LSB_METHOD_444
from src.video_io import read_video_frames

SAMPLE_VIDEO = "samples/sample_video.avi"
OUTPUT_VIDEO  = "tests_output/stego_test_output.avi"
PESAN_PATH    = "samples/pesan.txt"
A51_KEY       = 0x123456789ABCDEF0
STEGO_KEY     = 42

@pytest.fixture(autouse=True)
def setup_output_dir():
    os.makedirs("tests_output", exist_ok=True)

@pytest.fixture
def pesan():
    if not os.path.exists(PESAN_PATH):
        pytest.skip(f"pesan.txt tidak ada: {PESAN_PATH}")
    with open(PESAN_PATH, 'rb') as f:
        return f.read()

# ─── KAPASITAS ────────────────────────────────────────────────────────────────

def test_capacity_332():
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip("Sample video tidak ada")
    frames, _ = read_video_frames(SAMPLE_VIDEO)
    cap = total_capacity_bytes(frames, LSB_METHOD_332)
    assert cap > 0
    print(f"\nKapasitas video (3-3-2): {cap} bytes = {cap // 1024} KB")

def test_capacity_111():
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip("Sample video tidak ada")
    frames, _ = read_video_frames(SAMPLE_VIDEO)
    cap = total_capacity_bytes(frames, LSB_METHOD_111)
    assert cap > 0
    print(f"\nKapasitas video (1-1-1): {cap} bytes = {cap // 1024} KB")

def test_capacity_444():
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip("Sample video tidak ada")
    frames, _ = read_video_frames(SAMPLE_VIDEO)
    cap = total_capacity_bytes(frames, LSB_METHOD_444)
    assert cap > 0
    print(f"\nKapasitas video (4-4-4): {cap} bytes = {cap // 1024} KB")

def test_capacity_ordering():
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip("Sample video tidak ada")
    frames, _ = read_video_frames(SAMPLE_VIDEO)
    c111 = total_capacity_bytes(frames, LSB_METHOD_111)
    c332 = total_capacity_bytes(frames, LSB_METHOD_332)
    c444 = total_capacity_bytes(frames, LSB_METHOD_444)
    assert c111 < c332 < c444

# ═══════════════════════════════════════════════════════════════════════════════
#  3-3-2 METHOD
# ═══════════════════════════════════════════════════════════════════════════════

def test_roundtrip_332_plain_sequential(pesan):
    result = embed_message(
        cover_path=SAMPLE_VIDEO, output_path=OUTPUT_VIDEO,
        message=pesan, is_text=True, extension=".txt", filename="pesan.txt",
        use_encryption=False, use_random=False, lsb_method=LSB_METHOD_332
    )
    print(f"\n3-3-2 PSNR: {result['psnr_avg']:.2f} dB | MSE: {result['mse_avg']:.4f}")
    assert result['psnr_avg'] > 30

    out = extract_message(stego_path=OUTPUT_VIDEO)
    assert out["message"] == pesan
    assert out["is_text"] == True
    assert out["is_encrypted"] == False
    assert out["is_random"] == False
    assert out["filename"] == "pesan.txt"
    assert out["lsb_method"] == LSB_METHOD_332

def test_roundtrip_332_encrypted_random(pesan):
    embed_message(
        cover_path=SAMPLE_VIDEO, output_path=OUTPUT_VIDEO,
        message=pesan, is_text=True, extension=".txt", filename="pesan.txt",
        use_encryption=True, a51_key=A51_KEY,
        use_random=True, stego_key=STEGO_KEY, lsb_method=LSB_METHOD_332
    )
    out = extract_message(stego_path=OUTPUT_VIDEO, a51_key=A51_KEY, stego_key=STEGO_KEY)
    assert out["message"] == pesan
    assert out["is_encrypted"] == True
    assert out["is_random"] == True
    print("3-3-2 Encrypted random roundtrip OK")

# ═══════════════════════════════════════════════════════════════════════════════
#  1-1-1 METHOD
# ═══════════════════════════════════════════════════════════════════════════════

def test_roundtrip_111_plain_sequential(pesan):
    result = embed_message(
        cover_path=SAMPLE_VIDEO, output_path=OUTPUT_VIDEO,
        message=pesan, is_text=True, extension=".txt", filename="pesan.txt",
        use_encryption=False, use_random=False, lsb_method=LSB_METHOD_111
    )
    print(f"\n1-1-1 PSNR: {result['psnr_avg']:.2f} dB | MSE: {result['mse_avg']:.4f}")
    assert result['psnr_avg'] > 40  # 1-1-1 should have higher PSNR

    out = extract_message(stego_path=OUTPUT_VIDEO)
    assert out["message"] == pesan
    assert out["lsb_method"] == LSB_METHOD_111

def test_roundtrip_111_encrypted_random(pesan):
    embed_message(
        cover_path=SAMPLE_VIDEO, output_path=OUTPUT_VIDEO,
        message=pesan, is_text=True, extension=".txt", filename="pesan.txt",
        use_encryption=True, a51_key=A51_KEY,
        use_random=True, stego_key=STEGO_KEY, lsb_method=LSB_METHOD_111
    )
    out = extract_message(stego_path=OUTPUT_VIDEO, a51_key=A51_KEY, stego_key=STEGO_KEY)
    assert out["message"] == pesan
    print("1-1-1 Encrypted random roundtrip OK")

# ═══════════════════════════════════════════════════════════════════════════════
#  4-4-4 METHOD
# ═══════════════════════════════════════════════════════════════════════════════

def test_roundtrip_444_plain_sequential(pesan):
    result = embed_message(
        cover_path=SAMPLE_VIDEO, output_path=OUTPUT_VIDEO,
        message=pesan, is_text=True, extension=".txt", filename="pesan.txt",
        use_encryption=False, use_random=False, lsb_method=LSB_METHOD_444
    )
    print(f"\n4-4-4 PSNR: {result['psnr_avg']:.2f} dB | MSE: {result['mse_avg']:.4f}")

    out = extract_message(stego_path=OUTPUT_VIDEO)
    assert out["message"] == pesan
    assert out["lsb_method"] == LSB_METHOD_444

def test_roundtrip_444_encrypted_random(pesan):
    embed_message(
        cover_path=SAMPLE_VIDEO, output_path=OUTPUT_VIDEO,
        message=pesan, is_text=True, extension=".txt", filename="pesan.txt",
        use_encryption=True, a51_key=A51_KEY,
        use_random=True, stego_key=STEGO_KEY, lsb_method=LSB_METHOD_444
    )
    out = extract_message(stego_path=OUTPUT_VIDEO, a51_key=A51_KEY, stego_key=STEGO_KEY)
    assert out["message"] == pesan
    print("4-4-4 Encrypted random roundtrip OK")

# ═══════════════════════════════════════════════════════════════════════════════
#  ERROR CASES
# ═══════════════════════════════════════════════════════════════════════════════

def test_capacity_exceeded():
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip("Sample video tidak ada")
    frames, _ = read_video_frames(SAMPLE_VIDEO)
    cap = total_capacity_bytes(frames, LSB_METHOD_332)
    oversized = b"X" * (cap + 1)

    with pytest.raises(ValueError, match="Pesan terlalu besar"):
        embed_message(
            cover_path=SAMPLE_VIDEO, output_path=OUTPUT_VIDEO,
            message=oversized, is_text=True, use_random=False,
            lsb_method=LSB_METHOD_332
        )

def test_wrong_a51_key_fails(pesan):
    embed_message(
        cover_path=SAMPLE_VIDEO, output_path=OUTPUT_VIDEO,
        message=pesan, is_text=True, use_encryption=True, a51_key=A51_KEY,
        use_random=False, lsb_method=LSB_METHOD_332
    )
    out = extract_message(stego_path=OUTPUT_VIDEO, a51_key=0xDEADBEEFDEADBEEF)
    assert out["message"] != pesan
    print("Wrong A5/1 key correctly fails to recover message")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

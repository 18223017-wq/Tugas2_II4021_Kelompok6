# tests/test_stego_lsb.py
import os
import sys
import hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.stego_lsb_utils import bytes_to_bits, bits_to_bytes, pixel_indices_random
from src.stego_lsb_332 import (
    capacity_332,
    embed_bits_sequential_332, extract_bits_sequential_332,
    embed_bits_random_332, extract_bits_random_332,
)
from src.stego_lsb_111 import (
    capacity_111,
    embed_bits_sequential_111, extract_bits_sequential_111,
    embed_bits_random_111, extract_bits_random_111,
)
from src.stego_lsb_444 import (
    capacity_444,
    embed_bits_sequential_444, extract_bits_sequential_444,
    embed_bits_random_444, extract_bits_random_444,
)
from src.video_io import read_video_frames

SAMPLE_VIDEO = "samples/sample_video.avi"

# ─── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture
def real_frame():
    """Frame pertama dari sample_video.avi."""
    if not os.path.exists(SAMPLE_VIDEO):
        pytest.skip(f"Sample video tidak ada: {SAMPLE_VIDEO}")
    frames, _ = read_video_frames(SAMPLE_VIDEO)
    return frames[0]

@pytest.fixture
def sample_message():
    return b"Pesan rahasia Tugas 2 II4021 Kelompok 6 ITB!"

@pytest.fixture
def sample_message_file():
    """Baca dari samples/pesan.txt."""
    path = "samples/pesan.txt"
    if not os.path.exists(path):
        pytest.skip(f"pesan.txt tidak ada: {path}")
    with open(path, 'rb') as f:
        return f.read()

# ─── UTILS ────────────────────────────────────────────────────────────────────

def test_bytes_to_bits_and_back(sample_message):
    bits = bytes_to_bits(sample_message)
    assert len(bits) == len(sample_message) * 8
    assert set(bits).issubset({0, 1})

    recovered = bits_to_bytes(bits)[:len(sample_message)]
    assert recovered == sample_message

def test_bits_to_bytes_padding():
    """Padding ke kelipatan 8 tidak error."""
    bits = np.array([1, 0, 1], dtype=np.uint8)
    result = bits_to_bytes(bits)
    assert isinstance(result, bytes)
    assert len(result) == 1  # 3 bit → padding jadi 8 bit → 1 byte

# ═══════════════════════════════════════════════════════════════════════════════
#  3-3-2 METHOD
# ═══════════════════════════════════════════════════════════════════════════════

def test_capacity_332(real_frame):
    h, w, _ = real_frame.shape
    cap = capacity_332(real_frame)
    assert cap == h * w * 8
    print(f"\nKapasitas 3-3-2 frame {w}x{h}: {cap} bit = {cap // 8} bytes = {cap // 8 // 1024} KB")

def test_332_sequential_roundtrip(real_frame, sample_message):
    """Embed → extract pesan pendek, harus identical."""
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_sequential_332(real_frame, bits)
    extracted_bits = extract_bits_sequential_332(stego, len(bits))
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]
    assert recovered == sample_message

def test_332_sequential_roundtrip_file(real_frame, sample_message_file):
    """Embed → extract isi pesan.txt, harus identical."""
    bits = bytes_to_bits(sample_message_file)
    cap = capacity_332(real_frame)
    assert bits.size <= cap, f"pesan.txt terlalu besar untuk 1 frame: {bits.size} > {cap}"

    stego = embed_bits_sequential_332(real_frame, bits)
    extracted_bits = extract_bits_sequential_332(stego, len(bits))
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message_file)]
    assert recovered == sample_message_file
    print(f"\npesan.txt roundtrip OK ({len(sample_message_file)} bytes)")

def test_332_stego_frame_diff(real_frame, sample_message):
    """Frame asli dan stego harus beda tapi sangat mirip."""
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_sequential_332(real_frame, bits)

    assert not np.array_equal(real_frame, stego), "Stego frame identik dengan cover"
    diff = np.abs(real_frame.astype(int) - stego.astype(int))
    max_diff = diff.max()
    # 3-3-2: max diff = 7 (R/G, 3 bit) atau 3 (B, 2 bit)
    assert max_diff <= 7, f"Max diff > 7 (tidak wajar untuk 3-3-2): {max_diff}"
    print(f"\n3-3-2 Max pixel diff: {max_diff} (max=7)")

def test_332_capacity_exceeded(real_frame):
    cap = capacity_332(real_frame)
    oversized_bits = np.ones(cap + 1, dtype=np.uint8)
    with pytest.raises(ValueError, match="Payload terlalu besar"):
        embed_bits_sequential_332(real_frame, oversized_bits)

def test_332_random_roundtrip(real_frame, sample_message):
    seed = 42
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_random_332(real_frame, bits, seed=seed)
    extracted_bits = extract_bits_random_332(stego, len(bits), seed=seed)
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]
    assert recovered == sample_message
    print(f"\n3-3-2 Random roundtrip OK (seed={seed})")

def test_332_random_wrong_seed_fails(real_frame, sample_message):
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_random_332(real_frame, bits, seed=42)
    extracted_bits = extract_bits_random_332(stego, len(bits), seed=999)
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]
    assert recovered != sample_message

# ═══════════════════════════════════════════════════════════════════════════════
#  1-1-1 METHOD
# ═══════════════════════════════════════════════════════════════════════════════

def test_capacity_111(real_frame):
    h, w, _ = real_frame.shape
    cap = capacity_111(real_frame)
    assert cap == h * w * 3
    print(f"\nKapasitas 1-1-1 frame {w}x{h}: {cap} bit = {cap // 8} bytes = {cap // 8 // 1024} KB")

def test_111_sequential_roundtrip(real_frame, sample_message):
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_sequential_111(real_frame, bits)
    extracted_bits = extract_bits_sequential_111(stego, len(bits))
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]
    assert recovered == sample_message

def test_111_stego_frame_diff(real_frame, sample_message):
    """1-1-1: max diff per channel = 1."""
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_sequential_111(real_frame, bits)

    assert not np.array_equal(real_frame, stego), "Stego frame identik dengan cover"
    diff = np.abs(real_frame.astype(int) - stego.astype(int))
    max_diff = diff.max()
    assert max_diff <= 1, f"Max diff > 1 (tidak wajar untuk 1-1-1): {max_diff}"
    print(f"\n1-1-1 Max pixel diff: {max_diff} (max=1)")

def test_111_capacity_exceeded(real_frame):
    cap = capacity_111(real_frame)
    oversized_bits = np.ones(cap + 1, dtype=np.uint8)
    with pytest.raises(ValueError, match="Payload terlalu besar"):
        embed_bits_sequential_111(real_frame, oversized_bits)

def test_111_random_roundtrip(real_frame, sample_message):
    seed = 42
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_random_111(real_frame, bits, seed=seed)
    extracted_bits = extract_bits_random_111(stego, len(bits), seed=seed)
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]
    assert recovered == sample_message
    print(f"\n1-1-1 Random roundtrip OK (seed={seed})")

# ═══════════════════════════════════════════════════════════════════════════════
#  4-4-4 METHOD
# ═══════════════════════════════════════════════════════════════════════════════

def test_capacity_444(real_frame):
    h, w, _ = real_frame.shape
    cap = capacity_444(real_frame)
    assert cap == h * w * 12
    print(f"\nKapasitas 4-4-4 frame {w}x{h}: {cap} bit = {cap // 8} bytes = {cap // 8 // 1024} KB")

def test_444_sequential_roundtrip(real_frame, sample_message):
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_sequential_444(real_frame, bits)
    extracted_bits = extract_bits_sequential_444(stego, len(bits))
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]
    assert recovered == sample_message

def test_444_stego_frame_diff(real_frame, sample_message):
    """4-4-4: max diff per channel = 15."""
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_sequential_444(real_frame, bits)

    assert not np.array_equal(real_frame, stego), "Stego frame identik dengan cover"
    diff = np.abs(real_frame.astype(int) - stego.astype(int))
    max_diff = diff.max()
    assert max_diff <= 15, f"Max diff > 15 (tidak wajar untuk 4-4-4): {max_diff}"
    print(f"\n4-4-4 Max pixel diff: {max_diff} (max=15)")

def test_444_capacity_exceeded(real_frame):
    cap = capacity_444(real_frame)
    oversized_bits = np.ones(cap + 1, dtype=np.uint8)
    with pytest.raises(ValueError, match="Payload terlalu besar"):
        embed_bits_sequential_444(real_frame, oversized_bits)

def test_444_random_roundtrip(real_frame, sample_message):
    seed = 42
    bits = bytes_to_bits(sample_message)
    stego = embed_bits_random_444(real_frame, bits, seed=seed)
    extracted_bits = extract_bits_random_444(stego, len(bits), seed=seed)
    recovered = bits_to_bytes(extracted_bits)[:len(sample_message)]
    assert recovered == sample_message
    print(f"\n4-4-4 Random roundtrip OK (seed={seed})")

# ═══════════════════════════════════════════════════════════════════════════════
#  CROSS-METHOD COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════════

def test_capacity_ordering(real_frame):
    """1-1-1 < 3-3-2 < 4-4-4 capacity."""
    c111 = capacity_111(real_frame)
    c332 = capacity_332(real_frame)
    c444 = capacity_444(real_frame)
    assert c111 < c332 < c444
    print(f"\nCapacity: 1-1-1={c111} < 3-3-2={c332} < 4-4-4={c444}")

def test_methods_produce_different_stego(real_frame, sample_message):
    """Same data embedded with different methods → different stego frames."""
    bits = bytes_to_bits(sample_message)
    s111 = embed_bits_sequential_111(real_frame, bits)
    s332 = embed_bits_sequential_332(real_frame, bits)
    s444 = embed_bits_sequential_444(real_frame, bits)
    assert not np.array_equal(s111, s332)
    assert not np.array_equal(s332, s444)
    assert not np.array_equal(s111, s444)
    print("\nAll 3 methods produce different stego frames ✓")

def test_random_vs_sequential_differ(real_frame, sample_message):
    """Stego hasil random dan sequential harus berbeda."""
    bits = bytes_to_bits(sample_message)
    stego_seq = embed_bits_sequential_332(real_frame, bits)
    stego_rnd = embed_bits_random_332(real_frame, bits, seed=42)
    assert not np.array_equal(stego_seq, stego_rnd)

def test_pixel_indices_random_reproducible():
    """Seed yang sama harus selalu hasilkan urutan yang sama."""
    idx1 = pixel_indices_random(480, 640, seed=42)
    idx2 = pixel_indices_random(480, 640, seed=42)
    assert np.array_equal(idx1, idx2)

    idx3 = pixel_indices_random(480, 640, seed=99)
    assert not np.array_equal(idx1, idx3)

# ─── OUTPUT SAVE ──────────────────────────────────────────────────────────────

def test_save_stego_output(real_frame, sample_message):
    """Simpan frame stego ke tests_output/ untuk inspeksi visual."""
    import cv2
    os.makedirs("tests_output", exist_ok=True)

    bits = bytes_to_bits(sample_message)

    stego_111 = embed_bits_sequential_111(real_frame, bits)
    stego_332 = embed_bits_sequential_332(real_frame, bits)
    stego_444 = embed_bits_sequential_444(real_frame, bits)

    cv2.imwrite("tests_output/cover_frame.png", real_frame)
    cv2.imwrite("tests_output/stego_111_frame.png", stego_111)
    cv2.imwrite("tests_output/stego_332_frame.png", stego_332)
    cv2.imwrite("tests_output/stego_444_frame.png", stego_444)

    diff_332 = cv2.absdiff(real_frame, stego_332)
    cv2.imwrite("tests_output/diff_332.png", diff_332 * 50)

    diff_444 = cv2.absdiff(real_frame, stego_444)
    cv2.imwrite("tests_output/diff_444.png", diff_444 * 20)

    print(f"\n  Saved to tests_output/")
    assert os.path.exists("tests_output/stego_332_frame.png")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

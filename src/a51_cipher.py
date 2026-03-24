# src/a51_cipher.py
from dataclasses import dataclass
from typing import List

BLOCK_SIZE_BITS = 228  

@dataclass
class LFSR:
    size: int
    taps: List[int]
    clk_bit: int  
    reg: int = 0

    def clock(self) -> int:
        out = self.reg & 1
        feedback = 0
        for t in self.taps:
            feedback ^= (self.reg >> t) & 1
        self.reg = ((self.reg >> 1) | (feedback << (self.size - 1))) & ((1 << self.size) - 1)
        return out

    def get_clk_bit(self) -> int:
        return (self.reg >> self.clk_bit) & 1


def majority(x: int, y: int, z: int) -> int:
    return 1 if (x + y + z) >= 2 else 0


class A51:
    """
    A5/1 stream cipher sesuai spesifikasi GSM + tugas:
    - Key 64-bit di-load ke 3 LFSR
    - Frame number (Fn) 22-bit di-load per blok
    - Majority clocking
    """

    def __init__(self):
        self.r1 = LFSR(size=19, taps=[13, 16, 17, 18], clk_bit=8)
        self.r2 = LFSR(size=22, taps=[20, 21],         clk_bit=10)
        self.r3 = LFSR(size=23, taps=[7, 20, 21, 22],  clk_bit=10)

    def _load_key(self, key_64bit: int):
        """XOR key 64-bit ke semua LFSR bit per bit."""
        self.r1.reg = 0
        self.r2.reg = 0
        self.r3.reg = 0

        for i in range(64):
            bit = (key_64bit >> i) & 1
            self.r1.reg ^= (bit << (i % self.r1.size))
            self.r2.reg ^= (bit << (i % self.r2.size))
            self.r3.reg ^= (bit << (i % self.r3.size))

    def _load_fn(self, fn: int):
        """XOR frame number 22-bit ke semua LFSR."""
        for i in range(22):
            bit = (fn >> i) & 1
            self.r1.reg ^= (bit << (i % self.r1.size))
            self.r2.reg ^= (bit << (i % self.r2.size))
            self.r3.reg ^= (bit << (i % self.r3.size))

    def _warmup(self, cycles: int = 100):
        """Jalankan clocking tanpa ambil output (warmup)."""
        for _ in range(cycles):
            self._majority_clock()

    def _majority_clock(self) -> int:
        """Clock berdasarkan majority dari clk_bit ketiga LFSR."""
        c1 = self.r1.get_clk_bit()
        c2 = self.r2.get_clk_bit()
        c3 = self.r3.get_clk_bit()
        maj = majority(c1, c2, c3)

        b1 = self.r1.clock() if c1 == maj else 0
        b2 = self.r2.clock() if c2 == maj else 0
        b3 = self.r3.clock() if c3 == maj else 0

        return b1 ^ b2 ^ b3

    def _init_for_block(self, key_64bit: int, fn: int):
        """Inisialisasi LFSR untuk satu blok dengan key + Fn."""
        self._load_key(key_64bit)
        self._load_fn(fn)
        self._warmup(100)

    def _generate_block_keystream(self, key_64bit: int, fn: int) -> List[int]:
        """Generate keystream 228 bit untuk satu blok."""
        self._init_for_block(key_64bit, fn)
        return [self._majority_clock() for _ in range(BLOCK_SIZE_BITS)]

    def encrypt(self, data: bytes, key_64bit: int) -> bytes:
        """
        Enkripsi data dengan A5/1.
        Payload dibagi blok 228-bit, tiap blok pakai Fn otomatis. [file:1]
        """
        bits_in = _bytes_to_bits_list(data)
        bits_out = []
        fn = 0

        for i in range(0, len(bits_in), BLOCK_SIZE_BITS):
            block = bits_in[i:i + BLOCK_SIZE_BITS]
            keystream = self._generate_block_keystream(key_64bit, fn)
            bits_out.extend(b ^ k for b, k in zip(block, keystream))
            fn += 1

        return _bits_list_to_bytes(bits_out)

    def decrypt(self, data: bytes, key_64bit: int) -> bytes:
        return self.encrypt(data, key_64bit) 




def _bytes_to_bits_list(data: bytes) -> List[int]:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def _bits_list_to_bytes(bits: List[int]) -> bytes:
    result = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        if len(chunk) < 8:
            chunk += [0] * (8 - len(chunk))
        byte = 0
        for b in chunk:
            byte = (byte << 1) | b
        result.append(byte)
    return bytes(result)


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def a51_encrypt_payload(payload: bytes, key_64bit: int) -> bytes:
    return A51().encrypt(payload, key_64bit)

def a51_decrypt_payload(ciphertext: bytes, key_64bit: int) -> bytes:
    return A51().decrypt(ciphertext, key_64bit)

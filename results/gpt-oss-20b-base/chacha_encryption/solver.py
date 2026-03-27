from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32  # 256‑bit key for ChaCha20
POLY1305_TAG_SIZE = 16  # 128‑bit authentication tag

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        if len(key) != CHACHA_KEY_SIZE:
            raise ValueError(f"Invalid key size: {len(key)}. Must be {CHACHA_KEY_SIZE}.")

        # ChaCha20-Poly1305 encrypts and appends a 16‑byte tag.
        ciphertext = ChaCha20Poly1305(key).encrypt(
            problem["nonce"], problem["plaintext"], problem["associated_data"]
        )

        # Separate ciphertext and tag.
        return {
            "ciphertext": ciphertext[:-POLY1305_TAG_SIZE],
            "tag": ciphertext[-POLY1305_TAG_SIZE:],
        }
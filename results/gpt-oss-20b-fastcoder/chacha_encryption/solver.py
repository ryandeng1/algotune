from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from typing import Any

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        aad = problem["associated_data"]

        # Validate key length (crucial for correctness, but cheap to check)
        if len(key) != CHACHA_KEY_SIZE:
            raise ValueError(f"Invalid key size: {len(key)}. Must be {CHACHA_KEY_SIZE}.")

        c = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)

        return {
            "ciphertext": c[:-POLY1305_TAG_SIZE],
            "tag": c[-POLY1305_TAG_SIZE:],
        }
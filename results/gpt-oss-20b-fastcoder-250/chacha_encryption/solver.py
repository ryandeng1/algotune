from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        if len(key) != CHACHA_KEY_SIZE:
            raise ValueError(f"Invalid key size: {len(key)}. Must be {CHACHA_KEY_SIZE}.")

        chacha = ChaCha20Poly1305(key)
        ciphertext = chacha.encrypt(problem["nonce"], problem["plaintext"], problem["associated_data"])

        if len(ciphertext) < POLY1305_TAG_SIZE:
            raise ValueError("Encrypted output is shorter than the expected tag size.")

        return {
            "ciphertext": ciphertext[:-POLY1305_TAG_SIZE],
            "tag": ciphertext[-POLY1305_TAG_SIZE:]
        }
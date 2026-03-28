from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AES_KEY_SIZES = {16, 24, 32}  # AES-128, AES-192, AES-256

GCM_TAG_SIZE = 16  # Standard tag size for GCM

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        key_len = len(key)
        if key_len not in AES_KEY_SIZES:
            raise ValueError(f"Invalid key size: {key_len}. Must be one of {AES_KEY_SIZES}.")

        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        if len(ciphertext) < GCM_TAG_SIZE:
            raise ValueError("Encrypted output is shorter than the expected tag size.")

        split_index = len(ciphertext) - GCM_TAG_SIZE
        actual_ciphertext = ciphertext[:split_index]
        tag = ciphertext[split_index:]

        return {"ciphertext": actual_ciphertext, "tag": tag}
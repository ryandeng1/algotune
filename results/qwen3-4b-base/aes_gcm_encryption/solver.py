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

        # Validate key size based on provided key length
        if len(key) not in AES_KEY_SIZES:
            raise ValueError(f"Invalid key size: {len(key)}. Must be one of {AES_KEY_SIZES}.")

        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        actual_ciphertext = ciphertext[:-GCM_TAG_SIZE]
        tag = ciphertext[-GCM_TAG_SIZE:]

        return {"ciphertext": actual_ciphertext, "tag": tag}
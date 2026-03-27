from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AES_KEY_SIZES = {16, 24, 32}          # set for O(1) membership test
GCM_TAG_SIZE = 16                     # standard tag size

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        if len(key) not in AES_KEY_SIZES:
            raise ValueError(
                f"Invalid key size: {len(key)}. Must be one of {sorted(AES_KEY_SIZES)}."
            )

        aesgcm = AESGCM(key)
        ct_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data)

        if len(ct_with_tag) < GCM_TAG_SIZE:
            raise ValueError("Encrypted output is shorter than the expected tag size.")

        # Split ciphertext and tag
        return {
            "ciphertext": ct_with_tag[:-GCM_TAG_SIZE],
            "tag": ct_with_tag[-GCM_TAG_SIZE:],
        }
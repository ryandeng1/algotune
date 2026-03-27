from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# Pre‑defined allowed sizes – using a set improves lookup speed
_ALLOWED_KEY_SIZES = {16, 24, 32}          # AES‑128/192/256

# Standard GCM tag size (bytes)
_GCM_TAG_SIZE = 16


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encrypt the plaintext using AES‑GCM from the `cryptography` library.

        The function assumes that all inputs (`key`, `nonce`, `plaintext`,
        `associated_data`) are already bytes-like objects of appropriate length.
        No additional error handling is performed beyond key‑size validation.
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        aad = problem["associated_data"]

        # Validate key size – O(1) set lookup
        if len(key) not in _ALLOWED_KEY_SIZES:
            raise ValueError(
                f"Invalid key size: {len(key)}. Must be one of "
                f"{sorted(_ALLOWED_KEY_SIZES)}."
            )

        # Create the cipher object once per call
        aesgcm = AESGCM(key)

        # Perform encryption – `encrypt` returns ciphertext with the tag appended
        ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext, aad)

        # Split ciphertext and tag
        ciphertext_len = len(ciphertext_and_tag) - _GCM_TAG_SIZE
        if ciphertext_len < 0:
            raise ValueError("Encrypted output shorter than expected tag size.")

        return {
            "ciphertext": ciphertext_and_tag[:ciphertext_len],
            "tag": ciphertext_and_tag[ciphertext_len:],
        }
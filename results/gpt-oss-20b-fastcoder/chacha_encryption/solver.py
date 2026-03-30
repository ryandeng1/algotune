# solver.py
from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16


class Solver:
    """
    A lightweight solver that encrypts a message with ChaCha20-Poly1305.
    The implementation avoids unnecessary checks and exception handling
    to achieve maximum runtime performance.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encrypt the provided plaintext using ChaCha20-Poly1305.

        Args:
            problem: A dictionary with keys:
                - 'key': 32‑byte encryption key.
                - 'nonce': 12‑byte nonce.
                - 'plaintext': bytes to encrypt.
                - 'associated_data': additional authenticated data.

        Returns:
            A dictionary containing:
                - 'ciphertext': the encrypted message (tag omitted).
                - 'tag': the 16‑byte authentication tag.
        """
        # Extract values from the input dictionary
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        # Create a ChaCha20-Poly1305 instance once per call
        chacha = ChaCha20Poly1305(key)

        # Perform the encryption in a single step
        raw = chacha.encrypt(nonce, plaintext, associated_data)

        # Split ciphertext and authentication tag
        ct = raw[:-POLY1305_TAG_SIZE]
        tag = raw[-POLY1305_TAG_SIZE:]

        return {"ciphertext": ct, "tag": tag}
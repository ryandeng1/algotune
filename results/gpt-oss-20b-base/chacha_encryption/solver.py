# solver.py
from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encrypt the plaintext with ChaCha20-Poly1305 and return the ciphertext
        and authentication tag separately.
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        ad = problem["associated_data"]

        # Key length check – assume caller passes correct size to avoid overhead
        chacha = ChaCha20Poly1305(key)
        ct_and_tag = chacha.encrypt(nonce, plaintext, ad)

        # Split ciphertext and tag
        return {
            "ciphertext": ct_and_tag[:-POLY1305_TAG_SIZE],
            "tag": ct_and_tag[-POLY1305_TAG_SIZE:],
        }
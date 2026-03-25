# solver.py
import hmac
from typing import Any, Dict
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32            # 256-bit key
POLY1305_TAG_SIZE = 16          # 16-byte tag


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Encrypt the plaintext using ChaCha20-Poly1305.

        The input dictionary must contain:
            - "key": 32‑byte key
            - "nonce": 12‑byte nonce
            - "plaintext": bytes to encrypt
            - "associated_data": optional bytes (can be None or b'')

        Returns a dictionary with:
            - "ciphertext": encrypted data without the tag
            - "tag": 16‑byte authentication tag
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        aad = problem.get("associated_data", None)

        if not isinstance(key, bytes) or len(key) != CHACHA_KEY_SIZE:
            raise ValueError(
                f"Key must be {CHACHA_KEY_SIZE} bytes; got {len(key) if isinstance(key, bytes) else type(key)}"
            )
        if not isinstance(nonce, bytes) or len(nonce) != 12:
            raise ValueError("Nonce must be 12 bytes for ChaCha20-Poly1305")

        # Encrypt and split tag
        chacha = ChaCha20Poly1305(key)
        ciphertext_with_tag = chacha.encrypt(nonce, plaintext, aad)
        if len(ciphertext_with_tag) < POLY1305_TAG_SIZE:
            raise ValueError("Ciphertext too short to contain a valid tag")

        return {
            "ciphertext": ciphertext_with_tag[:-POLY1305_TAG_SIZE],
            "tag": ciphertext_with_tag[-POLY1305_TAG_SIZE:],
        }

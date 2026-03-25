# solver.py
import hmac
from typing import Any, Dict
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# Constants for key and tag sizes (standard for ChaCha20-Poly1305)
CHACHA_KEY_SIZE = 32          # 256-bit key
POLY1305_TAG_SIZE = 16        # Authentication tag length

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Encrypts the plaintext using ChaCha20-Poly1305 AEAD cipher.

        Args:
            problem: A dictionary with the following keys:
                - "key": 32-byte key
                - "nonce": 12-byte nonce
                - "plaintext": data to encrypt
                - "associated_data": optional associated data (bytes or None)

        Returns:
            A dictionary with:
                - "ciphertext": encrypted data (without the tag)
                - "tag": 16-byte authentication tag
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        # Validate key size
        if len(key) != CHACHA_KEY_SIZE:
            raise ValueError(f"Invalid key size: {len(key)}. Expected {CHACHA_KEY_SIZE} bytes.")

        # Use ChaCha20Poly1305 AEAD cipher
        chacha = ChaCha20Poly1305(key)
        # encrypt returns ciphertext + tag appended at the end
        ct_and_tag = chacha.encrypt(nonce, plaintext, associated_data or b"")

        # Separate ciphertext and tag
        if len(ct_and_tag) < POLY1305_TAG_SIZE:
            raise ValueError("Ciphertext too short to contain a valid tag.")
        ciphertext = ct_and_tag[:-POLY1305_TAG_SIZE]
        tag = ct_and_tag[-POLY1305_TAG_SIZE:]

        return {"ciphertext": ciphertext, "tag": tag}

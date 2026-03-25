# solver.py
from typing import Any, Dict
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Constants matching the reference implementation
_AES_KEY_SIZES = {16, 24, 32}
_GCM_TAG_SIZE = 16

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Encrypt the provided plaintext with AES-GCM.

        Parameters
        ----------
        problem : dict
            A dictionary containing the keys:
                - "key" : bytes (16, 24, or 32 bytes)
                - "nonce" : bytes (recommended 12 bytes)
                - "plaintext" : bytes
                - "associated_data" : bytes or None

        Returns
        -------
        dict
            {'ciphertext': bytes, 'tag': bytes}
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        # Basic key size validation
        if len(key) not in _AES_KEY_SIZES:
            raise ValueError(f"Invalid key size: {len(key)}. Must be one of {_AES_KEY_SIZES}.")

        aesgcm = AESGCM(key)
        ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data)

        # Split ciphertext and tag
        if len(ciphertext_with_tag) < _GCM_TAG_SIZE:
            raise ValueError("Encrypted output is shorter than the expected tag size.")

        return {
            "ciphertext": ciphertext_with_tag[:- _GCM_TAG_SIZE],
            "tag": ciphertext_with_tag[-_GCM_TAG_SIZE:],
        }

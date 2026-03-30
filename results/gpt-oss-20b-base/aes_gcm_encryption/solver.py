# solver.py
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Any, Dict

# Pre‑set constants to avoid repeated look‑ups
_AES_KEY_SIZES = {16, 24, 32}
_GCM_TAG_SIZE = 16

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Encrypt the plaintext using AES‑GCM with minimal overhead.

        Parameters
        ----------
        problem : dict
            Must contain the keys 'key', 'nonce', 'plaintext', 'associated_data'.

        Returns
        -------
        dict
            Dictionary with 'ciphertext' and 'tag' keys containing the
            encrypted data and GCM authentication tag respectively.
        """
        key = problem["key"]
        # Validate key size once – this is cheap but essential
        if len(key) not in _AES_KEY_SIZES:
            raise ValueError(
                f"Invalid key size {len(key)}. Expected one of {_AES_KEY_SIZES}."
            )

        aesgcm = AESGCM(key)

        # The cryptography library returns ciphertext + tag
        ct_tag = aesgcm.encrypt(
            problem["nonce"],
            problem["plaintext"],
            problem["associated_data"],
        )

        # Split into ciphertext and tag without extra allocations
        tag = ct_tag[-_GCM_TAG_SIZE:]
        ciphertext = ct_tag[:-_GCM_TAG_SIZE]
        return {"ciphertext": ciphertext, "tag": tag}
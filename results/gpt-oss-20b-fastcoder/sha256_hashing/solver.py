# solver.py

from typing import Any, Dict

import hashlib


class Solver:
    """
    Compute the SHA‑256 hash of a given plaintext.
    Uses the highly optimised `hashlib` backend for speed.
    """

    def __call__(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Return the SHA‑256 digest of the provided plaintext.

        Parameters
        ----------
        problem :
            Dictionary containing a key ``"plaintext"`` whose value is a ``bytes`` object.
            If a string is provided it is encoded using UTF‑8.

        Returns
        -------
        dict
            Dictionary with a single key ``"digest"`` containing the hash as bytes.
        """
        plaintext = problem.get("plaintext")
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError("plaintext must be bytes-like")

        # hashlib handles the hashing internally in C for maximum speed
        hash_value = hashlib.sha256(plaintext).digest()
        return {"digest": hash_value}
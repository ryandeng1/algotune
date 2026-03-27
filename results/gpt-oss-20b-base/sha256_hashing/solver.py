from typing import Any
import hashlib


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Compute the SHA-256 hash of the plaintext using the standard hashlib module,
        which is significantly faster than the cryptography library for this use case.

        :param problem: A dictionary containing the problem with key "plaintext".
        :return: A dictionary with key "digest" containing the SHA-256 hash value.
        """
        plaintext = problem["plaintext"]
        # `hashlib.sha256` accepts bytes-like objects directly
        hash_value = hashlib.sha256(plaintext).digest()
        return {"digest": hash_value}
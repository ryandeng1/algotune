import hashlib
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Compute the SHA-256 hash of the plaintext using the standard library hashlib.

        :param problem: A dictionary containing the problem with key "plaintext".
        :return: A dictionary with key "digest" containing the SHA-256 hash value.
        """
        plaintext = problem['plaintext']
        # hashlib works with bytes; the input is assumed to be bytes
        return {'digest': hashlib.sha256(plaintext).digest()}
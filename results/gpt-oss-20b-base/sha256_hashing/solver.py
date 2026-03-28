import hashlib
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """Return the SHA-256 digest of the given bytes."""
        # `hashlib.sha256` is a C implementation and significantly faster
        # than the pure‑Python fallback used by the cryptography library.
        return {"digest": hashlib.sha256(problem["plaintext"]).digest()}
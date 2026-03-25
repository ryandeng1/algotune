from __future__ import annotations
import gzip
import math
from typing import Any, Dict


class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Dict[str, bytes]:
        plaintext = problem.get("plaintext")
        if plaintext is None:
            raise ValueError("Problem missing 'plaintext' key")

        # Compress using deterministic gzip parameters
        compressed_data = gzip.compress(plaintext, compresslevel=9, mtime=0)

        # Ensure result is bytes (gzip.compress already returns bytes)
        compressed_data = bytes(compressed_data)

        return {"compressed_data": compressed_data}

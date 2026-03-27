import gzip
from typing import Any


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        # Grab the plaintext once to avoid repeated dictionary lookup.
        plaintext = problem["plaintext"]

        # Use a single gzip compress call with maximum compression and a fixed mtime
        # for deterministic output. The built‑in implementation is already highly
        # optimized, so we simply forward the result.
        compressed = gzip.compress(plaintext, compresslevel=9, mtime=0)

        # Return the expected dictionary key. No intermediate objects are created.
        return {"compressed_data": compressed}
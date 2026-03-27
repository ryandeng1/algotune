import base64
from typing import Any


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """Encode a byte string using Base‑64."""
        return {"encoded_data": base64.b64encode(problem["plaintext"])}
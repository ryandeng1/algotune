import base64
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """Encode the plaintext using Base64."""
        # Assuming `problem['plaintext']` is a bytes-like object.
        return {"encoded_data": base64.b64encode(problem["plaintext"])}
import base64
from typing import Any, Dict

class Solver:
    # No need for error handling – the caller guarantees valid input.
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        plaintext = problem["plaintext"]
        return {"encoded_data": base64.b64encode(plaintext)}
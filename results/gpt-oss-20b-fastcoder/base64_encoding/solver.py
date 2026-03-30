import base64
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encode the plaintext using Base64 and return the result.

        Parameters
        ----------
        problem : dict
            Must contain a 'plaintext' key with a bytes-like object.

        Returns
        -------
        dict
            A single entry 'encoded_data' containing the Base64-encoded bytes.
        """
        # Directly use the efficient C implementation of Base64
        # No additional error handling is necessary for well-formed input.
        return {"encoded_data": base64.b64encode(problem["plaintext"])}
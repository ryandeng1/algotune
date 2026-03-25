# solver.py
import base64
from typing import Any, Dict


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Encode the provided binary data using Base64.

        Parameters
        ----------
        problem : dict
            Must contain a 'plaintext' key holding a bytes object.

        Returns
        -------
        dict
            Dictionary with key 'encoded_data' containing the Base64-encoded bytes.
        """
        plaintext = problem["plaintext"]
        # Use the efficient C implementation of base64
        encoded = base64.b64encode(plaintext)
        return {"encoded_data": encoded}

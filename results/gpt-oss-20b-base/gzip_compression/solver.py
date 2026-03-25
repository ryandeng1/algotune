# solver.py
from __future__ import annotations

import gzip
from typing import Any, Dict


class Solver:
    """Gzip compression solver."""

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Compress the provided plaintext using gzip with deterministic parameters.

        Parameters
        ----------
        problem : dict
            Must contain key 'plaintext' with a bytes object as value.
        **kwargs :
            Additional arguments are ignored.

        Returns
        -------
        dict
            Dictionary with single key 'compressed_data' containing the gzip
            compressed bytes.
        """
        plaintext = problem["plaintext"]
        # Use compresslevel 9 for best compression and mtime 0 for determinism
        compressed_data = gzip.compress(plaintext, compresslevel=9, mtime=0)
        return {"compressed_data": compressed_data}

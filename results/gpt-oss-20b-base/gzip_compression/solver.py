import gzip
from typing import Any, Dict

def solve(problem: Dict[str, Any]) -> Dict[str, bytes]:
    """
    Compress the plaintext using gzip with maximum compression and no timestamp.
    """
    # gzip.compress handles bytes directly; compresslevel 9 is the fastest to produce smallest
    return {"compressed_data": gzip.compress(problem["plaintext"], compresslevel=9, mtime=0)}
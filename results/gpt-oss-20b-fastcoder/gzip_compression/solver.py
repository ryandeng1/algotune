import gzip

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Compress the plaintext using gzip with mtime=0 and maximum compression.
        """
        plaintext = problem["plaintext"]
        # gzip.compress uses ZLIB at level 9 by default and mtime=0 when passed,
        # but we specify compresslevel explicitly for clarity.
        compressed = gzip.compress(plaintext, compresslevel=9, mtime=0)
        return {"compressed_data": compressed}
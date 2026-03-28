from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    def solve(self, problem: dict) -> dict:
        key = problem["key"]
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(problem["nonce"], problem["plaintext"], problem["associated_data"])
        return {"ciphertext": ct[:-16], "tag": ct[-16:]}
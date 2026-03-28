from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AES_KEY_SIZES = {16, 24, 32}
GCM_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict[str, bytes]) -> dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        if len(key) not in AES_KEY_SIZES:
            raise ValueError(f"Invalid key size: {len(key)}. Must be one of {sorted(AES_KEY_SIZES)}")

        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, plaintext, associated_data)

        if len(ct) < GCM_TAG_SIZE:
            raise ValueError("Encrypted output is shorter than the expected tag size")

        return {"ciphertext": ct[:-GCM_TAG_SIZE], "tag": ct[-GCM_TAG_SIZE:]}
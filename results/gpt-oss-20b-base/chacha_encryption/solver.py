from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict) -> dict:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem["associated_data"]

        # assume caller passes correct input sizes
        chacha = ChaCha20Poly1305(key)
        ciphertext = chacha.encrypt(nonce, plaintext, associated_data)
        return {
            "ciphertext": ciphertext[:-POLY1305_TAG_SIZE],
            "tag": ciphertext[-POLY1305_TAG_SIZE:],
        }
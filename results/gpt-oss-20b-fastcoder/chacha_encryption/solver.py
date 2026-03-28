from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict) -> dict:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        ad = problem["associated_data"]

        # no error handling for speed
        chacha = ChaCha20Poly1305(key)
        ct_tag = chacha.encrypt(nonce, plaintext, ad)
        return {
            "ciphertext": ct_tag[:-POLY1305_TAG_SIZE],
            "tag": ct_tag[-POLY1305_TAG_SIZE:],
        }
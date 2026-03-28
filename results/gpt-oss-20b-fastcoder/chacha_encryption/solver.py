from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict[str, bytes]) -> dict[str, bytes]:
        key = problem['key']
        nonce = problem['nonce']
        plaintext = problem['plaintext']
        associated_data = problem['associated_data']

        chacha = ChaCha20Poly1305(key)
        ct = chacha.encrypt(nonce, plaintext, associated_data)

        return {
            'ciphertext': ct[:-POLY1305_TAG_SIZE],
            'tag': ct[-POLY1305_TAG_SIZE:]
        }
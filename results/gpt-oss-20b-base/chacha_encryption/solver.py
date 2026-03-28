from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Encrypt the plaintext with ChaCha20-Poly1305 and return the ciphertext
        (excluding the authentication tag) together with the tag itself.
        """
        key = problem['key']
        if len(key) != CHACHA_KEY_SIZE:
            raise ValueError(f'Invalid key size: {len(key)}. Must be {CHACHA_KEY_SIZE}.')

        chacha = ChaCha20Poly1305(key)
        # encrypt returns ciphertext || tag
        ct_tag = chacha.encrypt(problem['nonce'], problem['plaintext'], problem['associated_data'])
        if len(ct_tag) < POLY1305_TAG_SIZE:
            raise ValueError('Encrypted output is shorter than the expected tag size.')

        return {
            'ciphertext': ct_tag[:-POLY1305_TAG_SIZE],
            'tag': ct_tag[-POLY1305_TAG_SIZE:]
        }
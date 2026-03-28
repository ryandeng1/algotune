from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Acceptance constants
_AES_KEY_SIZES = {16, 24, 32}
_GCM_TAG_SIZE = 16

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        key = problem['key']
        nonce = problem['nonce']
        plaintext = problem['plaintext']
        ad = problem['associated_data']

        # Lightweight validation
        if len(key) not in _AES_KEY_SIZES:
            raise ValueError(f'Invalid key size: {len(key)}. Must be one of {sorted(_AES_KEY_SIZES)}.')

        aesgcm = AESGCM(key)
        ct_tag = aesgcm.encrypt(nonce, plaintext, ad)

        # Split ciphertext and tag
        return {
            'ciphertext': ct_tag[:- _GCM_TAG_SIZE],
            'tag': ct_tag[- _GCM_TAG_SIZE:],
        }
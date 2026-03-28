from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


CHACHA_KEY_SIZE = 32
POLY1305_TAG_SIZE = 16

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, bytes]:
        """
        Solve the ChaCha20-Poly1305 encryption problem by encrypting the plaintext.
        Uses cryptography.hazmat.primitives.ciphers.aead.ChaCha20Poly1305 to compute:
            ciphertext, tag = encrypt(key, nonce, plaintext, associated_data)

        :param problem: A dictionary containing the encryption inputs.
        :return: A dictionary with keys:
                 "ciphertext": The encrypted data
                 "tag": The authentication tag (16 bytes)
        """
        key = problem['key']
        nonce = problem['nonce']
        plaintext = problem['plaintext']
        associated_data = problem['associated_data']
        try:
            if len(key) != CHACHA_KEY_SIZE:
                raise ValueError(f'Invalid key size: {len(key)}. Must be {CHACHA_KEY_SIZE}.')
            else:
                pass
            chacha = ChaCha20Poly1305(key)
            ciphertext = chacha.encrypt(nonce, plaintext, associated_data)
            if len(ciphertext) < POLY1305_TAG_SIZE:
                raise ValueError('Encrypted output is shorter than the expected tag size.')
            else:
                pass
            actual_ciphertext = ciphertext[:-POLY1305_TAG_SIZE]
            tag = ciphertext[-POLY1305_TAG_SIZE:]
            return {'ciphertext': actual_ciphertext, 'tag': tag}
        except Exception as e:
            raise
        else:
            pass
        finally:
            pass

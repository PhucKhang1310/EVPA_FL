"""Cryptographic primitives for EVPA Protocol I implementation."""

import hashlib
import hmac
import secrets
from typing import Tuple, List, Dict
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import numpy as np


class KeyAgreement:
    """Simulated Key Agreement protocol for EVPA."""
    
    @staticmethod
    def gen(pp: bytes) -> Tuple[bytes, bytes]:
        """Generate a key pair (public_key, secret_key)."""
        secret_key = secrets.token_bytes(32)  # 256-bit secret key
        # Derive public key from secret key (simplified)
        public_key = hashlib.sha256(secret_key + pp).digest()
        return public_key, secret_key
    
    @staticmethod
    def agree(secret_key: bytes, public_key: bytes) -> bytes:
        """Compute shared secret from secret key and public key."""
        # Simplified key agreement - in practice would use ECDH
        return hashlib.sha256(secret_key + public_key).digest()


class AuthenticatedEncryption:
    """Authenticated Encryption for EVPA protocol."""
    
    @staticmethod
    def enc(shared_key: bytes, plaintext: bytes) -> bytes:
        """Encrypt plaintext with authenticated encryption."""
        # Derive encryption key from shared key
        kdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'evpa-encryption',
            backend=default_backend()
        )
        key = kdf.derive(shared_key)
        
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # AES-GCM encryption
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Return IV + ciphertext + tag
        return iv + ciphertext + encryptor.tag
    
    @staticmethod
    def dec(shared_key: bytes, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext with authenticated decryption."""
        # Derive encryption key from shared key
        kdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'evpa-encryption',
            backend=default_backend()
        )
        key = kdf.derive(shared_key)
        
        # Extract IV, ciphertext, and tag
        iv = ciphertext[:16]
        tag = ciphertext[-16:]
        ct = ciphertext[16:-16]
        
        # AES-GCM decryption
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ct) + decryptor.finalize()
        
        return plaintext


class PseudoRandomGenerator:
    """Pseudorandom Generator for mask generation."""
    
    @staticmethod
    def generate(seed: bytes, output_shape: Tuple) -> np.ndarray:
        """Generate pseudorandom array from seed."""
        # Use seed to initialize numpy random state
        hash_seed = int(hashlib.sha256(seed).hexdigest()[:8], 16)
        rng = np.random.RandomState(hash_seed)
        return rng.normal(0, 0.1, size=output_shape)


class EVPAProtocolState:
    """Manages the state for EVPA 4-round protocol."""
    
    def __init__(self, security_parameter: int = 256):
        self.security_parameter = security_parameter
        self.pp = secrets.token_bytes(32)  # Public parameters
        
        # Protocol state
        self.round_number = 0
        self.user_keys = {}  # {user_id: {1: (pk, sk), 2: (pk, sk), 3: (pk, sk)}}
        self.aux_keys = {}   # {aux_id: {1: (pk, sk), 2: (pk, sk), 3: (pk, sk)}}
        self.ciphertexts = {}  # {(user_id, aux_id): ciphertext}
        self.masked_gradients = {}  # {user_id: masked_gradient}
        self.verification_macs = {}  # {user_id: mac}
        self.aux_mask_contributions = {}  # {aux_id: mask_contribution}
        self.online_users = set()
        
        # Verification values
        self.universal_alpha = 0.0
        self.universal_K = 0.0
        self.aggregated_result = None
        self.aggregated_mac = 0.0
    
    def reset_round(self):
        """Reset state for new federated learning round."""
        self.round_number = 0
        self.user_keys.clear()
        self.aux_keys.clear()
        self.ciphertexts.clear()
        self.masked_gradients.clear()
        self.verification_macs.clear()
        self.aux_mask_contributions.clear()
        self.online_users.clear()
        self.universal_alpha = 0.0
        self.universal_K = 0.0
        self.aggregated_result = None
        self.aggregated_mac = 0.0


def encode_alpha_K(alpha: float, K: float) -> bytes:
    """Encode alpha and K values for encryption."""
    alpha_bytes = str(alpha).encode('utf-8')
    K_bytes = str(K).encode('utf-8')
    return alpha_bytes + b'||' + K_bytes


def decode_alpha_K(data: bytes) -> Tuple[float, float]:
    """Decode alpha and K values from decrypted data."""
    alpha_str, K_str = data.decode('utf-8').split('||')
    return float(alpha_str), float(K_str)

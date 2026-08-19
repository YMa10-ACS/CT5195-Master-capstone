#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1] / "distributed_vlm"
sys.path.insert(0, str(APP_DIR))

from attestation import (
    AttestationError,
    create_challenge,
    create_evidence,
    encode_challenge,
    parse_challenge,
    verify_evidence,
)


class AttestationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nonce = create_challenge()
        cls.challenge = encode_challenge(cls.nonce)
        cls.evidence = create_evidence(cls.challenge)

    def test_real_challenge_evidence_and_verification_flow(self):
        self.assertEqual(parse_challenge(self.challenge), self.nonce)
        self.assertEqual(self.evidence["version"], 1)
        self.assertTrue(verify_evidence(self.evidence, self.nonce))

    def test_challenge_rejects_wrong_nonce_length(self):
        challenge = encode_challenge(create_challenge())
        challenge["nonce"] = "MYQMYQMYQMYQMYQMYQMYQMYQMYQMYQMYQMYQMYQMYQMYQMYQMYQ"  # 32 bytes instead of 64
        with self.assertRaisesRegex(AttestationError, "exactly 64 bytes"):
            parse_challenge(challenge)

    def test_real_evidence_rejects_wrong_nonce(self):
        wrong_nonce = create_challenge()
        self.assertNotEqual(wrong_nonce, self.nonce)

        with self.assertRaisesRegex(AttestationError, "verification failed"):
            verify_evidence(self.evidence, wrong_nonce)


if __name__ == "__main__":
    unittest.main()

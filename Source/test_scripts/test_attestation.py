import sys
import unittest
import base64
import json
from pathlib import Path
from unittest.mock import Mock, patch


APP_DIR = Path(__file__).resolve().parents[1] / "distributed_vlm"
sys.path.insert(0, str(APP_DIR))

from attestation import (
    AttestationError,
    create_challenge,
    create_evidence,
    encode_challenge,
    parse_challenge,
    generate_tdx_quote,
    verify_tdx_quote,
    verify_evidence,
)


class AttestationTests(unittest.TestCase):
    @patch("attestation.secrets.token_bytes", return_value=b"n" * 64)
    def test_challenge_evidence_and_verification_flow(self, token_bytes):
        expected_nonce = create_challenge()
        challenge = encode_challenge(expected_nonce)
        quote_provider = Mock(return_value=b"tdx-quote")
        evidence = create_evidence(challenge, quote_provider)
        quote_verifier = Mock(return_value=True)

        self.assertTrue(
            verify_evidence(evidence, expected_nonce, quote_verifier)
        )
        token_bytes.assert_called_once_with(64)
        quote_provider.assert_called_once_with(expected_nonce)
        quote_verifier.assert_called_once_with(b"tdx-quote", expected_nonce)

    def test_challenge_rejects_wrong_nonce_length(self):
        challenge = encode_challenge(create_challenge())
        challenge["nonce"] = "YWJj"
        with self.assertRaisesRegex(AttestationError, "exactly 64 bytes"):
            parse_challenge(challenge)

    def test_evidence_requires_real_verifier_success(self):
        nonce = create_challenge()
        challenge = encode_challenge(nonce)
        evidence = create_evidence(challenge, lambda _: b"tdx-quote")
        with self.assertRaisesRegex(AttestationError, "verification failed"):
            verify_evidence(evidence, nonce, lambda _quote, _nonce: False)

    @patch("attestation.subprocess.run")
    def test_python_quote_wrapper_uses_json_stdin(self, run):
        run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({"quote": base64.b64encode(b"quote").decode()}),
            stderr="",
        )
        nonce = b"n" * 64

        self.assertEqual(generate_tdx_quote(nonce), b"quote")

        request = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(base64.b64decode(request["nonce"]), nonce)
        self.assertEqual(run.call_args.args[0][1], "quote")

    @patch("attestation.subprocess.run")
    def test_python_verify_wrapper_passes_quote_and_nonce(self, run):
        run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({"verified": True}),
            stderr="",
        )
        nonce = b"n" * 64

        self.assertTrue(verify_tdx_quote(b"quote", nonce))

        request = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(base64.b64decode(request["quote"]), b"quote")
        self.assertEqual(base64.b64decode(request["nonce"]), nonce)
        self.assertEqual(run.call_args.args[0][1], "verify")


if __name__ == "__main__":
    unittest.main()

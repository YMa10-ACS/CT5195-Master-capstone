import json
import struct
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

APP_DIR = Path(__file__).resolve().parents[1] / "distributed_vlm"
sys.path.insert(0, str(APP_DIR))

if "psutil" not in sys.modules:
    psutil_stub = types.ModuleType("psutil")
    process = Mock()
    process.cpu_times.return_value = types.SimpleNamespace(user=0.0, system=0.0)
    psutil_stub.Process = Mock(return_value=process)
    sys.modules["psutil"] = psutil_stub

from metrics import names as metric_names
from transfer import EmbeddingTransferClient, TransferError, transfer_embedding


class FakeArray:
    shape = (1, 1, 2)
    dtype = "float32"

    def tobytes(self):
        return struct.pack("ff", 1.0, 2.0)


class FakeProjected:
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return FakeArray()


class TransferEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.projected = FakeProjected()

    def _session_with_response(self, response_body):
        response = Mock(status_code=200)
        response.json.return_value = response_body
        session = Mock()
        session.post.return_value = response
        return session

    @patch("transfer._cpu_times", side_effect=[10.0, 10.25])
    @patch("transfer.time.perf_counter", side_effect=[20.0, 20.5])
    def test_http_request_payload_metadata_results_and_metrics(
        self, _perf_counter, _cpu_times
    ):
        session = self._session_with_response({
            "result": ["first", "second"],
            "metrics": {metric_names.CLOUD_TOTAL_REQUEST: 0.3},
        })

        results, metrics = transfer_embedding(
            session, "10.0.0.1", "http", self.projected, (1, 2), "Describe", 2
        )

        session.post.assert_called_once()
        url = session.post.call_args.args[0]
        kwargs = session.post.call_args.kwargs
        metadata = json.loads(kwargs["headers"]["Visual-Meta"])
        self.assertEqual(url, "http://10.0.0.1:8000")
        self.assertNotIn("verify", kwargs)
        self.assertEqual(kwargs["data"], FakeArray().tobytes())
        self.assertEqual(metadata, {
            "shape": [1, 1, 2],
            "dtype": "float32",
            "prompt": "Describe",
            "splitted_image_ratio": [1, 2],
            "generations": 2,
        })
        self.assertEqual(results, ["first", "second"])
        self.assertEqual(metrics[metric_names.PAYLOAD_BYTES], 8)
        self.assertAlmostEqual(metrics[metric_names.TRANSFER_LATENCY], 0.5)
        self.assertAlmostEqual(metrics[metric_names.NETWORK_OVERHEAD], 0.2)
        self.assertAlmostEqual(metrics[metric_names.EDGE_TRANSFER_CPU_TOTAL], 0.25)

    def test_https_transports_use_expected_ports_and_disable_verification(self):
        cases = (
            ("https-aes-gcm", "https://cloud.example:8443"),
            ("https-chacha20", "https://cloud.example:8444"),
        )
        for transport, expected_url in cases:
            with self.subTest(transport=transport):
                session = self._session_with_response({
                    "result": [],
                    "metrics": {metric_names.CLOUD_TOTAL_REQUEST: 0.0},
                })
                transfer_embedding(
                    session, "cloud.example", transport, self.projected, (1, 1), "", 1
                )
                self.assertEqual(session.post.call_args.args[0], expected_url)
                self.assertIs(session.post.call_args.kwargs["verify"], False)

    def test_unsupported_transport_raises_value_error_before_request(self):
        session = Mock()
        with self.assertRaisesRegex(ValueError, "Unsupported transport protocol"):
            transfer_embedding(
                session, "cloud.example", "ftp", self.projected, (1, 1), "", 1
            )
        session.post.assert_not_called()

    def test_missing_cloud_metrics_raises_transfer_error(self):
        session = self._session_with_response({"result": ["result"]})
        with self.assertRaisesRegex(TransferError, "missing the metrics object"):
            transfer_embedding(
                session, "cloud.example", "http", self.projected, (1, 1), "", 1
            )

    @patch("transfer.requests.Session")
    def test_client_reuses_and_closes_its_session(self, session_factory):
        session = self._session_with_response({
            "result": [],
            "metrics": {metric_names.CLOUD_TOTAL_REQUEST: 0.0},
        })
        session_factory.return_value = session

        with EmbeddingTransferClient("cloud.example", "http") as client:
            client.attested = True
            client.transfer(self.projected, (1, 1), "", 1)
            client.transfer(self.projected, (1, 1), "", 1)

        session_factory.assert_called_once_with()
        self.assertEqual(session.post.call_count, 2)
        session.close.assert_called_once_with()

    @patch("transfer.verify_evidence", return_value=True)
    @patch("transfer.create_challenge", return_value=b"n" * 64)
    def test_client_attests_with_json_on_reused_session(
        self, create_nonce, verify_evidence
    ):
        response = Mock(status_code=200)
        response.json.return_value = {
            "version": 1,
            "quote": "dGR4LXF1b3Rl",
        }
        session = Mock()
        session.post.return_value = response

        with patch("transfer.requests.Session", return_value=session):
            with EmbeddingTransferClient("cloud.example", "https-aes-gcm") as client:
                self.assertTrue(client.attest())
                self.assertTrue(client.attested)

        create_nonce.assert_called_once_with()
        session.post.assert_called_once()
        self.assertEqual(
            session.post.call_args.args[0],
            "https://cloud.example:8443/attestation",
        )
        self.assertEqual(session.post.call_args.kwargs["json"]["version"], 1)
        self.assertIs(session.post.call_args.kwargs["verify"], False)
        verify_evidence.assert_called_once_with(response.json.return_value, b"n" * 64)

    @patch("transfer.requests.Session")
    def test_client_rejects_transfer_before_attestation(self, _session_factory):
        with EmbeddingTransferClient("cloud.example", "http") as client:
            with self.assertRaisesRegex(TransferError, "must succeed"):
                client.transfer(self.projected, (1, 1), "", 1)


if __name__ == "__main__":
    unittest.main()

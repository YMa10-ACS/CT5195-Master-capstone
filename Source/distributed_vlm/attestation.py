#!/usr/bin/env python
'''
Description:
Date: 2026-08-10 17:15:13
Author: Yaoquan Ma
'''

"""Shared remote-attestation protocol helpers for the edge and cloud."""

import base64
import binascii
import json
import os
import secrets
import subprocess
import logging

logger = logging.getLogger(__name__)

ATTESTATION_VERSION = 1
NONCE_SIZE = 64
TDX_HELPER_ENV = "TDX_ATTEST_HELPER"
TDX_HELPER_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tdx_attest", "tdx-attest"
)


class AttestationError(RuntimeError):
    """Raised when attestation evidence is missing, invalid, or untrusted."""


def _encode_bytes(value):
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value, field_name):
    if not isinstance(value, str):
        raise AttestationError(f"Attestation {field_name} must be a Base64 string")

    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise AttestationError(
            f"Attestation {field_name} is not valid Base64"
        ) from error


def create_challenge():
    """Create a fresh 64-byte attestation nonce."""
    return secrets.token_bytes(NONCE_SIZE)


def _run_tdx_helper(command, payload):
    helper_path = os.environ.get(TDX_HELPER_ENV, TDX_HELPER_DEFAULT)

    # Print the command and payload for easier debugging and logging.
    logger.info(f"Running TDX attestation helper: {helper_path} {command}")
    logger.info(f"Sending payload: {json.dumps(payload)}")

    try:
        completed = subprocess.run(
            [helper_path, command],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise AttestationError("TDX attestation helper timed out") from error
    except OSError as error:
        raise AttestationError(
            f"Unable to run TDX attestation helper {helper_path}: {error}"
        ) from error

    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown helper error"
        raise AttestationError(f"TDX attestation helper failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise AttestationError(
            "TDX attestation helper returned invalid JSON"
        ) from error
    if not isinstance(result, dict):
        raise AttestationError("TDX attestation helper response must be a JSON object")
    return result


def generate_tdx_quote(nonce):

    """Generate a raw TDX quote whose REPORTDATA contains nonce."""
    if not isinstance(nonce, bytes) or len(nonce) != NONCE_SIZE:
        raise AttestationError(f"Attestation nonce must be exactly {NONCE_SIZE} bytes")
    
    result = _run_tdx_helper("quote", {"nonce": _encode_bytes(nonce)})
    quote = _decode_bytes(result.get("quote"), "quote")
    if not quote:
        raise AttestationError("TDX attestation helper returned an empty quote")
    return quote


def verify_tdx_quote(quote, expected_nonce):
    """Verify a raw TDX quote and its REPORTDATA binding."""
    if not isinstance(quote, bytes) or not quote:
        raise AttestationError("TDX quote must be non-empty bytes")
    
    if not isinstance(expected_nonce, bytes) or len(expected_nonce) != NONCE_SIZE:
        raise AttestationError("Expected nonce must be exactly 64 bytes")

    # Command the TDX helper to verify the quote and its binding to the expected nonce.
    # For example, 
    result = _run_tdx_helper(
        "verify",
        {
            "quote": _encode_bytes(quote),
            "nonce": _encode_bytes(expected_nonce),
        },
    )
    return result.get("verified") is True


def encode_challenge(nonce):
    """Encode a nonce as a JSON-compatible challenge message."""
    if not isinstance(nonce, bytes) or len(nonce) != NONCE_SIZE:
        raise AttestationError(
            f"Attestation nonce must be exactly {NONCE_SIZE} bytes"
        )
    return {
        "version": ATTESTATION_VERSION,
        "nonce": _encode_bytes(nonce),
    }


def parse_challenge(challenge):
    """Validate a challenge message and return its 64-byte nonce."""
    if not isinstance(challenge, dict):
        raise AttestationError("Attestation challenge must be a JSON object")
    if challenge.get("version") != ATTESTATION_VERSION:
        raise AttestationError("Unsupported attestation protocol version")

    nonce = _decode_bytes(challenge.get("nonce"), "nonce")
    if len(nonce) != NONCE_SIZE:
        raise AttestationError(
            f"Attestation nonce must be exactly {NONCE_SIZE} bytes"
        )
    return nonce


def create_evidence(challenge):
    """Cloud entry point: generate TDX evidence bound to the challenge nonce."""
    nonce = parse_challenge(challenge)
    try:
        quote = generate_tdx_quote(nonce)
    except Exception as error:
        raise AttestationError(f"Unable to generate TDX quote: {error}") from error

    return {
        "version": ATTESTATION_VERSION,
        "quote": _encode_bytes(quote),
    }


def verify_evidence(evidence, expected_nonce):
    """Edge entry point: verify a TDX quote and its nonce binding."""

    # Print the expected nonce in hex format for easier debugging and logging.
    logger.info(f"Verifying attestation evidence with expected nonce {expected_nonce.hex()}")
    
    if not isinstance(expected_nonce, bytes) or len(expected_nonce) != NONCE_SIZE:
        raise AttestationError("Expected nonce must be exactly 64 bytes")

    if not isinstance(evidence, dict):
        raise AttestationError("Attestation evidence must be a JSON object")

    if evidence.get("version") != ATTESTATION_VERSION:
        raise AttestationError("Unsupported attestation protocol version")

    quote = _decode_bytes(evidence.get("quote"), "quote")
    if not quote:
        raise AttestationError("Attestation quote is empty")

    try:
        verified = verify_tdx_quote(quote, expected_nonce)
    except Exception as error:
        raise AttestationError(f"TDX quote verification failed: {error}") from error

    # Print the verification result for easier debugging and logging.
    if verified is not True:
        logger.error(f"Attestation verification failed for {expected_nonce.hex()}")
        raise AttestationError("TDX quote verification failed")
    else:
        logger.info(f"Attestation verification succeeded for {expected_nonce.hex()}")

    return True

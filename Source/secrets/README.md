# TLS Secrets

This directory contains locally generated TLS material:

- key.pem: The server's 2048-bit RSA private key. It is used to prove the server's identity and establish TLS connections.

- cert.pem: The public X.509 certificate paired with `key.pem`. It is self-signed, uses an RSA/SHA-256 signature, and is valid for 365 days. It allows the edge client to authenticate the HTTPS server. The certificate has no Subject Alternative Name (SAN), so it is intended only for controlled experiments.

- cert_generate.sh: An OpenSSL helper that creates both files using `openssl req -x509 -newkey rsa:2048`. It prompts interactively for certificate subject information and writes the generated files to the current working directory.

These files are used only in a controlled experiment environment.

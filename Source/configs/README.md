# Configuration

This directory stores runtime configuration.

## TLS configuration

- openssl_tls13_aes.cnf: restricts the TLS 1.3 experiment to its configured AES-GCM cipher suite.
- openssl_tls13_chacha20.cnf: restricts the TLS 1.3 experiment to its configured ChaCha20-Poly1305 cipher suite.

These files support controlled transport comparisons. Record the selected file in each experiment manifest so that results remain reproducible.



# TDX attestation helper

This binary exposes the Go `go-tdx-guest` quote and verification APIs to the
Python application through JSON on standard input and output.

Build it on a Linux machine with Go 1.19 or newer:

```bash
cd Source/distributed_vlm/tdx_attest
go mod tidy
go build -o tdx-attest .
```

The cloud uses `tdx-attest quote` and therefore must run inside a TDX guest.
The edge uses `tdx-attest verify` and does not require `/dev/tdx_guest`.
Set `TDX_ATTEST_HELPER` if the binary is installed at another path.

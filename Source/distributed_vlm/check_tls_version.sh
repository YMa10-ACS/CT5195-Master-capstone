#!/usr/bin/env bash

HOST="${1:-127.0.0.1}"
PORT="${2:-8443}"

check_tls() {
  local version="$1"
  local output

  output="$(openssl s_client -brief -connect "${HOST}:${PORT}" \
    -servername "$HOST" "$version" </dev/null 2>&1)"

  if echo "$output" | grep -q "Protocol version:"; then
    echo "$output" | grep -E "Protocol version:|Ciphersuite:"
    return 0
  fi

  echo "Handshake failed"
  return 1
}

echo "Target: ${HOST}:${PORT}"

echo "TLS 1.3:"
if check_tls -tls1_3; then
  tls13=yes
else
  tls13=no
fi

echo
echo "TLS 1.2:"
if check_tls -tls1_2; then
  tls12=yes
else
  tls12=no
fi

echo
if [ "$tls13" = yes ] && [ "$tls12" = no ]; then
  echo "Result: TLS 1.3 only"
elif [ "$tls13" = yes ] && [ "$tls12" = yes ]; then
  echo "Result: TLS 1.3 and TLS 1.2 are both accepted"
else
  echo "Result: TLS 1.3 is not available"
fi

#!/usr/bin/env bash
set -u

# Collect hardware, OS, kernel, Python, and library information so two VMs
# can be compared before interpreting benchmark differences.

LABEL="${1:-vm}"
OUT_DIR="${2:-./env_check}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BASE="${OUT_DIR}/${LABEL}_${TS}"
TXT="${BASE}.txt"
JSON="${BASE}.json"

mkdir -p "$OUT_DIR"

section() {
  printf "\n===== %s =====\n" "$1"
}

run_cmd() {
  printf "\n$ %s\n" "$*"
  "$@" 2>&1 || true
}

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

capture() {
  "$@" 2>&1 || true
}

{
  echo "Environment check label: $LABEL"
  echo "UTC timestamp: $TS"
  echo "Hostname: $(hostname 2>/dev/null || true)"

  section "OS and kernel"
  run_cmd uname -a
  run_cmd cat /etc/os-release
  run_cmd lsb_release -a
  run_cmd hostnamectl

  section "CPU hardware"
  run_cmd lscpu
  run_cmd nproc
  run_cmd cat /proc/cpuinfo

  section "Memory"
  run_cmd free -h
  run_cmd free -b
  run_cmd cat /proc/meminfo

  section "Disk and filesystem"
  run_cmd lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINT
  run_cmd df -hT

  section "Network"
  run_cmd ip addr
  run_cmd ip route
  run_cmd ss -tan

  section "TDX / virtualization indicators"
  run_cmd systemd-detect-virt
  run_cmd virt-what
  run_cmd dmesg
  run_cmd grep -iE "tdx|sev|snp|confidential|td quote|secure arbitration|vmx|hypervisor" /proc/cpuinfo
  run_cmd grep -iE "tdx|sev|snp|confidential|hypervisor|virt" /var/log/dmesg

  section "Compiler and system libraries"
  run_cmd gcc --version
  run_cmd g++ --version
  run_cmd ldd --version
  run_cmd openssl version -a
  run_cmd curl --version
  run_cmd git --version

  section "Python runtime"
  run_cmd which python3
  run_cmd python3 --version
  run_cmd python3 -c "import sys,platform; print('executable:', sys.executable); print('version:', sys.version); print('platform:', platform.platform()); print('machine:', platform.machine())"
  run_cmd python3 -m pip --version
  run_cmd python3 -m pip freeze

  section "Project Python libraries"
  run_cmd python3 -c "mods=['torch','torchvision','numpy','PIL','requests','flask','psutil','transformers']; import importlib; [print(f'{m}: '+getattr(importlib.import_module(m), '__version__', 'version_attr_missing')) if importlib.util.find_spec(m) else print(f'{m}: not installed') for m in mods]"
  run_cmd python3 -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', getattr(torch.version, 'cuda', None)); print('mps available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()); print('num threads:', torch.get_num_threads()); print('num interop threads:', torch.get_num_interop_threads())"

  section "Important environment variables"
  run_cmd env
} | tee "$TXT" >/dev/null

OS_RELEASE="$(capture cat /etc/os-release)"
UNAME="$(capture uname -a)"
LSCPU="$(capture lscpu)"
FREE_B="$(capture free -b)"
PY_VERSION="$(capture python3 --version)"
PIP_FREEZE="$(capture python3 -m pip freeze)"
OPENSSL_VERSION="$(capture openssl version -a)"
TORCH_INFO="$(capture python3 -c "import torch; print('torch='+torch.__version__); print('cuda_available='+str(torch.cuda.is_available())); print('cuda_version='+str(getattr(torch.version, 'cuda', None))); print('num_threads='+str(torch.get_num_threads()))")"
VIRT="$(capture systemd-detect-virt)"

cat > "$JSON" <<EOF
{
  "label": $(printf "%s" "$LABEL" | json_escape),
  "utc_timestamp": $(printf "%s" "$TS" | json_escape),
  "hostname": $(hostname 2>/dev/null | json_escape),
  "uname": $(printf "%s" "$UNAME" | json_escape),
  "os_release": $(printf "%s" "$OS_RELEASE" | json_escape),
  "lscpu": $(printf "%s" "$LSCPU" | json_escape),
  "memory_free_b": $(printf "%s" "$FREE_B" | json_escape),
  "python_version": $(printf "%s" "$PY_VERSION" | json_escape),
  "pip_freeze": $(printf "%s" "$PIP_FREEZE" | json_escape),
  "openssl_version": $(printf "%s" "$OPENSSL_VERSION" | json_escape),
  "torch_info": $(printf "%s" "$TORCH_INFO" | json_escape),
  "virtualization": $(printf "%s" "$VIRT" | json_escape)
}
EOF

echo "Wrote:"
echo "  $TXT"
echo "  $JSON"
echo
echo "Run this script on both VMs, then compare:"
echo "  diff -u normal_vm_report.txt tdx_vm_report.txt"

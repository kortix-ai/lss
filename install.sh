#!/usr/bin/env bash
# LSS installer — https://github.com/kortix-ai/lss
# Usage: curl -fsSL https://raw.githubusercontent.com/kortix-ai/lss/main/install.sh | bash
set -euo pipefail

PACKAGE="local-semantic-search"
REPO="https://github.com/kortix-ai/lss"
MIN_PYTHON="3.9"

# --- Colors (auto-off if not a terminal) ---
if [ -t 1 ]; then
  R='\033[0;31m' G='\033[0;32m' Y='\033[0;33m' B='\033[0;34m' C='\033[0;36m' W='\033[1;37m' N='\033[0m'
else
  R='' G='' Y='' B='' C='' W='' N=''
fi

info()  { printf "${C}=>${N} %s\n" "$*"; }
ok()    { printf "${G}=>${N} %s\n" "$*"; }
warn()  { printf "${Y}=>${N} %s\n" "$*"; }
fail()  { printf "${R}ERROR:${N} %s\n" "$*" >&2; exit 1; }

# --- Check Python >= 3.9 ---
check_python() {
  local py=""
  for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
      local ver
      ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
      if [ -n "$ver" ]; then
        local major minor
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
          py="$cmd"
          break
        fi
      fi
    fi
  done
  if [ -z "$py" ]; then
    fail "Python >= ${MIN_PYTHON} is required but not found. Install Python first: https://python.org"
  fi
  echo "$py"
}

# --- Detect best installer ---
detect_installer() {
  if command -v pipx &>/dev/null; then
    echo "pipx"
  elif command -v uv &>/dev/null; then
    echo "uv"
  elif command -v pip3 &>/dev/null; then
    echo "pip3"
  elif command -v pip &>/dev/null; then
    echo "pip"
  else
    echo "none"
  fi
}

# --- Check if lss is already installed ---
is_installed() {
  command -v lss &>/dev/null
}

get_current_version() {
  lss --version 2>/dev/null | awk '{print $2}' || echo ""
}

# --- Main ---
main() {
  printf "\n${W}  LSS — Local Semantic Search${N}\n"
  printf "  ${C}${REPO}${N}\n\n"

  # Detect if this is an upgrade
  local upgrading=false
  local current_ver=""
  if is_installed; then
    current_ver=$(get_current_version)
    upgrading=true
    info "lss ${current_ver} is already installed — upgrading..."
  fi

  info "Checking Python..."
  local py
  py=$(check_python)
  local pyver
  pyver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
  ok "Python ${pyver} (${py})"

  info "Detecting package installer..."
  local installer
  installer=$(detect_installer)

  case "$installer" in
    pipx)
      ok "Found pipx"
      if $upgrading; then
        info "Upgrading ${PACKAGE} via pipx..."
        pipx upgrade "$PACKAGE" 2>/dev/null || pipx install "$PACKAGE" --force
      else
        info "Installing ${PACKAGE} via pipx..."
        pipx install "$PACKAGE"
      fi
      ;;
    uv)
      ok "Found uv"
      if $upgrading; then
        info "Upgrading ${PACKAGE} via uv tool..."
        uv tool upgrade "$PACKAGE" 2>/dev/null || uv tool install "$PACKAGE" --force
      else
        info "Installing ${PACKAGE} via uv tool..."
        uv tool install "$PACKAGE"
      fi
      ;;
    pip3|pip)
      warn "pipx not found — using ${installer} (consider installing pipx for isolated environments)"
      info "Installing ${PACKAGE} via ${installer}..."
      "$installer" install --upgrade "$PACKAGE"
      ;;
    none)
      warn "No pip/pipx/uv found. Installing pipx first..."
      "$py" -m pip install --user pipx 2>/dev/null || "$py" -m ensurepip --default-pip && "$py" -m pip install --user pipx
      export PATH="$HOME/.local/bin:$PATH"
      if command -v pipx &>/dev/null; then
        info "Installing ${PACKAGE} via pipx..."
        pipx install "$PACKAGE"
      else
        fail "Could not install pipx. Please install manually: ${REPO}#install"
      fi
      ;;
  esac

  # Verify
  printf "\n"
  if command -v lss &>/dev/null; then
    local installed_ver
    installed_ver=$(lss --version 2>/dev/null || echo "unknown")

    if $upgrading && [ -n "$current_ver" ]; then
      ok "Updated: lss ${current_ver} -> ${installed_ver##* }"
    else
      ok "Installed: ${installed_ver}"
    fi

    printf "\n"
    printf "  ${W}Get started:${N}\n"
    printf "    export OPENAI_API_KEY=\"sk-...\"   ${C}# add to ~/.zshrc or ~/.bashrc${N}\n"
    printf "    lss \"your search query\"           ${C}# search current directory${N}\n"
    printf "    lss \"query\" ~/Documents           ${C}# search a specific path${N}\n"
    printf "    lss update                        ${C}# check for updates${N}\n"
    printf "    lss --help                        ${C}# see all commands${N}\n"
    printf "\n"
  else
    warn "lss installed but not found in PATH."
    warn "You may need to add ~/.local/bin to your PATH:"
    printf "\n"
    printf "    export PATH=\"\$HOME/.local/bin:\$PATH\"\n"
    printf "\n"
  fi
}

main

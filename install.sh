#!/bin/sh
# Install mmlmRAG from an unpacked release archive.
# Requires macOS and python3.13. Installs the rag command for this user.
set -eu

if [ "$(uname -s)" != "Darwin" ]; then
  echo "mmlmRAG installs on macOS only." >&2
  exit 1
fi

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "python3.13 is required. Install it, then run this script again." >&2
  exit 1
fi

SOURCE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
HOME_DIR=${HOME:?}
INSTALL_ROOT="$HOME_DIR/.mmlmrag"
VENV="$INSTALL_ROOT/venv"
BIN_DIR="$HOME_DIR/.local/bin"
CONFIG="$INSTALL_ROOT/rag.toml"

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"
if [ ! -f "$CONFIG" ]; then
  cp "$SOURCE/rag.toml" "$CONFIG"
fi

python3.13 -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip
"$VENV/bin/python" -m pip install "$SOURCE"

cat > "$BIN_DIR/rag" << EOF
#!/bin/sh
set -eu
VENV="$VENV"
CONFIG="$CONFIG"
for arg in "\$@"; do
  case "\$arg" in
    --config|--config=*)
      exec "\$VENV/bin/rag" "\$@"
      ;;
  esac
done
dir=\$(pwd)
while :; do
  if [ -f "\$dir/rag.toml" ]; then
    exec "\$VENV/bin/rag" "\$@"
  fi
  [ "\$dir" = "/" ] && break
  dir=\$(dirname "\$dir")
done
exec "\$VENV/bin/rag" --config "\$CONFIG" "\$@"
EOF
chmod 755 "$BIN_DIR/rag"

echo "Installed rag to $BIN_DIR/rag"
echo "Default config: $CONFIG"
echo "A directory that contains rag.toml uses that file. Any other directory uses the default config."
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo "Add this to your shell startup file:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    ;;
esac

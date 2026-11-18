#!/bin/bash
# Sets up the MuJoCo C library in third_party/mujoco (include/, lib/, sample/).
#
# OUR CODE (whole file). It downloads Google DeepMind's official MuJoCo release
# (Apache 2.0): https://github.com/google-deepmind/mujoco/releases
# The download URL follows the release page's naming. The --pip option instead
# reuses the library shipped inside the "mujoco" Python package
# (https://pypi.org/project/mujoco/).
#
#   scripts/get_mujoco.sh            download the official release
#   scripts/get_mujoco.sh --pip      use a pip-installed "mujoco" package
#
# Set MUJOCO_VERSION to choose a release (default 3.3.7).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=third_party/mujoco
VERSION=${MUJOCO_VERSION:-3.3.7}
rm -rf "$DEST"
mkdir -p third_party

if [[ "${1:-}" == "--pip" ]]; then
    PKG=$(python3 -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")
    LIB=$(ls "$PKG"/libmujoco.so.* | head -n 1)
    mkdir -p "$DEST/lib"
    ln -s "$PKG/include" "$DEST/include"
    ln -s "$LIB" "$DEST/lib/$(basename "$LIB")"
    ln -s "$(basename "$LIB")" "$DEST/lib/libmujoco.so"
    echo "Linked MuJoCo from pip package: $PKG"
else
    URL="https://github.com/google-deepmind/mujoco/releases/download/${VERSION}/mujoco-${VERSION}-linux-x86_64.tar.gz"
    echo "Downloading $URL"
    curl -fL "$URL" -o third_party/mujoco.tar.gz
    tar -xzf third_party/mujoco.tar.gz -C third_party
    mv "third_party/mujoco-${VERSION}" "$DEST"
    rm third_party/mujoco.tar.gz
    echo "Installed MuJoCo ${VERSION} in $DEST"
fi
ls "$DEST/include/mujoco/mujoco.h" "$DEST/lib/libmujoco.so" >/dev/null && echo "OK"

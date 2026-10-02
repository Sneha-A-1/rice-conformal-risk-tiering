#!/usr/bin/env bash
# ImageNet-pretrained MobileNetV3-Large weights (timm, GitHub release). SHA-256 checked.
set -euo pipefail
mkdir -p ../weights && cd ../weights
URL=https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth
curl -sSL -O "$URL"
echo "f55367f56f62fa6115a03033835fd60ef3094a1e0fdce6f38a5c97cbab27295f  mobilenetv3_large_100_ra-f55367f5.pth" | sha256sum -c -

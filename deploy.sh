#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

npm install
npm --workspace @aetherforecast/infra run build
npm --workspace @aetherforecast/infra run bootstrap
npm --workspace @aetherforecast/infra run deploy

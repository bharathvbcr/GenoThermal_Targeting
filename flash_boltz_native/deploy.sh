#!/usr/bin/env bash
# Deploy the Boltz-2 fold endpoint on RunPod Flash WITHOUT Docker.
#
# WHY NO DEPS AT BUILD TIME: Flash's build-time pip resolver is wheels-only against a fixed
# target and can't install boltz's tree (fairscale==0.4.13 -> "from versions: none"). So we
# deploy with dependencies=[] (which builds cleanly) and install boltz AT RUNTIME on the
# worker — a normal Linux GPU box with real PyPI access — on the first fold. No image, no
# docker build, 100% Flash-native.
#
# Usage:
#   WORKERS=1 ./flash_boltz_native/deploy.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WORKERS="${WORKERS:-1}"      # A100s are pricey + quota is shared; keep small
STAGE="$HERE/.stage"

echo ">> [1/2] stage isolated deploy dir + patch endpoint (dependencies=[], runtime boltz install)"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp "$ROOT/flash_boltz.py" "$STAGE/flash_boltz.py"
python3 - "$STAGE/flash_boltz.py" "$WORKERS" <<'PY'
import re, sys
path, workers = sys.argv[1], int(sys.argv[2])
src = open(path).read()

# 1) No build-time deps — boltz is installed on the worker at runtime instead.
src = src.replace('dependencies=["boltz", "pynvml"],',
                  'dependencies=[],   # Docker-free: boltz installed at runtime on the worker')
# 2) Smaller worker ceiling (quota is shared account-wide).
src = re.sub(r'workers=\(0, 20\),', f'workers=(0, {workers}),', src)

# 3) Inject a runtime bootstrap and call it before the first fold (worker-side only).
bootstrap = '''
def _ensure_boltz():
    """Docker-free: install boltz on the Flash worker at runtime. Flash's BUILD pip can't
    resolve boltz, but the WORKER has normal PyPI access. Cached on the warm worker; point
    BOLTZ_CACHE at a mounted network volume to persist weights across cold starts."""
    import shutil, subprocess, sys as _sys
    if shutil.which("boltz"):
        return
    subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "boltz==2.1.1", "pynvml"],
                   check=True)

'''
# Define _ensure_boltz just above the Flash-endpoint block.
src = src.replace("# --- Flash endpoint", bootstrap + "# --- Flash endpoint", 1)
# Call it at the top of the worker handler.
src = src.replace('        """payload = {target_seq, candidate, use_msa_server?}"""\n',
                  '        """payload = {target_seq, candidate, use_msa_server?}"""\n        _ensure_boltz()\n')
open(path, "w").write(src)
print("patched:", path)
PY

echo ">> [2/2] deploy (no docker; dependencies=[] builds cleanly)"
cd "$STAGE"
RUNPOD_API_KEY="$(grep '^RUNPOD_API_KEY=' "$ROOT/.env" | cut -d= -f2- | tr -d '[:space:]')" \
  "$ROOT/.venv-flash/bin/flash" deploy --app Geno-Thermal_Targeting --env production

echo ">> done. First fold cold-starts (installs boltz + downloads weights, a few min); then warm."
echo ">> Fold for real:  cd $ROOT && GENOTHERMAL_FLASH=1 .venv-flash/bin/python boltz_designer.py --output_csv outputs/reports/candidate_library_flash.csv"

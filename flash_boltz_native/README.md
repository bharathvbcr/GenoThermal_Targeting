# Boltz-2 folding on RunPod Flash — Docker-free

The hackathon forbids Docker, so the prebuilt-image route is out. This is the
**Flash-native** way to run real Boltz-2 GPU folding.

## The trick
Flash's **build-time** pip resolver is wheels-only against a fixed target and cannot
install boltz's tree:
```
ERROR: No matching distribution found for fairscale==0.4.13   (from versions: none)
```
(`--python-version 3.11` does not help — verified.)

But the Flash **worker** is an ordinary Linux GPU box with normal PyPI access. So we:
1. Deploy the endpoint with `dependencies=[]` — this builds cleanly (same as the
   `genothermal-fitness` endpoint, which deploys and runs today).
2. Install boltz **at runtime on the worker**, on the first fold, via a tiny
   `_ensure_boltz()` bootstrap. No Docker, no image, no registry.

`deploy.sh` stages an isolated copy of `flash_boltz.py`, applies exactly that patch, and
deploys it.

## Run it
```bash
# 1. Make sure there's worker-quota headroom for one A100 (quota is account-wide).
#    e.g. lower genothermal-fitness to (0,9) first, or raise the RunPod quota.
WORKERS=1 ./flash_boltz_native/deploy.sh

# 2. Fold for real on Flash GPUs (same candidate_library schema the rest of the pipeline reads):
GENOTHERMAL_FLASH=1 .venv-flash/bin/python boltz_designer.py \
    --output_csv outputs/reports/candidate_library_flash.csv
```

## Cold-start cost & the volume optimization
The first fold on a fresh worker installs boltz (~1–2 min) and downloads Boltz-2 weights
(~a few GB). `idle_timeout=120` keeps a warm worker across a candidate burst, so only the
first request pays it.

To persist weights across cold starts (Docker-free), attach a RunPod **network volume** and
set `BOLTZ_CACHE` to its mount path via the endpoint's `env=` — then weights download once
and every future cold worker reuses them. (`NetworkVolume` is supported by the `@Endpoint`
decorator's `volume=` argument.)

## Why this is the right answer for the hackathon
- Zero Docker. Only `flash deploy` + a runtime pip install on the worker.
- Reuses 100% of `flash_boltz.py`'s `fold_complex` (boltz CLI, MSA server, ipTM + affinity
  parsing). The only change is *when* boltz arrives on the worker.

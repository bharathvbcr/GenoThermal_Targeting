# Geno-Thermal × RunPod Flash — demo runbook.
# Use the interpreter that has your deps:  make PYTHON=~/.venvs/ml/bin/python preflight
PYTHON ?= python

.PHONY: help preflight install demo demo-local monitor pipeline panel screen sweep dashboard snapshot replay story board intel clean claude-science claude-science-flash mcp-selftest

# The Claude Science MCP server uses the env that has every dep (numpy/alphagenome/runpod_flash/mcp).
SCIENCE_PYTHON ?= .venv-flash/bin/python

help:                ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:             ## Install driver-side deps (Flash SDK, pandas, matplotlib)
	$(PYTHON) -m pip install -r requirements-flash.txt

preflight:           ## Run the 12 local sanity checks (do this FIRST at the venue)
	$(PYTHON) preflight.py

demo:                ## One-shot judge demo on Flash (smoke + flash + keep-going + dashboard)
	$(PYTHON) run_pipeline.py --demo
	-@command -v open >/dev/null 2>&1 && [ -f outputs/figures/flash_scaling.png ] && open outputs/figures/flash_scaling.png || true

demo-local:          ## Same demo but fully local (no Flash SDK needed)
	$(PYTHON) run_pipeline.py --smoke --keep-going

monitor:             ## Live demo with the browser progress pop-up (smoke + flash + keep-going + monitor)
	$(PYTHON) run_pipeline.py --smoke --flash --keep-going --monitor

pipeline:            ## Full pipeline on Flash (real workload, not smoke)
	$(PYTHON) run_pipeline.py --flash

panel:               ## Multi-oncogene selectivity matrix (targets × candidates)
	$(PYTHON) target_panel.py --intended EGFR

screen:              ## Small-molecule virtual screen (Boltz-2 affinity head)
	$(PYTHON) boltz_designer.py --candidates_file data/sample_data/small_molecule_candidates.csv

board:               ## Unified peptide + small-molecule design leaderboard
	$(PYTHON) leaderboard.py

intel:               ## Live target intel via Bright Data, fanned out on Flash (stub without a token)
	GENOTHERMAL_FLASH=1 $(PYTHON) bright_data_intel.py

sweep:               ## PPO seed sweep — fan 8 seeds, keep the best design
	GENOTHERMAL_FLASH=1 $(PYTHON) flash_gpu_jobs.py ppo --sweep 8

dashboard:           ## Render outputs/figures/flash_scaling.png from recorded metrics
	$(PYTHON) flash_dashboard.py

snapshot:            ## Build the illustrative outputs/reports/demo_metrics.json + outputs/figures/flash_scaling.png fallback
	$(PYTHON) make_demo_snapshot.py

replay:              ## Re-render the fallback chart from outputs/reports/demo_metrics.json (if a live call stalls)
	$(PYTHON) flash_dashboard.py --metrics outputs/reports/demo_metrics.json --out outputs/figures/flash_scaling.png

story:               ## 3-min demo path ONLY: fresh GA fan-out -> selectivity panel -> dashboard -> summary
	rm -f flash_metrics.json   # fresh metrics for this run (the demo fallback lives in outputs/reports/demo_metrics.json)
	@echo ">> [1/4] GA fan-out — the headline (0 -> N workers; GENOTHERMAL_LIVE shows the live ticker)"
	GENOTHERMAL_FLASH=1 GENOTHERMAL_LIVE=1 $(PYTHON) hard_mode/evolver.py || echo ">> GA step errored — continuing the story"
	@echo ">> [2/4] Selectivity panel — novelty (EGFR/KRAS/HER2/BRAF matrix + heatmap)"
	GENOTHERMAL_FLASH=1 $(PYTHON) target_panel.py --intended EGFR || echo ">> panel step errored — continuing the story"
	@echo ">> [3/4] Dashboard — autoscaling + cost + reliability"
	$(PYTHON) flash_dashboard.py || { echo ">> no live metrics — rendering the recorded fallback (outputs/reports/demo_metrics.json)"; $(PYTHON) flash_dashboard.py --metrics outputs/reports/demo_metrics.json; }
	@echo ">> [4/4] Closing terminal summary"
	$(PYTHON) summary_report.py
	@echo ">> story done — chart: outputs/figures/flash_scaling.png | heatmap: outputs/figures/panel_selectivity_heatmap.png"

mcp-selftest:        ## Exercise every Claude Science MCP tool in-process (no client needed)
	$(SCIENCE_PYTHON) mcp_geno_thermal.py --selftest

claude-science:      ## Run the full discover->design->verify loop locally (Claude Science tools)
	$(SCIENCE_PYTHON) -c "import json,mcp_geno_thermal as m; print(json.dumps(m.screen_and_verify(target_gene='EGFR'), indent=2))"

claude-science-flash: ## Same loop; GA fitness fans out on RunPod Flash GPUs (needs RUNPOD_API_KEY)
	@echo ">> GA fitness scoring fans out on the deployed 'genothermal-fitness' Flash endpoint (0->N->0)."
	@echo ">> Boltz-2 *folding* on Flash uses the Docker-free runtime-install endpoint (boltz can't"
	@echo ">> pip-install in Flash's build sandbox, so it installs on the worker at first fold)."
	@echo ">> Deploy it once (needs RunPod worker-quota headroom):  WORKERS=1 ./flash_boltz_native/deploy.sh"
	@echo ">> (see flash_boltz_native/README.md). Until then, folding uses the committed library + BioNeMo verify."
	GENOTHERMAL_FLASH=1 $(SCIENCE_PYTHON) -c "import json,mcp_geno_thermal as m; print(json.dumps(m.screen_and_verify(target_gene='EGFR', use_flash=True), indent=2))"

clean:               ## Remove generated (untracked) artifacts — leaves git-tracked files alone
	rm -f flash_metrics.json outputs/figures/flash_scaling.png _preflight*.json _preflight*.png \
	      *.log outputs/reports/panel_selectivity*.csv outputs/reports/panel_selectivity*_matrix.csv

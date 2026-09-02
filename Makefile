.PHONY: install install-ui api ui test verify clean reproduce generate overlap features train evaluate graph rings spikes fusion simulate seed

install:
	pip install -r requirements.txt

install-ui:
	cd frontend && npm install

api:
	uvicorn backend.app.main:app --reload --port 8000

# Run in a second terminal alongside `make api`.
ui:
	cd frontend && npm run dev

test:
	pytest

generate:
	python -m ml.generate --seed 42

overlap:
	python -m ml.inspect_overlap

features:
	python -m ml.features

train:
	python -m ml.train --seed 42

evaluate:
	python -m ml.evaluate

graph:
	python -m ml.inspect_graph

rings:
	python -m ml.rings

spikes:
	python -m ml.spike
	python -m ml.inspect_spike

fusion:
	python -m ml.fusion

simulate:
	python -m ml.simulate

seed:
	python -m backend.app.core.seed

verify:
	bash scripts/verify_phase1.sh
	bash scripts/verify_phase2.sh
	bash scripts/verify_phase3.sh
	bash scripts/verify_phase4.sh
	bash scripts/verify_phase5.sh
	bash scripts/verify_phase6.sh
	bash scripts/verify_phase7.sh
	bash scripts/verify_phase8.sh
	bash scripts/verify_phase9.sh
	bash scripts/verify_phase10.sh
	bash scripts/verify_phase11.sh
	bash scripts/verify_phase12.sh
	bash scripts/verify_phase13.sh
	bash scripts/verify_phase14.sh
	bash scripts/verify_phase15.sh

# The single command a judge can run to regenerate every number we claim.
reproduce: generate overlap features train evaluate graph rings spikes fusion simulate seed test
	@echo ""
	@echo "Regenerated from seed 42. Nothing above was hardcoded."

clean:
	rm -rf artifacts/*.json artifacts/*.pkl artifacts/*.joblib data/events data/labels data/features.parquet razorshield.db .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

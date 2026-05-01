PYTHON ?= python3
DEEPACCIDENT_ROOT ?= /home/elicer/deepaccident_mini_dataset
PLY_OUT ?= outputs/town03_4dashcam_collision_3dgs_45000.ply

.PHONY: serve rebuild-viewer rebuild-ply verify-ply check

serve:
	$(PYTHON) scripts/serve_viewer.py --port 8132

rebuild-viewer:
	$(PYTHON) scripts/build_four_vehicle_collision_viewer.py \
		--dataset "$(DEEPACCIDENT_ROOT)" \
		--output-root .

rebuild-ply:
	mkdir -p outputs
	$(PYTHON) scripts/build_town03_clean_hybrid_gaussian_ply.py \
		--dataset "$(DEEPACCIDENT_ROOT)" \
		--out "$(PLY_OUT)" \
		--frame-start 1 \
		--frame-end 56 \
		--static-limit 1100000 \
		--vehicle-limit-each 120000 \
		--static-voxel 0.050 \
		--vehicle-voxel 0.025 \
		--stats "$(PLY_OUT:.ply=.stats.json)"

verify-ply:
	$(PYTHON) scripts/verify_ply.py "$(PLY_OUT)"

check:
	$(PYTHON) -m py_compile scripts/*.py

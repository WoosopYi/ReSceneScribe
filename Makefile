PYTHON ?= python3
DEEPACCIDENT_ROOT ?= $(CURDIR)/deepaccident_mini_dataset
CAMERA_CATEGORY ?= type1_subtype2_accident
CAMERA_SCENARIO ?= Town04_type001_subtype0002_scenario00017
SLAM3R_ROOT ?= third_party/SLAM3R

CAMERA_EXPORT ?= outputs/town04_type1_subtype2_multicam_export
CAMERA_BASE ?= outputs/town04_type1_subtype2_slam3r_reconstruction
CAMERA_LAYERS ?= outputs/town04_type1_subtype2_slam3r_incremental_layers

LEGACY_CATEGORY ?= type1_subtype1_accident
LEGACY_SCENARIO ?= Town03_type001_subtype0001_scenario00024
LEGACY_PLY_OUT ?= outputs/town03_4dashcam_collision_3dgs_45000.ply

.PHONY: serve serve-camera camera-export camera-slam3r camera-layers camera-final \
	legacy-rebuild-viewer legacy-rebuild-ply verify-legacy-ply check

serve:
	$(PYTHON) scripts/serve_viewer.py --port 8132

serve-camera: serve

camera-export:
	$(PYTHON) scripts/run_multicam_world_reconstruction.py \
		--dataset "$(DEEPACCIDENT_ROOT)" \
		--category "$(CAMERA_CATEGORY)" \
		--scenario "$(CAMERA_SCENARIO)" \
		--out "$(CAMERA_EXPORT)" \
		--frame-start 1 \
		--frame-end 49 \
		--frame-step 1

camera-slam3r:
	$(PYTHON) scripts/run_slam3r_deepaccident_reconstruction.py \
		--dataset "$(DEEPACCIDENT_ROOT)" \
		--category "$(CAMERA_CATEGORY)" \
		--scenario "$(CAMERA_SCENARIO)" \
		--mask-export "$(CAMERA_EXPORT)" \
		--out "$(CAMERA_BASE)" \
		--slam3r-root "$(SLAM3R_ROOT)"

camera-layers:
	$(PYTHON) scripts/build_slam3r_incremental_layers.py \
		--dataset "$(DEEPACCIDENT_ROOT)" \
		--source "$(CAMERA_BASE)" \
		--mask-export "$(CAMERA_EXPORT)" \
		--out "$(CAMERA_LAYERS)" \
		--slam3r-root "$(SLAM3R_ROOT)"

camera-final:
	$(PYTHON) scripts/run_town04_camera_only_final_pipeline.py \
		--dataset "$(DEEPACCIDENT_ROOT)" \
		--out outputs/camera_only_reconstruction_town04_final \
		--calibrated-output "$(CAMERA_EXPORT)" \
		--scenario "$(CAMERA_SCENARIO)" \
		--category "$(CAMERA_CATEGORY)" \
		--vggt-root third_party/vggt

legacy-rebuild-viewer:
	$(PYTHON) scripts/build_four_vehicle_collision_viewer.py \
		--dataset "$(DEEPACCIDENT_ROOT)" \
		--category "$(LEGACY_CATEGORY)" \
		--scenario "$(LEGACY_SCENARIO)" \
		--output-root .

legacy-rebuild-ply:
	mkdir -p outputs
	$(PYTHON) scripts/build_town03_clean_hybrid_gaussian_ply.py \
		--dataset "$(DEEPACCIDENT_ROOT)" \
		--category "$(LEGACY_CATEGORY)" \
		--scenario "$(LEGACY_SCENARIO)" \
		--out "$(LEGACY_PLY_OUT)" \
		--frame-start 1 \
		--frame-end 56 \
		--static-limit 1100000 \
		--vehicle-limit-each 120000 \
		--static-voxel 0.050 \
		--vehicle-voxel 0.025 \
		--stats "$(LEGACY_PLY_OUT:.ply=.stats.json)"

verify-legacy-ply:
	$(PYTHON) scripts/verify_ply.py "$(LEGACY_PLY_OUT)"

check:
	$(PYTHON) -m py_compile scripts/*.py research_camera_only/scripts/*.py
	$(PYTHON) -m unittest discover -s tests

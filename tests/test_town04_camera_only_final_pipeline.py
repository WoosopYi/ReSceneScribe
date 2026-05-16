from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_town04_camera_only_final_pipeline as final_pipeline  # noqa: E402
import run_multicamera_vggt_colmap_backend as vggt_backend  # noqa: E402


class Town04FinalPipelineTests(unittest.TestCase):
    def test_parse_frame_list_accepts_ranges_and_commas(self) -> None:
        self.assertEqual(final_pipeline.parse_frame_list("1,3-5,8..10", 2, 9), (3, 4, 5, 8, 9))

    def test_cross_agent_discovery_finds_facing_collision_views(self) -> None:
        views = []
        for frame in (45, 46, 47):
            c2w_a = np.eye(4, dtype=np.float64)
            c2w_a[:3, 3] = [0.0, 0.0, 0.0]
            c2w_a[:3, 2] = [1.0, 0.0, 0.0]
            c2w_b = np.eye(4, dtype=np.float64)
            c2w_b[:3, 3] = [10.0, 0.0, 0.0]
            c2w_b[:3, 2] = [-1.0, 0.0, 0.0]
            views.append(
                vggt_backend.ViewRecord(
                    view_id=f"a:{frame}",
                    agent="agent_a",
                    camera="Camera_Front",
                    frame=frame,
                    rgb_path="a.jpg",
                    mask_path="a.png",
                    c2w=c2w_a,
                    intrinsic_raw=np.eye(3),
                )
            )
            views.append(
                vggt_backend.ViewRecord(
                    view_id=f"b:{frame}",
                    agent="agent_b",
                    camera="Camera_Back",
                    frame=frame,
                    rgb_path="b.jpg",
                    mask_path="b.png",
                    c2w=c2w_b,
                    intrinsic_raw=np.eye(3),
                )
            )

        shards = final_pipeline.discover_cross_agent_shards(views, (45, 46, 47), max_cross_shards=2)

        self.assertEqual(len(shards), 1)
        self.assertEqual(shards[0].kind, "cross_agent_collision_overlap")
        self.assertEqual(shards[0].frames, (45, 46, 47))

    def test_fit_similarity_umeyama_recovers_known_transform(self) -> None:
        src = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.2, 1.0],
                [-0.4, 0.3, 0.6],
            ],
            dtype=np.float64,
        )
        theta = 0.37
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        scale = 2.4
        trans = np.array([3.0, -1.5, 0.75], dtype=np.float64)
        dst = (scale * (rot @ src.T)).T + trans

        got_scale, got_rot, got_trans, report = final_pipeline.fit_similarity_umeyama(src, dst)

        self.assertAlmostEqual(got_scale, scale, places=8)
        np.testing.assert_allclose(got_rot, rot, atol=1e-8)
        np.testing.assert_allclose(got_trans, trans, atol=1e-8)
        self.assertLess(report["rmse_m"], 1e-8)

    def test_voxel_downsample_averages_colors_and_preserves_counts(self) -> None:
        points = np.array([[0.01, 0.01, 0.0], [0.02, 0.03, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        colors = np.array([[10, 20, 30], [20, 30, 40], [100, 110, 120]], dtype=np.uint8)
        source_ids = np.array([1, 1, 2], dtype=np.int32)

        out_points, out_colors, out_sources, report = final_pipeline.voxel_downsample_with_metadata(
            points, colors, source_ids, voxel=0.1, max_points=0, seed=3
        )

        self.assertEqual(report["before"], 3)
        self.assertEqual(report["after_voxel"], 2)
        self.assertEqual(report["after_limit"], 2)
        self.assertEqual(len(out_points), 2)
        self.assertTrue(any(np.array_equal(color, np.array([15, 25, 35], dtype=np.uint8)) for color in out_colors))
        self.assertIn(1, out_sources.tolist())
        self.assertIn(2, out_sources.tolist())

    def test_ply_roundtrip_and_bounds_are_finite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "points.ply"
            points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
            colors = np.array([[1, 2, 3], [250, 251, 252]], dtype=np.uint8)
            final_pipeline.write_ply(path, points, colors)

            got_points, got_colors = final_pipeline.read_ply_points(path)
            bounds = final_pipeline.point_bounds(got_points)

            np.testing.assert_allclose(got_points, points, atol=1e-6)
            np.testing.assert_array_equal(got_colors, colors)
            self.assertTrue(bounds["finite"])
            self.assertEqual(bounds["point_count"], 2)
            self.assertEqual(bounds["bbox_span"], [3.0, 3.0, 3.0])

    def test_quality_gate_accepts_good_shard_and_rejects_bad_alignment(self) -> None:
        args = argparse.Namespace(
            allow_no_ba=False,
            min_registered_ratio=0.70,
            max_median_align_error_m=1.5,
            max_rmse_align_error_m=4.0,
            min_shard_points=100,
            max_shard_bbox_m=260.0,
            max_shard_z_span_m=80.0,
        )
        alignment = {
            "colmap_registered_images": 8,
            "colmap_points3D": 150,
            "residuals": {"median_error_m": 0.4, "rmse_m": 0.9},
        }
        ba_report = {"requested": True, "converged": True}
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp)
            (shard / "scene" / "sparse").mkdir(parents=True)
            (shard / "scene" / "sparse" / "points3D.bin").write_bytes(b"not empty")
            final_pipeline.write_ply(
                shard / "alignment" / "points3D_aligned_world.ply",
                np.random.default_rng(2).normal(size=(150, 3)),
                np.full((150, 3), 128, dtype=np.uint8),
            )

            quality = final_pipeline.evaluate_shard_quality(shard, 10, alignment, ba_report, args)
            self.assertTrue(quality["accepted"], quality["reasons"])

            bad = dict(alignment)
            bad["residuals"] = {"median_error_m": 3.0, "rmse_m": 3.5}
            rejected = final_pipeline.evaluate_shard_quality(shard, 10, bad, ba_report, args)
            self.assertFalse(rejected["accepted"])
            self.assertTrue(any("median_alignment_error" in reason for reason in rejected["reasons"]))

    def test_forbidden_path_audit_allows_audit_mentions_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "reports").mkdir()
            (out / "reconstruction").mkdir()
            (out / "manifest.json").write_text('{"lidar_used": false}', encoding="utf-8")
            (out / "reports" / "quality_report.md").write_text("quality ok", encoding="utf-8")
            (out / "reports" / "no_lidar_audit.md").write_text("lidar01 not read; viewer_assets not read", encoding="utf-8")
            (out / "reconstruction" / "fused_sparse_world_metadata.json").write_text("{}", encoding="utf-8")
            self.assertEqual(final_pipeline.audit_forbidden_paths(out)["status"], "pass")

            (out / "reports" / "quality_report.md").write_text("used viewer_assets/foo.glb", encoding="utf-8")
            self.assertEqual(final_pipeline.audit_forbidden_paths(out)["status"], "fail")


if __name__ == "__main__":
    unittest.main()

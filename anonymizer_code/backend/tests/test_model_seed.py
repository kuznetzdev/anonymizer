from __future__ import annotations

import hashlib
import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.sparse import csr_matrix

from app.model import FEATURES
from app.seed import SeedArtifactError, SeedModel, load_seed_model
from app.store import install_seed_model


def make_artifact(
    *, version: int = 2, shape: tuple[int, int] = (3, FEATURES),
) -> bytes:
    matrix = csr_matrix(shape, dtype=np.float64)
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        version=np.array([version], dtype=np.int16),
        features=np.array([FEATURES if shape[1] == FEATURES else shape[1]], dtype=np.int32),
        classes=np.array(["B-PER", "I-PER", "O"]),
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.array(matrix.shape, dtype=np.int32),
        intercept=np.zeros(3, dtype=np.float64),
    )
    return buffer.getvalue()


class SeedManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manifest_path = Path(self.temp.name) / "ru-pii-base-0.1.0.json"
        self.artifact_path = self.manifest_path.with_suffix(".npz")
        self.artifact = make_artifact()
        self.manifest: dict[str, object] = {
            "model_series": "ru-pii",
            "version": "0.1.0",
            "artifact_version": 2,
            "tokenizer_version": 2,
            "sha256": hashlib.sha256(self.artifact).hexdigest(),
            "classes": ["B-PER", "I-PER", "O"],
            "categories": ["PER"],
            "metrics": {"entity_micro_f1": 0.91},
            "training": {
                "corpus_id": "synthetic-ru-pii",
                "corpus_version": "1.9.0",
                "split_counts": {"train": 20, "validation": 5, "test": 5},
                "metrics": {"entity_micro_f1": 0.91},
            },
        }

    def write_seed(self, *, artifact: bytes | None = None) -> None:
        self.artifact_path.write_bytes(self.artifact if artifact is None else artifact)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_loads_valid_manifest_and_hash_without_corpus_files(self) -> None:
        self.write_seed()
        seed = load_seed_model(self.manifest_path)
        self.assertEqual(seed.sha256, hashlib.sha256(self.artifact).hexdigest())
        self.assertEqual(seed.classes, ("B-PER", "I-PER", "O"))
        self.assertEqual(seed.categories, ("PER",))
        self.assertEqual(seed.metrics, {"entity_micro_f1": 0.91})
        self.assertEqual(seed.training["corpus_id"], "synthetic-ru-pii")

    def test_rejects_tampered_artifact_before_decoding(self) -> None:
        self.write_seed(artifact=self.artifact + b"tampered")
        with self.assertRaisesRegex(SeedArtifactError, "SHA-256"):
            load_seed_model(self.manifest_path)

    def test_rejects_non_v2_artifact_with_matching_manifest_hash(self) -> None:
        incompatible = make_artifact(version=1)
        self.manifest["sha256"] = hashlib.sha256(incompatible).hexdigest()
        self.write_seed(artifact=incompatible)
        with self.assertRaisesRegex(SeedArtifactError, "artifact version 2"):
            load_seed_model(self.manifest_path)

    def test_rejects_incompatible_weight_dimensions(self) -> None:
        incompatible = make_artifact(shape=(3, 7))
        self.manifest["sha256"] = hashlib.sha256(incompatible).hexdigest()
        self.write_seed(artifact=incompatible)
        with self.assertRaisesRegex(SeedArtifactError, "coefficient dimensions"):
            load_seed_model(self.manifest_path)

    def test_rejects_categories_that_disagree_with_bio_classes(self) -> None:
        self.manifest["categories"] = ["PHONE"]
        self.write_seed()
        with self.assertRaisesRegex(SeedArtifactError, "exactly match"):
            load_seed_model(self.manifest_path)

    def test_rejects_local_paths_in_public_training_metadata(self) -> None:
        training = dict(self.manifest["training"])
        training["corpus_id"] = "C:/private/training-corpus"
        self.manifest["training"] = training
        self.write_seed()
        with self.assertRaisesRegex(SeedArtifactError, "safe identifier"):
            load_seed_model(self.manifest_path)


class SeedInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        artifact = make_artifact()
        self.seed = SeedModel(
            artifact=artifact,
            sha256=hashlib.sha256(artifact).hexdigest(),
            version="0.1.0",
            classes=("B-PER", "I-PER", "O"),
            categories=("PER",),
            metrics={"entity_micro_f1": 0.91},
            training={"corpus_id": "synthetic-ru-pii"},
        )

    def install_with_rows(self, rows: list[dict[str, int] | None]) -> tuple[bool, MagicMock]:
        connection = MagicMock()
        db = connection.__enter__.return_value
        db.execute.return_value.fetchone.side_effect = rows
        with patch("app.store.connect", return_value=connection):
            installed = install_seed_model(self.seed)
        return installed, db

    def test_installs_active_first_ru_pii_model_without_corpus_snapshot(self) -> None:
        installed, db = self.install_with_rows([None, None])
        self.assertTrue(installed)
        self.assertEqual(db.execute.call_count, 4)
        lock_query = db.execute.call_args_list[0].args[0]
        self.assertIn("pg_advisory_xact_lock(84327122)", lock_query)
        insert_query, params = db.execute.call_args_list[3].args
        self.assertIn("VALUES (%s,1,%s,%s,TRUE,'ru-pii',1,NULL,'[]'::jsonb,%s,0)", insert_query)
        self.assertEqual(params[0], self.seed.artifact)
        self.assertEqual(params[1], self.seed.sha256)
        self.assertEqual(params[2].obj["entity_micro_f1"], 0.91)
        self.assertEqual(params[2].obj["seed_model"]["version"], "0.1.0")
        self.assertEqual(params[3].obj, ["PER"])

    def test_does_not_insert_when_ru_pii_series_already_exists(self) -> None:
        installed, db = self.install_with_rows([{"id": 4}])
        self.assertFalse(installed)
        self.assertEqual(db.execute.call_count, 2)

    def test_does_not_replace_an_existing_active_model(self) -> None:
        installed, db = self.install_with_rows([None, {"id": 2}])
        self.assertFalse(installed)
        self.assertEqual(db.execute.call_count, 3)

    def test_rejects_seed_bytes_that_no_longer_match_manifest_hash(self) -> None:
        self.seed = SeedModel(
            artifact=self.seed.artifact + b"changed",
            sha256=self.seed.sha256,
            version=self.seed.version,
            classes=self.seed.classes,
            categories=self.seed.categories,
            metrics=self.seed.metrics,
            training=self.seed.training,
        )
        with self.assertRaisesRegex(ValueError, "integrity validation"):
            install_seed_model(self.seed)


if __name__ == "__main__":
    unittest.main()

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torch.utils.data import WeightedRandomSampler

from api.data_preprocessing.pad.data_loader import get_dataloader_pad


PREPARE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "PAD" / "prepare_pad.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_pad", PREPARE_PATH)
prepare_pad = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_pad)


class PADPreprocessingTest(unittest.TestCase):
    def _make_synthetic_dataset(self, root):
        raw_dir = root / "raw" / "imgs_part_1" / "imgs_part_1"
        raw_dir.mkdir(parents=True)
        metadata_path = root / "raw" / "metadata.csv"
        fieldnames = [
            "patient_id",
            "lesion_id",
            "age",
            "diagnostic",
            "img_id",
        ]
        rows = []
        for patient_number in range(72):
            diagnosis = prepare_pad.EXPECTED_DIAGNOSES[patient_number % 6]
            images_for_patient = 2 if patient_number % 9 == 0 else 1
            for image_number in range(images_for_patient):
                image_id = f"PAT_{patient_number}_{image_number}.png"
                value = patient_number * 2 + image_number
                Image.new(
                    "RGB",
                    (12, 12),
                    (value % 256, (value * 3) % 256, (value * 7) % 256),
                ).save(raw_dir / image_id)
                rows.append(
                    {
                        "patient_id": f"PAT_{patient_number}",
                        "lesion_id": f"LESION_{patient_number}_{image_number}",
                        "age": "" if patient_number % 5 == 0 else str(20 + patient_number),
                        "diagnostic": diagnosis,
                        "img_id": image_id,
                    }
                )
        with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def test_patient_safe_split_and_client_shards(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_rows = self._make_synthetic_dataset(root)
            summary = prepare_pad.prepare_dataset(
                root,
                client_count=4,
                test_fraction=0.2,
                alpha=0.5,
                seed=42,
                strict=False,
            )

            processed = root / "processed"
            with (processed / "PAD_metadata_train.csv").open(newline="") as csv_file:
                train_rows = list(csv.DictReader(csv_file))
            with (processed / "PAD_metadata_test.csv").open(newline="") as csv_file:
                test_rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(source_rows), len(train_rows) + len(test_rows))
            self.assertFalse(
                {row["patient_id"] for row in train_rows}
                & {row["patient_id"] for row in test_rows}
            )
            self.assertEqual(set(range(6)), {int(row["label"]) for row in train_rows})
            self.assertEqual(set(range(6)), {int(row["label"]) for row in test_rows})

            client_rows = []
            patient_owners = {}
            for client_id in range(1, 5):
                with (processed / f"client_{client_id}_train.csv").open(
                    newline=""
                ) as csv_file:
                    shard = list(csv.DictReader(csv_file))
                client_rows.extend(shard)
                for patient_id in {row["patient_id"] for row in shard}:
                    self.assertNotIn(patient_id, patient_owners)
                    patient_owners[patient_id] = client_id

            self.assertEqual(
                {row["img_id"] for row in train_rows},
                {row["img_id"] for row in client_rows},
            )
            self.assertEqual(len(train_rows), len(client_rows))
            self.assertEqual(6, len(summary["label_mapping"]))
            self.assertEqual(len(source_rows), summary["source_images"])
            self.assertEqual(0, summary["duplicate_audit"]["excluded_images"])
            self.assertEqual(3, len(summary["normalization"]["mean"]))
            self.assertTrue(all(value > 0 for value in summary["normalization"]["std"]))

            with (processed / "PAD_preprocessing_summary.json").open() as summary_file:
                saved_summary = json.load(summary_file)
            self.assertEqual(summary["train_images"], saved_summary["train_images"])
            self.assertEqual(summary["test_images"], saved_summary["test_images"])

            train_loader, test_loader = get_dataloader_pad(
                str(root),
                train_batch_size=8,
                test_batch_size=8,
                train_csv=str(processed / "PAD_metadata_train.csv"),
                test_csv=str(processed / "PAD_metadata_test.csv"),
            )
            train_images, train_labels = next(iter(train_loader))
            test_images, test_labels = next(iter(test_loader))
            self.assertIsInstance(train_loader.sampler, WeightedRandomSampler)
            self.assertEqual((3, 64, 64), tuple(train_images.shape[1:]))
            self.assertEqual((3, 64, 64), tuple(test_images.shape[1:]))
            self.assertEqual(train_images.shape[0], train_labels.shape[0])
            self.assertEqual(test_images.shape[0], test_labels.shape[0])

    def test_conflicting_exact_duplicate_images_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_rows = self._make_synthetic_dataset(root)
            ack = next(row for row in source_rows if row["diagnostic"] == "ACK")
            bcc = next(row for row in source_rows if row["diagnostic"] == "BCC")
            image_dir = root / "raw" / "imgs_part_1" / "imgs_part_1"
            (image_dir / bcc["img_id"]).write_bytes(
                (image_dir / ack["img_id"]).read_bytes()
            )

            summary = prepare_pad.prepare_dataset(
                root,
                client_count=4,
                test_fraction=0.2,
                alpha=0.5,
                seed=42,
                strict=False,
            )

            with (root / "processed" / "PAD_metadata_preprocessed.csv").open(
                newline=""
            ) as csv_file:
                retained_ids = {row["img_id"] for row in csv.DictReader(csv_file)}
            self.assertNotIn(ack["img_id"], retained_ids)
            self.assertNotIn(bcc["img_id"], retained_ids)
            self.assertEqual(
                1, summary["duplicate_audit"]["conflicting_duplicate_groups"]
            )
            self.assertEqual(2, summary["duplicate_audit"]["conflicting_images_removed"])

    def test_stale_client_shards_are_removed(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._make_synthetic_dataset(root)
            processed = root / "processed"
            processed.mkdir()
            (processed / "client_99_train.csv").write_text("stale\n", encoding="utf-8")

            prepare_pad.prepare_dataset(
                root,
                client_count=4,
                test_fraction=0.2,
                alpha=0.5,
                seed=42,
                strict=False,
            )

            self.assertFalse((processed / "client_99_train.csv").exists())
            self.assertEqual(4, len(list(processed.glob("client_*_train.csv"))))


if __name__ == "__main__":
    unittest.main()

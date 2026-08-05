import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


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
                (raw_dir / image_id).touch()
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

            with (processed / "PAD_preprocessing_summary.json").open() as summary_file:
                saved_summary = json.load(summary_file)
            self.assertEqual(summary["train_images"], saved_summary["train_images"])
            self.assertEqual(summary["test_images"], saved_summary["test_images"])


if __name__ == "__main__":
    unittest.main()


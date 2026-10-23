"""Train the LBPH model from one face per image in each student folder."""

import csv
import hashlib
import os
import re
import shutil
import tempfile

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
MODEL_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
LABEL_MAP_PATH = os.path.join(TRAINER_DIR, "label_map.csv")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
STUDENTS_PATH = os.path.join(BASE_DIR, "students.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_SIZE = (200, 200)
MIN_TRAINING_SAMPLES = 5

# Keep the established labels stable even if the old label map is unavailable.
PREFERRED_STUDENT_ORDER = [
    "S25CSEU2146",
    "S25CSEU2083",
    "S25CSEU2113",
    "S25CSEU2112",
]


def load_students():
    """Load and validate the roster; IDs are case-insensitive."""
    if not os.path.isfile(STUDENTS_PATH):
        raise FileNotFoundError(f"Student roster not found: {STUDENTS_PATH}")
    students = {}
    with open(STUDENTS_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"Student_ID", "Student_Name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("students.csv must have Student_ID,Student_Name columns")
        for row in reader:
            student_id = (row.get("Student_ID") or "").strip().upper()
            student_name = (row.get("Student_Name") or "").strip()
            if not student_id or not student_name:
                raise ValueError("students.csv contains a blank student ID or name")
            if not re.fullmatch(r"[A-Z0-9_-]+", student_id):
                raise ValueError(f"Invalid student ID in students.csv: {student_id!r}")
            if student_id in students:
                raise ValueError(f"Duplicate student ID in students.csv: {student_id}")
            students[student_id] = student_name
    if not students:
        raise ValueError("students.csv contains no students")
    return students


def get_student_folders(students):
    """Return registered student folders in stable label order."""
    folders = [
        name for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    ]
    by_id = {}
    for folder_name in folders:
        # Hidden directories are never valid student-ID folders. This also keeps
        # leftover temporary/OneDrive staging folders out of the model.
        if folder_name.startswith("."):
            print(f"IGNORED hidden dataset folder: {folder_name}")
            continue
        key = folder_name.casefold()
        if key in by_id:
            raise ValueError(f"Duplicate dataset folders for student ID: {folder_name}")
        student_id = folder_name.upper()
        if student_id not in students:
            folder_path = os.path.join(DATASET_DIR, folder_name)
            if not os.listdir(folder_path):
                print(
                    f"IGNORED empty unregistered dataset folder: {folder_name}. "
                    "Register the student before adding face images."
                )
                continue
            raise ValueError(
                f"Dataset folder {folder_name!r} contains data but is not registered "
                "in students.csv. Register the correct student ID before training."
            )
        by_id[key] = folder_name
    ordered = [
        by_id[student_id.casefold()]
        for student_id in PREFERRED_STUDENT_ORDER
        if student_id.casefold() in by_id
    ]
    selected = {name.casefold() for name in ordered}
    remaining = sorted(
        (name for name in by_id.values() if name.casefold() not in selected),
        key=str.casefold,
    )
    ordered.extend(remaining)
    return [(folder_name.upper(), folder_name) for folder_name in ordered]


def load_previous_label_map():
    """Read the last label map when present so existing labels do not shift."""
    if not os.path.isfile(LABEL_MAP_PATH):
        return {}
    previous = {}
    with open(LABEL_MAP_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"Label", "Student_ID"}.issubset(reader.fieldnames):
            print("Ignoring malformed previous label_map.csv; new labels will be assigned.")
            return {}
        for row in reader:
            try:
                label = int((row.get("Label") or "").strip())
                student_id = (row.get("Student_ID") or "").strip().upper()
            except (TypeError, ValueError, KeyError, AttributeError):
                continue
            if label > 0 and re.fullmatch(r"[A-Z0-9_-]+", student_id):
                previous[label] = student_id
    return previous


def build_label_assignments(student_folders):
    """Keep existing labels; put every newly added student at the next free label."""
    folder_by_id = dict(student_folders)
    assignments = {}
    assigned_ids = set()

    # The established students keep their documented labels whenever present.
    for label, student_id in enumerate(PREFERRED_STUDENT_ORDER, start=1):
        if student_id in folder_by_id:
            assignments[label] = student_id
            assigned_ids.add(student_id)

    for label, student_id in sorted(load_previous_label_map().items()):
        if (
            label not in assignments
            and student_id in folder_by_id
            and student_id not in assigned_ids
        ):
            assignments[label] = student_id
            assigned_ids.add(student_id)

    next_label = 1
    for student_id in sorted(folder_by_id, key=str.casefold):
        if student_id in assigned_ids:
            continue
        while next_label in assignments:
            next_label += 1
        assignments[next_label] = student_id
        assigned_ids.add(student_id)

    return [
        (label, student_id, folder_by_id[student_id])
        for label, student_id in sorted(assignments.items())
    ]


def main():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")
    if not os.path.isfile(CASCADE_PATH):
        raise FileNotFoundError(f"Haar Cascade file not found: {CASCADE_PATH}")

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        raise RuntimeError(f"Could not load Haar Cascade: {CASCADE_PATH}")
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError as error:
        raise RuntimeError(
            "OpenCV face module is missing. Install opencv-contrib-python."
        ) from error

    students = load_students()
    student_folders = get_student_folders(students)
    if not student_folders:
        raise RuntimeError("No registered student folders found inside dataset/.")

    folder_ids = {student_id for student_id, _ in student_folders}
    for student_id in sorted(set(students) - folder_ids):
        print(f"SKIPPED roster entry without dataset folder: {student_id}")

    faces = []
    labels = []
    label_map = {}
    sample_owners = {}

    for label, student_id, folder_name in build_label_assignments(student_folders):
        label_map[label] = student_id
        folder_path = os.path.join(DATASET_DIR, folder_name)
        image_names = sorted(os.listdir(folder_path), key=str.casefold)
        candidate_names = [
            name for name in image_names
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            and os.path.isfile(os.path.join(folder_path, name))
        ]
        valid_count = 0

        for image_name in candidate_names:
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"SKIPPED unreadable image: {image_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            found = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            if len(found) == 0:
                # register.py stores already-detected face crops. Haar may not
                # detect a second face inside a tight crop, so safely reuse the
                # whole crop only when its filename follows our capture format.
                capture_name = re.fullmatch(
                    rf"{re.escape(student_id)}_\d+",
                    os.path.splitext(image_name)[0],
                    flags=re.IGNORECASE,
                )
                if capture_name and min(gray.shape[:2]) >= 40:
                    face = gray
                    print(f"USING registered face crop: {image_name}")
                else:
                    print(f"SKIPPED (no face detected): {image_name}")
                    continue
            elif len(found) > 1:
                print(f"SKIPPED (multiple faces detected): {image_name}")
                continue
            else:
                x, y, width, height = found[0]
                face = gray[y:y + height, x:x + width]
            if face.size == 0:
                print(f"SKIPPED (empty face crop): {image_name}")
                continue
            face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_AREA)
            fingerprint = hashlib.sha256(face.tobytes()).hexdigest()
            previous_owner = sample_owners.get(fingerprint)
            if previous_owner is not None:
                if previous_owner != student_id:
                    raise ValueError(
                        f"The same face image was found in both {previous_owner} and "
                        f"{student_id} ({image_name}). Check the student folders; "
                        "model files were not changed."
                    )
                print(f"SKIPPED duplicate face sample: {image_name}")
                continue
            sample_owners[fingerprint] = student_id
            faces.append(face)
            labels.append(label)
            valid_count += 1

        print(
            f"{student_id}: {len(candidate_names)} images found; "
            f"{valid_count} valid face samples used (label {label})"
        )
        if valid_count < MIN_TRAINING_SAMPLES:
            raise RuntimeError(
                f"{student_id} has only {valid_count} valid face samples; "
                f"at least {MIN_TRAINING_SAMPLES} are required. "
                "Existing model files were left unchanged."
            )

    if not faces:
        raise RuntimeError("No valid face samples found; model was not changed.")

    os.makedirs(TRAINER_DIR, exist_ok=True)
    recognizer.train(faces, np.asarray(labels, dtype=np.int32))
    staging_dir = tempfile.mkdtemp(prefix=".training-", dir=TRAINER_DIR)
    staged_model = os.path.join(staging_dir, "trainer.yml")
    staged_label_map = os.path.join(staging_dir, "label_map.csv")
    old_label_map = os.path.join(staging_dir, "previous-label_map.csv")
    had_label_map = os.path.isfile(LABEL_MAP_PATH)
    try:
        recognizer.write(staged_model)
        with open(staged_label_map, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Label", "Student_ID"])
            writer.writerows(label_map.items())
            file.flush()
            os.fsync(file.fileno())
        if not os.path.isfile(staged_model) or os.path.getsize(staged_model) == 0:
            raise RuntimeError("OpenCV did not produce a usable model file.")

        if had_label_map:
            shutil.copy2(LABEL_MAP_PATH, old_label_map)
        os.replace(staged_label_map, LABEL_MAP_PATH)
        try:
            os.replace(staged_model, MODEL_PATH)
        except Exception:
            if had_label_map:
                os.replace(old_label_map, LABEL_MAP_PATH)
            elif os.path.exists(LABEL_MAP_PATH):
                os.remove(LABEL_MAP_PATH)
            raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Label map saved: {LABEL_MAP_PATH}")
    print(f"Total valid face samples: {len(faces)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)

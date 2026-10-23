"""Register one student and capture face crops without mixing identities."""

import csv
import os
import re
import tempfile
import time

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
STUDENTS_PATH = os.path.join(BASE_DIR, "students.csv")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
WINDOW_NAME = "Student Face Registration"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SAMPLES_TO_CAPTURE = 30
CAPTURE_INTERVAL_SECONDS = 0.4
MIN_TRAINING_SAMPLES = 5


def load_roster():
    """Return roster records and reject duplicate/corrupt IDs."""
    if not os.path.isfile(STUDENTS_PATH):
        return {}
    students = {}
    with open(STUDENTS_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"Student_ID", "Student_Name"}.issubset(reader.fieldnames):
            raise ValueError("students.csv must have Student_ID,Student_Name columns")
        for row in reader:
            student_id = (row.get("Student_ID") or "").strip().upper()
            student_name = (row.get("Student_Name") or "").strip()
            if not student_id or not student_name:
                raise ValueError("students.csv contains a blank student ID or name")
            if student_id in students:
                raise ValueError(f"Duplicate student ID in students.csv: {student_id}")
            if not re.fullmatch(r"[A-Z0-9_-]+", student_id):
                raise ValueError(f"Invalid student ID in students.csv: {student_id!r}")
            students[student_id] = student_name
    return students


def save_roster(students, student_id, student_name):
    """Atomically save the new roster row after a usable face set is captured."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=BASE_DIR,
            prefix=".students-", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = file.name
            writer = csv.writer(file)
            writer.writerow(["Student_ID", "Student_Name"])
            writer.writerows(students.items())
            writer.writerow([student_id, student_name])
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, STUDENTS_PATH)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def next_image_number(student_folder, student_id):
    """Choose a filename index that never overwrites existing face images."""
    pattern = re.compile(rf"^{re.escape(student_id)}_(\d+)\.[^.]+$", re.IGNORECASE)
    highest = 0
    for name in os.listdir(student_folder):
        match = pattern.match(name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def find_student_folder(student_id):
    """Resolve an existing dataset folder without creating a case duplicate."""
    matches = [
        os.path.join(DATASET_DIR, name)
        for name in os.listdir(DATASET_DIR)
        if not name.startswith(".")
        and os.path.isdir(os.path.join(DATASET_DIR, name))
        and name.casefold() == student_id.casefold()
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple dataset folders match student ID {student_id}.")
    return matches[0] if matches else os.path.join(DATASET_DIR, student_id)


def is_valid_camera_frame(ret, frame):
    """Reject failed, malformed, or fully black camera frames."""
    if not ret or frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return False
    if frame.ndim != 3 or frame.shape[2] < 3 or min(frame.shape[:2]) <= 0:
        return False
    try:
        sample = cv2.resize(frame, (16, 12), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        _, max_value, _, _ = cv2.minMaxLoc(gray)
    except cv2.error:
        return False
    return max_value > 2


def initialize_camera():
    """Try DirectShow first, then Media Foundation if frames stay black."""
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
    ]
    for backend, backend_name in backends:
        camera = None
        try:
            camera = cv2.VideoCapture(0, backend)
            if not camera.isOpened():
                camera.release()
                continue
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            if camera is not None:
                camera.release()
            continue

        # Let the webcam settle and discard empty/black startup frames.
        for _ in range(3):
            try:
                ret, frame = camera.read()
            except cv2.error:
                break
            if is_valid_camera_frame(ret, frame):
                print(f"Camera ready using {backend_name}.")
                return camera, frame
            time.sleep(0.01)
        camera.release()

    raise RuntimeError(
        "The camera returned only black frames with DirectShow and Media Foundation. "
        "Check the lens shutter and Windows camera privacy permission, and close "
        "other apps that may be using the camera."
    )


def main():
    if not os.path.isfile(CASCADE_PATH):
        raise FileNotFoundError(f"Haar Cascade not found: {CASCADE_PATH}")
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar Cascade: {CASCADE_PATH}")

    print("STUDENT REGISTRATION")
    student_id = input("Enter Student ID: ").strip().upper()
    student_name = input("Enter Student Name: ").strip()
    if not re.fullmatch(r"[A-Z0-9_-]+", student_id):
        raise ValueError("Student ID may contain only letters, numbers, '_' and '-'.")
    if not student_name:
        raise ValueError("Student name cannot be blank.")

    students = load_roster()
    if student_id in students and students[student_id].casefold() != student_name.casefold():
        raise ValueError(
            f"{student_id} is already registered as {students[student_id]!r}. "
            "Update students.csv deliberately before changing that name."
        )
    is_new_student = student_id not in students
    os.makedirs(DATASET_DIR, exist_ok=True)
    final_student_folder = find_student_folder(student_id)
    pending_folder = os.path.join(DATASET_DIR, f".pending-{student_id}")
    if is_new_student:
        if os.path.isdir(final_student_folder) and os.listdir(final_student_folder):
            raise ValueError(
                f"Unregistered dataset folder {final_student_folder!r} already has files. "
                "Move those images aside or register the matching student ID first."
            )
        student_folder = pending_folder
    else:
        student_folder = final_student_folder
    os.makedirs(student_folder, exist_ok=True)

    image_number = next_image_number(student_folder, student_id)
    saved_count = 0
    last_saved_at = 0.0
    camera, first_frame = initialize_camera()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1000, 700)
        try:
            # Keep the separate OpenCV preview visible above the VS Code terminal.
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass
        print("Camera opened. The preview window should appear in front of VS Code.")
        print("Show one face in that preview. Samples save automatically every 0.4 seconds.")
        print("Wait for Saved: 30/30; at least 5 clear samples are needed to train.")

        while saved_count < SAMPLES_TO_CAPTURE:
            if first_frame is not None:
                frame = first_frame
                first_frame = None
                ret = True
            else:
                ret, frame = camera.read()
            if not is_valid_camera_frame(ret, frame):
                # Keep the registration UI alive during transient webcam drops,
                # but never run detection or save a crop from an invalid frame.
                frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 24, dtype=np.uint8)
                cv2.putText(
                    frame, "Camera frame unavailable - retrying", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2,
                )
                cv2.putText(
                    frame, f"Saved: {saved_count}/{SAMPLES_TO_CAPTURE}", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                )
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                continue
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100)
            )

            status = "Show one face; samples are spaced apart"
            color = (0, 255, 0)
            if len(faces) > 1:
                status = "Multiple faces: no image saved"
                color = (0, 0, 255)
            elif len(faces) == 0:
                status = "No face detected - move closer and face the camera"
                color = (0, 0, 255)
            else:
                x, y, width, height = (int(value) for value in faces[0])
                now = time.monotonic()
                if now - last_saved_at >= CAPTURE_INTERVAL_SECONDS:
                    crop = gray[y:y + height, x:x + width]
                    image_path = os.path.join(
                        student_folder, f"{student_id}_{image_number}.jpg"
                    )
                    if cv2.imwrite(image_path, crop):
                        image_number += 1
                        saved_count += 1
                        last_saved_at = now
                    else:
                        status = "Could not save image"
                        color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

            cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2)
            cv2.putText(frame, f"Saved: {saved_count}/{SAMPLES_TO_CAPTURE}",
                        (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if is_new_student and saved_count >= MIN_TRAINING_SAMPLES:
        if os.path.isdir(final_student_folder):
            os.rmdir(final_student_folder)  # It was checked empty before capture.
        os.replace(student_folder, final_student_folder)
        try:
            save_roster(students, student_id, student_name)
        except Exception:
            # Keep the captured images out of the active training set if the
            # roster could not be committed (for example, while OneDrive locks it).
            os.replace(final_student_folder, student_folder)
            raise
        print("Student roster saved after confirming usable face samples.")
    elif is_new_student:
        print(
            f"Student not added to roster yet. Capture at least "
            f"{MIN_TRAINING_SAMPLES} samples, then rerun registration for {student_id}."
        )

    print(f"Student: {student_name} ({student_id})")
    print(f"New images saved: {saved_count}")
    print(f"Dataset folder: {student_folder}")
    if saved_count < SAMPLES_TO_CAPTURE:
        print("Registration capture stopped early; run train.py only when enough good samples exist.")
    else:
        print("Next: run train.py, then attendance.py.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)

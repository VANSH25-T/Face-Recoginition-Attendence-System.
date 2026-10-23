"""Camera attendance using the existing LBPH model and label map."""

import csv
import math
import os
import re
import time

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(BASE_DIR, "trainer", "trainer.yml")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "trainer", "label_map.csv")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "attendance.csv")
STUDENTS_PATH = os.path.join(BASE_DIR, "students.csv")

# LBPH returns a distance: lower is a closer match. Use conservative acceptance
# and require a clear lead over the next student's best training sample.
MAX_DISTANCE = 55.0
MIN_DISTANCE_MARGIN = 5.0
REQUIRED_CONFIRMATIONS = 5
EXPECTED_LABEL_MAP = {
    1: "S25CSEU2146",
    2: "S25CSEU2083",
    3: "S25CSEU2113",
    4: "S25CSEU2112",
}
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_SCALE = 0.5
MIN_FACE_SIZE = (40, 40)  # Detection runs at half resolution: 80x80 source pixels.
FACE_EDGE_MARGIN = 8
MIN_FACE_SHARPNESS = 25.0
WINDOW_NAME = "Face Recognition Attendance"
PANEL_WIDTH = 360
WINDOW_WIDTH = CAMERA_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = CAMERA_HEIGHT
SUCCESS_DISPLAY_SECONDS = 2.0
CAMERA_BACKENDS = (
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_MSMF, "Media Foundation"),
)


def check_required_files():
    """Stop with a clear message if model, map, camera cascade, or CSV is absent."""
    paths = [CASCADE_PATH, MODEL_PATH, LABEL_MAP_PATH, ATTENDANCE_PATH, STUDENTS_PATH]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Required project files are missing:\n" + "\n".join(missing)
        )
    with open(ATTENDANCE_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        header = next(reader, None)
    expected = ["Student_ID", "Student_Name", "Date", "Time", "Status"]
    if header != expected:
        raise ValueError(
            "attendance.csv must start with this header: " + ",".join(expected)
        )


def load_students():
    """Read display names, keyed by normalized student ID."""
    students = {}
    with open(STUDENTS_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"Student_ID", "Student_Name"}.issubset(reader.fieldnames):
            raise ValueError("students.csv must have Student_ID,Student_Name columns")
        for row in reader:
            student_id = (row.get("Student_ID") or "").strip().upper()
            student_name = (row.get("Student_Name") or "").strip()
            if student_id and student_name:
                if not re.fullmatch(r"[A-Z0-9_-]+", student_id):
                    raise ValueError(f"Invalid student ID in students.csv: {student_id!r}")
                if student_id in students:
                    raise ValueError(f"Duplicate student ID in students.csv: {student_id}")
                students[student_id] = student_name
    return students


def load_label_map():
    """Read numeric LBPH labels from trainer/label_map.csv."""
    label_map = {}
    with open(LABEL_MAP_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"Label", "Student_ID"}.issubset(reader.fieldnames):
            raise ValueError("label_map.csv must have Label,Student_ID columns")
        for row in reader:
            try:
                label = int((row.get("Label") or "").strip())
            except (TypeError, ValueError) as error:
                raise ValueError("label_map.csv contains a non-numeric label") from error
            student_id = (row.get("Student_ID") or "").strip().upper()
            if (
                label <= 0
                or not re.fullmatch(r"[A-Z0-9_-]+", student_id)
                or label in label_map
                or student_id in label_map.values()
            ):
                raise ValueError("label_map.csv contains an invalid or duplicate label")
            label_map[label] = student_id
    if not label_map:
        raise ValueError("label_map.csv contains no student labels")
    return label_map


def validate_model_label_set(recognizer, label_map):
    """Fail closed when model labels and the ID map do not describe same classes."""
    wrong_expected = {
        label: student_id
        for label, student_id in EXPECTED_LABEL_MAP.items()
        if label_map.get(label) != student_id
    }
    if wrong_expected:
        raise ValueError(
            "trainer/label_map.csv does not match the established student labels. "
            "Expected labels 1-4 to map to S25CSEU2146, S25CSEU2083, "
            "S25CSEU2113, and S25CSEU2112 in that order. Retrain before attendance."
        )
    labels = np.asarray(recognizer.getLabels()).reshape(-1)
    model_label_set = {int(label) for label in labels}
    if model_label_set != set(label_map):
        raise ValueError(
            "The trained model labels do not match trainer/label_map.csv. "
            "Retrain the model and label map together before taking attendance."
        )


def is_valid_camera_frame(ret, frame):
    """Reject missing, malformed, or fully black frames before display/detection."""
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
    # A completely black stream often means the selected camera/backend did not
    # deliver an image. Treat it as unavailable instead of reporting no face.
    return max_value > 2


def initialize_camera():
    """Open the built-in camera at index 0; use MSMF if DirectShow is unavailable."""
    for backend, backend_name in CAMERA_BACKENDS:
        camera = None
        try:
            camera = cv2.VideoCapture(0, backend)
            if not camera.isOpened():
                camera.release()
                continue
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            if camera is not None:
                camera.release()
            continue

        # Read immediately, then allow only two quick retries for camera startup.
        for attempt in range(3):
            try:
                ret, frame = camera.read()
            except cv2.error:
                break
            if is_valid_camera_frame(ret, frame):
                height, width = frame.shape[:2]
                print("Camera index used: 0")
                print(f"Camera backend used: {backend_name}")
                print(f"Camera opened: {camera.isOpened()}")
                print("First frame received: True")
                print(f"Frame size: {width} x {height}")
                return camera, frame
            if attempt < 2:
                time.sleep(0.03)
        camera.release()

    raise RuntimeError(
        "Could not open camera index 0 or receive a valid frame using DirectShow "
        "or Media Foundation. Check camera permissions, the lens/shutter, and "
        "whether another app is using the webcam."
    )


def make_camera_unavailable_view(today_entries):
    """Build a status screen instead of showing a stale or black camera frame."""
    frame = np.full(
        (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        (31, 35, 43),
        dtype=np.uint8,
    )
    cv2.putText(
        frame,
        "CAMERA STATUS",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (235, 235, 235),
        2,
    )
    cv2.putText(
        frame,
        "Camera frame unavailable",
        (24, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (60, 80, 255),
        2,
    )
    cv2.putText(
        frame,
        "Retrying camera...  Q = Quit  R = Reset",
        (24, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
    )
    dashboard = cv2.copyMakeBorder(
        frame,
        0,
        0,
        0,
        PANEL_WIDTH,
        cv2.BORDER_CONSTANT,
        value=(27, 32, 40),
    )
    draw_dashboard_panel(
        dashboard,
        frame.shape[1],
        ["CAMERA FRAME", "UNAVAILABLE"],
        (70, 70, 255),
        ["Waiting for a valid frame", "Face detection is paused"],
        today_entries,
    )
    return dashboard


def already_marked(student_id, attendance_date):
    """Check the CSV case-insensitively to avoid same-day duplicate entries."""
    with open(ATTENDANCE_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            saved_id = (row.get("Student_ID") or "").strip().upper()
            saved_date = (row.get("Date") or "").strip()
            status = (row.get("Status") or "").strip().casefold()
            if saved_id == student_id.upper() and saved_date == attendance_date and status == "present":
                return True
    return False


def load_today_attendance():
    """Read today's existing Present records for the dashboard activity list."""
    today = time.strftime("%Y-%m-%d")
    entries = []
    with open(ATTENDANCE_PATH, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if (
                (row.get("Date") or "").strip() == today
                and (row.get("Status") or "").strip().casefold() == "present"
            ):
                entries.append({
                    "time": (row.get("Time") or "").strip(),
                    "name": (row.get("Student_Name") or "").strip(),
                    "student_id": (row.get("Student_ID") or "").strip().upper(),
                })
    return entries


def mark_attendance(student_id, students):
    """Append one Present record only after successful frame confirmation."""
    now = time.localtime()
    date_text = time.strftime("%Y-%m-%d", now)
    if already_marked(student_id, date_text):
        print(f"Already marked today: {student_id}")
        return False
    with open(ATTENDANCE_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            student_id,
            students[student_id],
            date_text,
            time.strftime("%H:%M:%S", now),
            "Present",
        ])
    print(f"Attendance marked: {student_id} - {students[student_id]}")
    return True


def lbph_feature_histogram(face_gray, recognizer):
    """Compute OpenCV LBPH's normalized spatial histogram for one gray face."""
    radius = int(recognizer.getRadius())
    neighbors = int(recognizer.getNeighbors())
    grid_x = int(recognizer.getGridX())
    grid_y = int(recognizer.getGridY())
    source = np.asarray(face_gray, dtype=np.uint8)
    height, width = source.shape[:2]
    if radius < 1 or neighbors < 1 or grid_x < 1 or grid_y < 1:
        raise ValueError("The LBPH model has invalid radius, neighbor, or grid settings")
    if height <= radius * 2 or width <= radius * 2:
        raise ValueError("Face crop is too small for the LBPH model")

    source_float = source.astype(np.float32)
    center = source_float[radius:height - radius, radius:width - radius]
    lbp = np.zeros(center.shape, dtype=np.int32)
    epsilon = np.finfo(np.float32).eps
    for neighbor in range(neighbors):
        angle = 2.0 * np.pi * neighbor / float(neighbors)
        x_offset = float(np.float32(radius * np.cos(angle)))
        y_offset = float(np.float32(-radius * np.sin(angle)))
        x_floor, y_floor = int(np.floor(x_offset)), int(np.floor(y_offset))
        x_ceil, y_ceil = int(np.ceil(x_offset)), int(np.ceil(y_offset))
        tx, ty = np.float32(x_offset - x_floor), np.float32(y_offset - y_floor)
        w1 = np.float32((1.0 - tx) * (1.0 - ty))
        w2 = np.float32(tx * (1.0 - ty))
        w3 = np.float32((1.0 - tx) * ty)
        w4 = np.float32(tx * ty)
        rows = slice(radius + y_floor, height - radius + y_floor)
        rows_ceil = slice(radius + y_ceil, height - radius + y_ceil)
        cols = slice(radius + x_floor, width - radius + x_floor)
        cols_ceil = slice(radius + x_ceil, width - radius + x_ceil)
        sample = (
            w1 * source_float[rows, cols]
            + w2 * source_float[rows, cols_ceil]
            + w3 * source_float[rows_ceil, cols]
            + w4 * source_float[rows_ceil, cols_ceil]
        )
        bit = (sample > center) | (np.abs(sample - center) < epsilon)
        lbp |= bit.astype(np.int32) << neighbor

    cell_width = lbp.shape[1] // grid_x
    cell_height = lbp.shape[0] // grid_y
    if cell_width == 0 or cell_height == 0:
        raise ValueError("Face crop is too small for the LBPH model grid")
    bins = 1 << neighbors
    cells = []
    for grid_row in range(grid_y):
        for grid_column in range(grid_x):
            cell = lbp[
                grid_row * cell_height:(grid_row + 1) * cell_height,
                grid_column * cell_width:(grid_column + 1) * cell_width,
            ]
            histogram = np.bincount(cell.ravel(), minlength=bins).astype(np.float32)
            histogram /= max(1, cell.size)
            cells.append(histogram)
    return np.concatenate(cells).reshape(1, -1).astype(np.float32)


def recognize_face(face_gray, recognizer, label_map, students):
    """Return (student ID, best distance, result reason); fail closed if uncertain."""
    face_gray = cv2.equalizeHist(face_gray)
    face_gray = cv2.resize(face_gray, (200, 200), interpolation=cv2.INTER_AREA)
    query_histogram = lbph_feature_histogram(face_gray, recognizer)
    model_labels = np.asarray(recognizer.getLabels()).reshape(-1)
    model_histograms = recognizer.getHistograms()
    if len(model_labels) != len(model_histograms) or not model_histograms:
        return None, float("inf"), "UNCERTAIN"

    # LBPH stores one histogram per training image. Compare the nearest sample
    # for each distinct student, so another photo of the same student cannot
    # masquerade as the runner-up candidate.
    best_by_label = {}
    for raw_label, trained_histogram in zip(model_labels, model_histograms):
        label = int(raw_label)
        distance = float(cv2.compareHist(
            query_histogram,
            np.asarray(trained_histogram, dtype=np.float32).reshape(1, -1),
            cv2.HISTCMP_CHISQR_ALT,
        ))
        if math.isfinite(distance):
            best_by_label[label] = min(best_by_label.get(label, float("inf")), distance)
    ranked = sorted(best_by_label.items(), key=lambda item: item[1])
    if not ranked:
        return None, float("inf"), "UNCERTAIN"

    best_label, best_distance = ranked[0]
    # Guard the manual per-class ranking against differences between this
    # OpenCV build and the LBPH prediction implementation. If they disagree,
    # reject the frame instead of risking a wrong identity.
    predicted_label, predicted_distance = recognizer.predict(face_gray)
    if (
        int(predicted_label) != best_label
        or abs(float(predicted_distance) - best_distance) > 0.5
    ):
        return None, best_distance, "UNCERTAIN"
    student_id = label_map.get(best_label)
    if best_distance > MAX_DISTANCE or student_id not in students:
        return None, best_distance, "DISTANCE"
    if len(ranked) > 1:
        margin = ranked[1][1] - best_distance
        if margin < MIN_DISTANCE_MARGIN:
            return None, best_distance, "UNCERTAIN"
    return student_id, best_distance, "MATCH"


def face_quality_reason(face_gray, box, frame_shape):
    """Reject face crops that are too small, near an edge, or too blurry."""
    x, y, width, height = box
    frame_height, frame_width = frame_shape[:2]
    if width < MIN_FACE_SIZE[0] / FRAME_SCALE or height < MIN_FACE_SIZE[1] / FRAME_SCALE:
        return "SMALL"
    if (
        x < FACE_EDGE_MARGIN or y < FACE_EDGE_MARGIN
        or x + width > frame_width - FACE_EDGE_MARGIN
        or y + height > frame_height - FACE_EDGE_MARGIN
    ):
        return "EDGE"
    if face_gray.size == 0:
        return "BLUR"
    sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    if sharpness < MIN_FACE_SHARPNESS:
        return "BLUR"
    return None


def apply_zoom(frame, zoom):
    """Center crop and resize back to source dimensions, preserving aspect ratio."""
    if zoom <= 1.0:
        return frame
    height, width = frame.shape[:2]
    crop_width = max(1, int(width / zoom))
    crop_height = max(1, int(height / zoom))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    crop = frame[top:top + crop_height, left:left + crop_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_interface(frame, faces, face_results, candidate_label, confirmation_count,
                   attendance_done, students):
    """Draw each current frame's result and its current confirmation progress."""
    cv2.putText(frame, "FACE ATTENDANCE", (20, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2)
    cv2.putText(frame, "Q = Quit   R = Reset recognition   * = Zoom in   - = Zoom out",
                (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if not faces:
        cv2.putText(frame, "No face detected", (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2)
        return

    for (x, y, width, height), (student_id, distance, reason) in zip(faces, face_results):
        distance_text = f"{distance:.1f}" if math.isfinite(distance) else "N/A"
        if student_id is None:
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            label_text = "UNKNOWN / UNCERTAIN" if reason == "UNCERTAIN" else "UNKNOWN"
            cv2.putText(frame, label_text, (x, max(25, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Distance: {distance_text}", (x, y + height + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(frame, f"Confirming: 0/{REQUIRED_CONFIRMATIONS}",
                        (x, max(25, y - 38)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)
            continue

        color = (0, 255, 0) if attendance_done else (0, 220, 255)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        identity_confirmed = (
            candidate_label == student_id
            and confirmation_count >= REQUIRED_CONFIRMATIONS
        )
        cv2.putText(frame, students[student_id], (x, max(25, y - 62)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"ID: {student_id}", (x, max(25, y - 36)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"Distance: {distance_text}", (x, y + height + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        shown_count = confirmation_count if candidate_label == student_id else 0
        cv2.putText(frame, f"Confirming: {shown_count}/{REQUIRED_CONFIRMATIONS}",
                    (x, max(25, y - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)
        if attendance_done and identity_confirmed:
            cv2.putText(frame, "PRESENT", (x, y + height + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_dashboard_panel(canvas, panel_x, title_lines, status_color,
                         detail_lines, today_entries):
    """Draw a dark status panel and recent attendance list beside the camera."""
    panel_right = canvas.shape[1]
    cv2.rectangle(canvas, (panel_x, 0), (panel_right, canvas.shape[0]),
                  (27, 32, 40), -1)
    x = panel_x + 22
    cv2.putText(canvas, "ATTENDANCE DASHBOARD", (x, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)
    cv2.line(canvas, (x, 54), (panel_right - 18, 54), (78, 88, 102), 1)

    if any("MARKED" in line for line in title_lines):
        # Draw a clear green check beside the success state.
        cv2.line(canvas, (panel_right - 54, 77), (panel_right - 43, 88),
                 (75, 245, 115), 4)
        cv2.line(canvas, (panel_right - 43, 88), (panel_right - 22, 65),
                 (75, 245, 115), 4)

    y = 92
    for line in title_lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, status_color, 2)
        y += 28
    for line in detail_lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (215, 220, 228), 1)
        y += 23

    history_y = max(190, y + 26)
    cv2.line(canvas, (x, history_y - 18), (panel_right - 18, history_y - 18),
             (78, 88, 102), 1)
    cv2.putText(canvas, "TODAY'S ATTENDANCE", (x, history_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 2)
    history_y += 32
    if not today_entries:
        cv2.putText(canvas, "No entries yet", (x, history_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (155, 165, 178), 1)
        return

    for entry in reversed(today_entries[-6:]):
        if history_y > canvas.shape[0] - 55:
            break
        name = entry["name"][:27]
        cv2.putText(canvas, f"{entry['time']}  {name}", (x, history_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (100, 235, 135), 1)
        history_y += 20
        cv2.putText(canvas, entry["student_id"][:28], (x, history_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (175, 185, 197), 1)
        history_y += 24


def main():
    print("Face recognition can be uncertain. High-distance faces are rejected as UNKNOWN.")
    check_required_files()
    students = load_students()
    label_map = load_label_map()
    unknown_ids = sorted(set(label_map.values()) - set(students))
    if unknown_ids:
        raise ValueError(
            "The trained label map contains IDs missing from students.csv: "
            + ", ".join(unknown_ids)
            + ". Retrain after correcting the roster."
        )

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar Cascade: {CASCADE_PATH}")
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError as error:
        raise RuntimeError("OpenCV face module is missing; install opencv-contrib-python.") from error
    recognizer.read(MODEL_PATH)
    validate_model_label_set(recognizer, label_map)

    # A label without a name is refused by recognize_face and treated as UNKNOWN.
    for label, student_id in label_map.items():
        print(f"Label {label} -> {student_id} -> {students.get(student_id, 'unregistered')}")

    today_entries = load_today_attendance()
    camera = None
    first_frame = None
    zoom = 1.0
    candidate_label = None
    confirmation_count = 0
    marked_this_run = set()
    attendance_status = {}
    close_at = None
    print("Press Q to quit, R to reset recognition, * to zoom in, - to zoom out.")
    print(
        f"Maximum distance: {MAX_DISTANCE:.1f}; margin: {MIN_DISTANCE_MARGIN:.1f}; "
        f"confirmations: {REQUIRED_CONFIRMATIONS}"
    )

    try:
        camera, first_frame = initialize_camera()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        while True:
            if first_frame is not None:
                frame = first_frame
                first_frame = None
                ret = True
            else:
                ret, frame = camera.read()

            if not is_valid_camera_frame(ret, frame):
                # An unavailable frame breaks consecutiveness and must never be
                # passed into resize, face detection, or LBPH prediction.
                candidate_label = None
                confirmation_count = 0
                cv2.imshow(
                    WINDOW_NAME,
                    make_camera_unavailable_view(today_entries),
                )
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("r"), ord("R")):
                    candidate_label = None
                    confirmation_count = 0
                    print("Recognition confirmation reset.")
                elif key == ord("*"):
                    zoom = min(2.0, zoom + 0.1)
                elif key in (ord("-"), ord("_")):
                    zoom = max(1.0, zoom - 0.1)
                continue

            frame = cv2.flip(frame, 1)
            frame = apply_zoom(frame, zoom)
            small_frame = cv2.resize(
                frame, None, fx=FRAME_SCALE, fy=FRAME_SCALE,
                interpolation=cv2.INTER_AREA,
            )
            small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            detected_small = cascade.detectMultiScale(
                small_gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE
            )
            small_faces = [tuple(int(value) for value in face) for face in detected_small]
            frame_height, frame_width = frame.shape[:2]
            faces = []
            face_results = []
            for small_x, small_y, small_width, small_height in small_faces:
                x = int(round(small_x / FRAME_SCALE))
                y = int(round(small_y / FRAME_SCALE))
                width = int(round(small_width / FRAME_SCALE))
                height = int(round(small_height / FRAME_SCALE))
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                width = max(1, min(width, frame_width - x))
                height = max(1, min(height, frame_height - y))
                faces.append((x, y, width, height))

                # This UI is designed for one-person attendance. Never identify
                # anyone while the frame contains multiple faces.
                if len(small_faces) > 1:
                    face_results.append((None, float("inf"), "MULTIPLE"))
                    continue

                # Check image quality before the crop reaches the recognizer.
                roi = small_gray[
                    small_y:small_y + small_height,
                    small_x:small_x + small_width,
                ]
                quality_issue = face_quality_reason(roi, (x, y, width, height), frame.shape)
                if quality_issue:
                    face_results.append((None, float("inf"), quality_issue))
                elif roi.size == 0:
                    face_results.append((None, float("inf"), "BLUR"))
                else:
                    try:
                        face_results.append(
                            recognize_face(roi, recognizer, label_map, students)
                        )
                    except cv2.error:
                        face_results.append((None, float("inf"), "UNCERTAIN"))

            # Only a single current-frame identity can advance attendance.
            # UNKNOWN, no face, or multiple faces resets the count immediately.
            current_student = face_results[0][0] if len(faces) == 1 else None
            if current_student is None:
                candidate_label = None
                confirmation_count = 0
            elif current_student == candidate_label:
                confirmation_count = min(
                    REQUIRED_CONFIRMATIONS, confirmation_count + 1
                )
            else:
                candidate_label = current_student
                confirmation_count = 1

            attendance_done = False
            if current_student and confirmation_count >= REQUIRED_CONFIRMATIONS:
                if current_student not in marked_this_run:
                    was_added = mark_attendance(current_student, students)
                    marked_this_run.add(current_student)
                    if was_added:
                        attendance_status[current_student] = "ATTENDANCE MARKED"
                        today_entries.append({
                            "time": time.strftime("%H:%M:%S"),
                            "name": students[current_student],
                            "student_id": current_student,
                        })
                    else:
                        attendance_status[current_student] = "ALREADY MARKED TODAY"
                    if was_added:
                        print(
                            f"ATTENDANCE MARKED for {students[current_student]} "
                            f"({current_student}). Closing camera..."
                        )
                    else:
                        print(
                            f"Attendance was already marked today for "
                            f"{students[current_student]} ({current_student}). Closing camera..."
                        )
                    close_at = time.monotonic() + SUCCESS_DISPLAY_SECONDS
                attendance_done = True

            draw_interface(frame, faces, face_results, candidate_label,
                           confirmation_count, attendance_done, students)

            if len(faces) > 1:
                panel_title = ["MULTIPLE FACES", "PLEASE STAND ALONE"]
                panel_color = (0, 185, 255)
                panel_details = ["Show one person at a time"]
            elif not faces:
                panel_title = ["READY", "NO FACE DETECTED"]
                panel_color = (170, 180, 190)
                panel_details = ["Center one face in the camera"]
            elif not face_results[0][0]:
                reason = face_results[0][2]
                if reason == "UNCERTAIN":
                    panel_title = ["UNKNOWN / UNCERTAIN"]
                    panel_details = ["Competing student matches are too close",
                                     f"Best distance: {face_results[0][1]:.1f}",
                                     "No attendance recorded"]
                elif reason in {"SMALL", "EDGE", "BLUR"}:
                    panel_title = ["UNKNOWN", "FACE QUALITY LOW"]
                    panel_details = ["Center closer and keep still",
                                     "No attendance recorded"]
                else:
                    panel_title = ["UNKNOWN", "NOT REGISTERED"]
                    panel_details = ["Distance too high or no roster match",
                                     f"Distance: {face_results[0][1]:.1f}",
                                     "No attendance recorded"]
                panel_color = (70, 70, 255)
                panel_details.append(f"Confirming: 0/{REQUIRED_CONFIRMATIONS}")
            elif attendance_done:
                panel_title = [attendance_status.get(current_student, "ATTENDANCE MARKED")]
                panel_color = (75, 230, 115)
                panel_details = [students[current_student], current_student,
                                 f"Distance: {face_results[0][1]:.1f}"]
            else:
                panel_title = ["VERIFYING MATCH"]
                panel_color = (0, 190, 255)
                panel_details = ["Identity not confirmed yet",
                                 f"Distance: {face_results[0][1]:.1f}",
                                 f"Confirmations: {confirmation_count}/{REQUIRED_CONFIRMATIONS}"]

            dashboard = cv2.copyMakeBorder(
                frame, 0, 0, 0, PANEL_WIDTH, cv2.BORDER_CONSTANT,
                value=(27, 32, 40)
            )
            draw_dashboard_panel(dashboard, frame.shape[1], panel_title,
                                 panel_color, panel_details, today_entries)

            cv2.imshow(WINDOW_NAME, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if close_at is not None and time.monotonic() >= close_at:
                break
            if key in (ord("r"), ord("R")):
                candidate_label = None
                confirmation_count = 0
                print("Recognition confirmation reset.")
            elif key == ord("*"):
                zoom = min(2.0, zoom + 0.1)
            elif key in (ord("-"), ord("_")):
                zoom = max(1.0, zoom - 0.1)
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError, cv2.error) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)

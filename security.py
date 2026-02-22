import cv2
import time
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from supabase_helper import upload_alert


def in_poly(poly, point):
    return cv2.pointPolygonTest(np.array(poly, np.int32), point, False) >= 0


# =========================
# SETTINGS
# =========================
VIDEO_SOURCE = "rtsp://127.0.0.1:8554/org08_camera1"

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Cannot open video/webcam. Check VIDEO_SOURCE.")
    raise SystemExit

CONF = 0.4

# Door 2 zones (ACTIVE)
ZONE2_A = [(234, 1734), (579, 1554), (621, 1641), (192, 1839)]    # outside
ZONE2_B = [(216, 1689), (240, 1620), (564, 1455), (564, 1515)]    # inside

# Office hours
OFFICE_START = 8
OFFICE_END = 11

# Periodic snapshots
SNAPSHOT_DIR = "snapshots"
SNAPSHOT_INTERVAL = 5 * 60
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Entry snapshots
ENTRY_SNAPSHOT_DIR = "entry_snapshots"
os.makedirs(ENTRY_SNAPSHOT_DIR, exist_ok=True)
ENTRY_SNAPSHOT_THROTTLE_SEC = 2.0
last_entry_snapshot_time = 0

# Exit snapshots
EXIT_SNAPSHOT_DIR = "exit_snapshots"
os.makedirs(EXIT_SNAPSHOT_DIR, exist_ok=True)
EXIT_SNAPSHOT_THROTTLE_SEC = 2.0
last_exit_snapshot_time = 0

# Alerts
ALERT_DIR = "alerts"
os.makedirs(ALERT_DIR, exist_ok=True)
ALERT_THROTTLE_SEC = 60
last_alert_time = 0

# Suspicious rules (after-hours)
MAX_STAY_SECONDS = 20
MAX_PEOPLE_AFTER_HOURS = 1

# =========================
# AFTER-HOURS DASHBOARD ALERT SETTINGS
# =========================
AFTER_HOURS_PERSIST_SEC = 1.5
DASHBOARD_ALERT_THROTTLE_SEC = 15

after_hours_first_seen = {}            # tid -> first time seen after hours
after_hours_alerted = set()            # tids already alerted (per session)
last_dashboard_alert_time = 0          # global throttle timestamp


# =========================
# INIT
# =========================
person_model = YOLO("yolo11n.pt")
box_model = YOLO("best.pt")

BOX_CONF = 0.25

# Tracking settings for boxes
BOX_TRACKER = "botsort.yaml"   # you can swap to "bytetrack.yaml" if preferred
BOX_PERSIST = True

cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RGB", 1600, 900)

# Door 2 memory
door2_seenA = {}
door2_seenB = {}

# To avoid double counting
entered_ids = set()
exited_ids = set()

first_seen = {}        # id -> first seen time
room_count = 0
initialized = False
last_snapshot_time = time.time()


# =========================
# HELPERS
# =========================
def after_hours():
    h = datetime.now().hour
    return h < OFFICE_START or h >= OFFICE_END

def draw_zone(img, poly, color=(255, 255, 255), alpha=0.02, thickness=1):
    overlay = img.copy()
    pts = np.array(poly, np.int32)

    if alpha > 0:
        cv2.fillPoly(overlay, [pts], color)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    if thickness > 0:
        cv2.polylines(img, [pts], True, color, thickness)

    return img

def save_alert(frame, reason, count):
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_THROTTLE_SEC:
        return
    last_alert_time = now

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_reason = reason.replace(" ", "_").replace(":", "_")
    path = f"{ALERT_DIR}/ALERT_{ts}_{safe_reason}_count{count}.jpg"
    cv2.imwrite(path, frame)
    print(f"🚨 ALERT [{reason}] saved: {path}")

    upload_alert(
        frame_path=path,
        camera_name="Camera-01",
        alert_type=reason,
        message=f"🚨 Person detected after office hours! {reason}",
        room_count=count
    )

def save_entry_snapshot(frame, tid, count):
    global last_entry_snapshot_time
    now = time.time()
    if now - last_entry_snapshot_time < ENTRY_SNAPSHOT_THROTTLE_SEC:
        return
    last_entry_snapshot_time = now

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{ENTRY_SNAPSHOT_DIR}/ENTRY_{ts}_ID{tid}_count{count}.jpg"
    cv2.imwrite(path, frame)
    print(f"📸 ENTRY snapshot saved: {path}")

def save_exit_snapshot(frame, tid, count):
    global last_exit_snapshot_time
    now = time.time()
    if now - last_exit_snapshot_time < EXIT_SNAPSHOT_THROTTLE_SEC:
        return
    last_exit_snapshot_time = now

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{EXIT_SNAPSHOT_DIR}/EXIT_{ts}_ID{tid}_count{count}.jpg"
    cv2.imwrite(path, frame)
    print(f"📸 EXIT snapshot saved: {path}")


print("✅ Running... Press Q or ESC to quit")


# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = cap.read()
    if not ret:
        print("⚠️ No frame... reconnecting RTSP")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    now = time.time()
    is_after = after_hours()

    # Draw Door2 zones
    frame = draw_zone(frame, ZONE2_A, color=(255, 255, 255), alpha=0.01, thickness=1)
    frame = draw_zone(frame, ZONE2_B, color=(255, 255, 255), alpha=0.01, thickness=1)

    # =========================
    # BOX DETECTION + TRACKING (UPDATED)
    # =========================
    stock_boxes_now = 0
    unique_box_ids_now = 0

    box_results = box_model.track(
        frame,
        persist=BOX_PERSIST,
        conf=BOX_CONF,
        tracker=BOX_TRACKER,
        verbose=False
        # If your model has multiple classes and "box" is a specific class,
        # you can add: classes=[<box_class_id>]
    )

    if box_results and len(box_results) > 0:
        b = box_results[0].boxes
        if b is not None and b.xyxy is not None:
            box_xyxy = b.xyxy.int().cpu().tolist()
            stock_boxes_now = len(box_xyxy)

            if b.id is not None:
                box_ids = b.id.int().cpu().tolist()
                unique_box_ids_now = len(set(box_ids))
            else:
                box_ids = [None] * len(box_xyxy)

            for (x1, y1, x2, y2), bid in zip(box_xyxy, box_ids):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f"BOX ID:{bid}" if bid is not None else "BOX"
                cv2.putText(frame, label, (x1, max(20, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    print(f"[DEBUG] hour={datetime.now().hour} is_after={is_after} boxes_now={stock_boxes_now} unique_box_ids_now={unique_box_ids_now}")

    # =========================
    # PERSON TRACKING
    # =========================
    results = person_model.track(
        frame,
        persist=True,
        conf=CONF,
        classes=[0],
        tracker="botsort.yaml",
        verbose=False
    )

    boxes = results[0].boxes if results and len(results) > 0 else None
    detected_now = 0

    if boxes is not None and boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.int().cpu().tolist()
        detected_now = len(ids)

        print(f"[DEBUG] detected_now={detected_now} ids_sample={ids[:5]}")

        # cleanup: remove IDs no longer present (after-hours alert session memory)
        current_ids = set(ids)
        for old_id in list(after_hours_first_seen.keys()):
            if old_id not in current_ids:
                after_hours_first_seen.pop(old_id, None)
                after_hours_alerted.discard(old_id)

        if not initialized:
            room_count = detected_now
            initialized = True

        for box, tid in zip(xyxy, ids):
            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            foot = (cx, y2)

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, foot, 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ID:{tid}", (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # =========================
            # AFTER-HOURS HUMAN ALERT (screen-wide)
            # =========================
            if is_after:
                if tid not in after_hours_first_seen:
                    after_hours_first_seen[tid] = now

                persisted = (now - after_hours_first_seen[tid]) >= AFTER_HOURS_PERSIST_SEC
                throttled_ok = (now - last_dashboard_alert_time) >= DASHBOARD_ALERT_THROTTLE_SEC

                if persisted and throttled_ok and tid not in after_hours_alerted:
                    after_hours_alerted.add(tid)
                    last_dashboard_alert_time = now
                    save_alert(frame, f"human_after_hours_ID{tid}", room_count)

            # Door 2: detect A->B (IN) and B->A (OUT)
            if in_poly(ZONE2_A, foot):
                door2_seenA[tid] = time.time()

            if in_poly(ZONE2_B, foot):
                door2_seenB[tid] = time.time()

            # IN via door2
            if tid in door2_seenA and in_poly(ZONE2_B, foot) and tid not in entered_ids:
                entered_ids.add(tid)
                room_count += 1
                save_entry_snapshot(frame, tid, room_count)

            # OUT via door2
            if tid in door2_seenB and in_poly(ZONE2_A, foot) and tid not in exited_ids:
                exited_ids.add(tid)
                room_count = max(0, room_count - 1)
                save_exit_snapshot(frame, tid, room_count)

            if tid not in first_seen:
                first_seen[tid] = now

            # after-hours loitering
            if is_after and (now - first_seen.get(tid, now) > MAX_STAY_SECONDS):
                save_alert(frame, f"loitering_ID{tid}", room_count)

    # after-hours multiple people
    if is_after and room_count > MAX_PEOPLE_AFTER_HOURS:
        save_alert(frame, f"multiple_people_{room_count}", room_count)

    # periodic snapshot
    if now - last_snapshot_time >= SNAPSHOT_INTERVAL:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{SNAPSHOT_DIR}/snap_{ts}_count{room_count}.jpg"
        cv2.imwrite(path, frame)
        print(f"📸 Snapshot saved: {path}")
        last_snapshot_time = now

    # Display info
    cv2.putText(frame, f"Detected Now: {detected_now}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Room Count: {room_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"Stock Boxes Now: {stock_boxes_now}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Unique Box IDs Now: {unique_box_ids_now}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("RGB", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Stopped.")

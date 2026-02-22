import cv2

VIDEO_SOURCE = "rtsp://127.0.0.1:8554/org08_camera1"

# Read one frame (use FFMPEG for RTSP)
cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
ret, frame0 = cap.read()
cap.release()

if not ret:
    print("❌ Can't read first frame")
    raise SystemExit

points = []

# --- display target size (change if needed) ---
DISPLAY_W, DISPLAY_H = 1280, 720

h0, w0 = frame0.shape[:2]
scale = min(DISPLAY_W / w0, DISPLAY_H / h0)  # keep aspect ratio
new_w, new_h = int(w0 * scale), int(h0 * scale)

def mouse_cb(event, x, y, flags, param):
    global points, scale
    if event == cv2.EVENT_LBUTTONDOWN:
        # convert display click -> original coords
        ox = int(x / scale)
        oy = int(y / scale)
        points.append((ox, oy))
        print(points[-1])

cv2.namedWindow("Pick", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pick", DISPLAY_W, DISPLAY_H)
cv2.setMouseCallback("Pick", mouse_cb)

while True:
    # resize for display
    show = cv2.resize(frame0, (new_w, new_h))

    # draw clicked points (original -> display coords)
    for (ox, oy) in points:
        dx = int(ox * scale)
        dy = int(oy * scale)
        cv2.circle(show, (dx, dy), 5, (0, 255, 0), -1)

    cv2.imshow("Pick", show)
    k = cv2.waitKey(1) & 0xFF

    if k == ord("c"):
        points = []
        print("cleared")
    if k == ord("q") or k == 27:
        break

cv2.destroyAllWindows()
print("FINAL POINTS (original frame coords):", points)

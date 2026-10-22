import os

import cv2
from ultralytics import YOLO

# ---------- Settings (change these to experiment) ----------
VIDEO_PATH = "videos/luggage_carousel_01.mp4"
OUTPUT_PATH = "runs/counting/luggage_carousel_01_counted.mp4"
MODEL_PATH = "yolo26n.pt"
BAG_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase
CONFIDENCE = 0.5            # ignore boxes the model is less sure about
FRAME_SKIP = 2              # process every 2nd frame (60 fps -> 30 fps)

# Counting line: a vertical line in the middle of the frame.
# Top and bottom are fractions of the frame height (0.0 = top, 1.0 = bottom).
LINE_TOP = 0.10
LINE_BOTTOM = 0.65
# -----------------------------------------------------------

# Read the video's size and speed so the output video matches
video = cv2.VideoCapture(VIDEO_PATH)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS) / FRAME_SKIP
video.release()

line_x = width // 2
line_y1 = int(height * LINE_TOP)
line_y2 = int(height * LINE_BOTTOM)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

model = YOLO(MODEL_PATH)
results = model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    stream=True,
    verbose=False,
    classes=BAG_CLASSES,
    conf=CONFIDENCE,
    vid_stride=FRAME_SKIP,
)

last_side = {}       # track ID -> which side of the line the bag was on last time
counted_ids = set()  # track IDs that have already been counted

cv2.namedWindow("BagCount", cv2.WINDOW_NORMAL)

for result in results:
    frame = result.orig_img.copy()
    boxes = result.boxes

    if boxes.id is not None:
        for box, track_id in zip(boxes.xyxy.tolist(), boxes.id.int().tolist()):
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Which side of the line is the bag's center on?
            side = "left" if center_x < line_x else "right"
            inside_line_area = line_y1 <= center_y <= line_y2

            # Count the bag when its center moves to the other side of the line
            previous_side = last_side.get(track_id)
            if (
                previous_side is not None
                and previous_side != side
                and inside_line_area
                and track_id not in counted_ids
            ):
                counted_ids.add(track_id)

            last_side[track_id] = side

            # Red = already counted, green = not counted yet (OpenCV colors are Blue, Green, Red)
            color = (0, 0, 255) if track_id in counted_ids else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.circle(frame, (center_x, center_y), 6, color, -1)
            cv2.putText(frame, f"id:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Draw the counting line and the total (on a black background so it's readable)
    cv2.line(frame, (line_x, line_y1), (line_x, line_y2), (255, 0, 255), 4)
    cv2.rectangle(frame, (20, 15), (560, 85), (0, 0, 0), -1)
    cv2.putText(frame, f"Bags counted: {len(counted_ids)}", (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

    writer.write(frame)
    cv2.imshow("BagCount", frame)

    # Press q on the video window to stop early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

writer.release()
cv2.destroyAllWindows()
print(f"Total bags counted: {len(counted_ids)}")
print(f"Saved video to {OUTPUT_PATH}")
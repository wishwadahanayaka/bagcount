from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.track(
    source="videos/luggage_carousel_01.mp4",
    show=True,
    tracker="bytetrack.yaml",
    save=True,
    stream=True,          # process one frame at a time (saves memory)
    verbose=False,        # stop printing a line for every frame
    classes=[24, 26, 28], # only detect backpack, handbag, suitcase
)

for frame_number, result in enumerate(results, start=1):
    if frame_number % 100 == 0:
        print(f"Processed {frame_number} frames...")

print("Done!")
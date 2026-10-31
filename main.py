import cv2
from datetime import datetime
from ultralytics import YOLO

# 1. Load the pretrained object-detection model
model = YOLO("yolo11n.pt")

# 2. Load the flood photo
image = cv2.imread("floodphoto.jpeg")

if image is None:
    raise FileNotFoundError(
        "Cannot find floodphoto.jpeg. Check the image filename and folder."
    )

# 3. Detect objects (ignore weak detections)
results = model(image, conf=0.40)
result = results[0]

# In a flood scene, "boat" is usually a submerged vehicle
for k, v in result.names.items():
    if v == "boat":
        result.names[k] = "flooded vehicle"

# 4. Draw YOLO bounding boxes and labels
output = result.plot()

# 5. Collect detected object names
detected = set()

if result.boxes is not None:
    for class_id in result.boxes.cls.tolist():
        name = result.names[int(class_id)]
        detected.add(name.lower())

print("Detected objects:", sorted(detected))

# 6. Draw an obstacle-aware path of arrows
height, width = output.shape[:2]

# Start at the bottom center of the image
start = (width // 2, height - 40)

# Divide the image into left, center, and right regions
regions = {
    "left": (0, width // 3),
    "center": (width // 3, 2 * width // 3),
    "right": (2 * width // 3, width)
}

# Obstacle score per region (sum of box areas)
scores = {"left": 0, "center": 0, "right": 0}

if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls[0])
        name = result.names[class_id].lower()

        if name in {
            "person", "car", "truck", "bus",
            "motorcycle", "bicycle", "flooded vehicle"
        }:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            center_x = (x1 + x2) // 2

            for region, (left, right) in regions.items():
                if left <= center_x < right:
                    scores[region] += (x2 - x1) * (y2 - y1)
                    break

# Select the region with the smallest obstacle area
direction = min(scores, key=scores.get)

# Rule: if even the freest zone is heavily blocked, there is no clear path
zone_area = (width // 3) * height
blocked = min(scores.values()) * 100 / zone_area >= 30

# Aim at the ground, not at the middle of the image
target_y = int(height * 0.65)

if direction == "left":
    end = (width // 6, target_y)
elif direction == "right":
    end = (5 * width // 6, target_y)
else:
    end = (width // 2, target_y)

num_arrows = 5

if not blocked:
    # Path made of consecutive arrows (thinner = farther away)
    for i in range(num_arrows):
        t1 = i / num_arrows
        t2 = (i + 0.75) / num_arrows  # small gap between arrows

        p1 = (
            int(start[0] + (end[0] - start[0]) * t1),
            int(start[1] + (end[1] - start[1]) * t1)
        )
        p2 = (
            int(start[0] + (end[0] - start[0]) * t2),
            int(start[1] + (end[1] - start[1]) * t2)
        )

        thickness_arrow = max(2, 7 - i)

        cv2.arrowedLine(
            output, p1, p2,
            (0, 255, 0), thickness_arrow, tipLength=0.5
        )

    cv2.putText(
        output,
        "Suggested direction only - NOT verified safe",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 255), 2
    )
else:
    cv2.putText(
        output,
        "NO CLEAR PATH - STAY PUT AND CALL FOR HELP",
        (10, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 255), 3
    )

# 7. Generate context-based safety advice
warnings = ["FLOOD SAFETY"]

if "person" in detected:
    warnings.append("Person detected: avoid walking through floodwater.")

if detected.intersection(
    {"car", "truck", "bus", "motorcycle", "flooded vehicle"}
):
    warnings.append("Vehicle detected: do not drive through floodwater.")

if detected.intersection({"bicycle"}):
    warnings.append("Bicycle detected: avoid flooded roads.")

if blocked:
    warnings.append(
        "No clear path detected: do not move through the water, call for help."
    )

warnings.append("Move to higher ground if flooding threatens.")
warnings.append("Follow official emergency instructions.")

# 8. Draw a safety panel that fits all messages
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.55
thickness = 1
line_height = 28
panel_height = 20 + line_height * len(warnings)

# Add space at the top so the panel does not cover detections
output = cv2.copyMakeBorder(
    output,
    panel_height, 0, 0, 0,
    cv2.BORDER_CONSTANT,
    value=(30, 30, 30)
)

for i, message in enumerate(warnings):
    y = 25 + i * line_height

    cv2.putText(
        output,
        message,
        (15, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

# 9. Generate the detailed emergency report
vehicle_names = ("car", "truck", "bus", "motorcycle", "flooded vehicle")

# Collect every detection with its confidence and position
detections = []
if result.boxes is not None:
    for box in result.boxes:
        name = result.names[int(box.cls[0])].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2

        if cx < width // 3:
            position = "left"
        elif cx < 2 * width // 3:
            position = "center"
        else:
            position = "right"

        detections.append((name, conf, position))

# Count objects
counts = {}
for name, conf, position in detections:
    counts[name] = counts.get(name, 0) + 1

vehicle_count = sum(counts.get(n, 0) for n in vehicle_names)
person_count = counts.get("person", 0)
bicycle_count = counts.get("bicycle", 0)

# Risk level with an explanation
if person_count > 0:
    risk = "HIGH"
    reason = "At least one person is in the scene."
elif vehicle_count >= 3:
    risk = "HIGH"
    reason = "Several vehicles (" + str(vehicle_count) + ") are in the flooded area."
elif blocked:
    risk = "HIGH"
    reason = "No clear path: every zone is heavily blocked."
elif vehicle_count > 0 or bicycle_count > 0:
    risk = "MEDIUM"
    reason = "Vehicle or bicycle detected in the flooded area."
else:
    risk = "LOW"
    reason = "No obstacle detected (water depth is NOT analyzed)."

# Obstacle coverage per zone (in % of the zone area)
coverage = {
    z: min(100, round(scores[z] * 100 / zone_area))
    for z in scores
}

direction_text = {
    "left": "left side",
    "center": "center",
    "right": "right side"
}[direction]

# Advice for each type of object
advice = {
    "person": "Keep people away from floodwater and help them reach higher ground.",
    "car": "Do not drive through floodwater: 30 cm of water can move a car.",
    "flooded vehicle": "Submerged vehicle: stay away, it may block the road or hide hazards.",
    "truck": "Do not drive through floodwater.",
    "bus": "Do not drive through floodwater.",
    "motorcycle": "Do not ride through floodwater.",
    "bicycle": "Avoid flooded roads, depth and obstacles are hidden."
}

report = []
report.append("AI FLOOD EMERGENCY REPORT")
report.append("=" * 50)
report.append("Date: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
report.append("Image analyzed: floodphoto.jpeg (" + str(width) + "x" + str(height) + " px)")
report.append("Model: yolo11n.pt | Confidence threshold: 0.40")
report.append("")

report.append("1. SUMMARY")
report.append("   Total detections: " + str(len(detections)))
report.append("   Risk level: " + risk)
report.append("   Reason: " + reason)
report.append("")

report.append("2. DETECTED OBJECTS")
if detections:
    for name, conf, position in sorted(detections, key=lambda d: -d[1]):
        report.append(
            "   - " + name + " | confidence " + str(round(conf * 100)) + "%"
            + " | position: " + position
        )
else:
    report.append("   - No object detected")
report.append("")

report.append("3. OBSTACLES PER ZONE")
for z in ("left", "center", "right"):
    report.append("   - " + z + ": " + str(coverage[z]) + "% covered")
report.append("")

if blocked:
    report.append("4. SUGGESTED DIRECTION: NONE (no clear path detected)")
else:
    report.append("4. SUGGESTED DIRECTION: " + direction_text)
    report.append("   Least obstructed zone. Suggestion only, NOT verified safe.")
report.append("")

report.append("5. RECOMMENDED ACTIONS")
for message in warnings[1:]:
    report.append("   - " + message)
for name in counts:
    if name in advice:
        report.append("   - [" + name + "] " + advice[name])
report.append("")

low_conf = [d for d in detections if d[1] < 0.50]
report.append("6. LIMITATIONS")
report.append("   - Water depth and current are not measured.")
report.append("   - Hidden obstacles (holes, debris) cannot be detected.")
if low_conf:
    report.append("   - " + str(len(low_conf)) + " detection(s) under 50% confidence: verify manually.")
report.append("   - Always follow official emergency instructions.")

with open("emergency_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n".join(report))

# 10. Save and display the annotated image
cv2.imwrite("emergency_result.jpg", output)

cv2.imshow("AI Flood Emergency Assistant", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
from flask import Flask, request, render_template, jsonify, send_from_directory, send_file
import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import torch
import threading
import ast
import cv2
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = YOLO("R:/BTP_WEBSITE/yolo12_100epoch.pt").to(device) 

SELECTED_CLASS_NAMES = ['Auto-Rickshaw', 'Bicycle', 'Bus', 'Car', 'Cycle-Rickshaw', 'E-Rickshaw', 'Motorcycle', 'Tractor', 'Truck']
name_to_id = {name: id for id, name in model.model.names.items()}
SELECTED_CLASS_IDS = [name_to_id[cls] for cls in SELECTED_CLASS_NAMES if cls in name_to_id]
print(f"Selected Class IDs: {SELECTED_CLASS_IDS}")

total_vehicles = defaultdict(set)
current_frame_vehicles = set()
processing_status = {"status": "idle", "progress": 0}
byte_tracker = None
polygon_zone = None
zone_annotator = None

def initialize_tracker_and_zone(points):
    global byte_tracker, polygon_zone, zone_annotator
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
        minimum_consecutive_frames=3
    )
    polygon_zone = sv.PolygonZone(polygon=np.array(points))
    zone_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, color=sv.Color.BLUE, thickness=2)

box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.RED)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.WHITE)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50, color=sv.Color.GREEN)

def process_video(source_path, target_path, user_polygon_points):
    global total_vehicles, current_frame_vehicles, processing_status
    
    total_vehicles = defaultdict(set)
    current_frame_vehicles = set()
    processing_status["status"] = "processing"
    processing_status["progress"] = 0

    initialize_tracker_and_zone(user_polygon_points)
    byte_tracker.reset()

    video_info = sv.VideoInfo.from_video_path(source_path)
    total_frames = video_info.total_frames

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        global total_vehicles, current_frame_vehicles, processing_status

        results = model(frame, verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
        detections = byte_tracker.update_with_detections(detections=detections)

        in_zone_mask = polygon_zone.trigger(detections=detections)
        detections_in_zone = detections[in_zone_mask]

        current_ids = detections_in_zone.tracker_id.astype(int).tolist()
        current_frame_vehicles = set(current_ids)

        for class_id, tracker_id in zip(detections_in_zone.class_id, current_ids):
            total_vehicles[model.model.names[class_id]].add(tracker_id)

        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections_in_zone.confidence, detections_in_zone.class_id, current_ids)
        ]

        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections_in_zone)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections_in_zone)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_in_zone, labels=labels)
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)

        count_text = f"Current: {len(current_frame_vehicles)} | Total: {sum(len(v) for v in total_vehicles.values())}"
        annotated_frame = sv.draw_text(
            scene=annotated_frame,
            text=count_text,
            text_anchor=sv.Point(50, 50),
            text_color=sv.Color.WHITE,
            background_color=sv.Color.BLACK,
            text_thickness=1,
            text_scale=0.8
        )

        processing_status["progress"] = int((index / total_frames) * 100)
        return annotated_frame

    sv.process_video(source_path=source_path, target_path=target_path, callback=callback)
    processing_status["status"] = "complete"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    polygon_input = request.form.get('polygon_points', '[[615, 504], [1064, 1052], [1905, 1062], [1912, 650], [1309, 502], [1235, 480]]')
    source_path = os.path.join(UPLOAD_FOLDER, file.filename)
    target_path = os.path.join(OUTPUT_FOLDER, "detected_" + file.filename)
    file.save(source_path)

    try:
        user_polygon_points = ast.literal_eval(polygon_input)
        user_polygon_points = np.array(user_polygon_points)
    except (ValueError, SyntaxError):
        return jsonify({"error": "Invalid polygon points format. Use: [[x1, y1], [x2, y2], ...]"}), 400

    thread = threading.Thread(target=process_video, args=(source_path, target_path, user_polygon_points))
    thread.start()

    return jsonify({"message": "Processing started", "filename": "detected_" + file.filename})

@app.route('/status')
def get_status():
    global processing_status, total_vehicles
    if processing_status["status"] == "complete":
        summary_table = (
            "<div style='width: 90%; max-width: 1000px; margin: 30px auto;'>"
            "<h2 style='text-align: center; font-size: 1.75rem; font-weight: bold; color: #fff; margin-bottom: 20px;'>Vehicle Count Summary</h2>"
            "<table style='width: 100%; border-collapse: collapse; background: rgba(255, 255, 255, 0.95); border-radius: 12px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>"
            "<thead>"
            "<tr style='background: #00ffcc; color: #000;'>"
            "<th style='padding: 20px; font-size: 1.2rem; border: 1px solid #ddd; color: #000;'>Vehicle Type</th>"
            "<th style='padding: 20px; font-size: 1.2rem; border: 1px solid #ddd; color: #000;'>Count</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
        )
        total_count = sum(len(v) for v in total_vehicles.values())
        for vehicle_class in SELECTED_CLASS_NAMES:
            count = len(total_vehicles.get(vehicle_class, set()))
            summary_table += (
                f"<tr style='transition: background 0.3s;'>"
                f"<td style='padding: 20px; font-size: 1.1rem; border: 1px solid #ddd; color: #000;'>{vehicle_class}</td>"
                f"<td style='padding: 20px; font-size: 1.1rem; border: 1px solid #ddd; color: #000;'>{count}</td>"
                f"</tr>"
            )
        summary_table += (
            f"<tr style='font-weight: bold; background: #f0f0f0;'>"
            f"<td style='padding: 20px; font-size: 1.1rem; border: 1px solid #ddd; color: #000;'>Total</td>"
            f"<td style='padding: 20px; font-size: 1.1rem; border: 1px solid #ddd; color: #000;'>{total_count}</td>"
            f"</tr>"
            "</tbody>"
            "</table>"
            "</div>"
        )
        return jsonify({"status": "complete", "progress": 100, "summary": summary_table})
    return jsonify({"status": processing_status["status"], "progress": processing_status["progress"]})

@app.route('/outputs/<filename>')
def serve_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False, mimetype='video/mp4')

@app.route('/get-first-frame', methods=['POST'])
def get_first_frame():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    source_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(source_path)

    cap = cv2.VideoCapture(source_path)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        os.remove(source_path)
        return jsonify({"error": "Failed to read video"}), 500
    
    cap.release()

    _, img_buffer = cv2.imencode(".jpg", frame)
    img_io = BytesIO(img_buffer.tobytes())
    os.remove(source_path) 
    
    return send_file(
        img_io,
        mimetype='image/jpeg',
        as_attachment=True,
        download_name=f"{file.filename.split('.')[0]}_first_frame.jpg"
    )

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template, jsonify, send_from_directory, send_file
import os
import threading
import cv2
import numpy as np
import ast
import time
import tempfile
import logging
import io
from collections import defaultdict
from ultralytics import YOLO
from supervision import LineZone, LineZoneAnnotator, BoxAnnotator, LabelAnnotator, TraceAnnotator, ByteTrack, Color, Point
import supervision as sv
import torch
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load YOLO model
try:
    model_path = os.path.join(MODEL_FOLDER, 'yolo12_100epoch.pt')
    model = YOLO(model_path).to(device)
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise e

# Class names and IDs
SELECTED_CLASS_NAMES = [
    'Auto-Rickshaw', 'Bicycle', 'Bus', 'Car', 'Cycle_rickshaw',
    'E-Rickshaw', 'Motorcycle', 'Tractor', 'Truck'
]
name_to_id = {name: id for id, name in model.model.names.items()}
SELECTED_CLASS_IDS = [name_to_id[cls] for cls in SELECTED_CLASS_NAMES if cls in name_to_id]
logger.info(f"Selected Class IDs: {SELECTED_CLASS_IDS}")

# Global state
progress = {'status': 'idle', 'progress': 0, 'summary': {}, 'live_counts': {}, 'entire_screen': False}
lock = threading.Lock()

# Annotators
box_annotator = BoxAnnotator(thickness=2, color=Color.RED)
label_annotator = LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=Color.WHITE)
trace_annotator = TraceAnnotator(thickness=2, trace_length=50, color=Color.GREEN)
line_annotator = LineZoneAnnotator(thickness=2, color=Color.BLUE)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0

def initialize_tracker():
    """Initialize ByteTrack for vehicle tracking."""
    return ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=60,
        minimum_consecutive_frames=3
    )

def estimate_text_width(text, text_scale=1.0):
    """Estimate text width in pixels (approximate)."""
    char_width = 12 * text_scale
    return int(len(text) * char_width)

def draw_count_summary(frame, counts, width, is_video=False):
    """Draw the count summary on the frame with a uniform background."""
    total_count = sum(counts.values()) if not is_video else sum(counts.get(cls, 0) for cls in counts)
    text_scale = 1.0
    text_thickness = 2
    padding = 20
    y_offset = 50
    line_height = 40

    # Prepare text lines
    text_lines = [f"Total: {total_count}"]
    for class_name in sorted(counts.keys()):
        text_lines.append(f"{class_name}: {counts[class_name]}")

    # Calculate max width for uniform background
    max_text_width = max(estimate_text_width(text, text_scale) for text in text_lines)
    rect_width = max_text_width + 2 * padding
    rect_height = len(text_lines) * line_height + padding
    rect_top_left = (width - rect_width - padding, 30)
    rect_bottom_right = (width - padding, 30 + rect_height)

    # Draw a single uniform background rectangle
    cv2.rectangle(
        frame,
        rect_top_left,
        rect_bottom_right,
        (0, 0, 0),
        -1
    )

    # Draw text lines with uniform alignment
    for idx, text in enumerate(text_lines):
        text_width = estimate_text_width(text, text_scale)
        text_anchor = Point(width - rect_width + padding, y_offset)
        frame = sv.draw_text(
            scene=frame,
            text=text,
            text_anchor=text_anchor,
            text_color=Color.WHITE,
            background_color=Color.BLACK,
            text_thickness=text_thickness,
            text_scale=text_scale
        )
        y_offset += line_height

    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global progress
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    media_type = request.form.get('media_type')
    if media_type not in ['image', 'video']:
        logger.error(f"Invalid media_type: {media_type}")
        return jsonify({'error': 'Invalid media_type'}), 400

    # Log the entire form data for debugging
    logger.debug(f"Received form data: {request.form}")

    # Get entire_screen flag
    entire_screen = request.form.get('entire_screen', 'false').lower() == 'true'
    left_line = 0
    start_line = [[0, 0], [0, 0]]
    end_line = [[0, 0], [0, 0]]
    distance = 0

    if not entire_screen:
        try:
            left_line = float(request.form.get('left_line', 0))
            start_line_str = request.form.get('start_line', '[[0,0],[0,0]]')
            end_line_str = request.form.get('end_line', '[[0,0],[0,0]]')
            start_line = ast.literal_eval(start_line_str)
            end_line = ast.literal_eval(end_line_str)
            distance = float(request.form.get('distance', 10)) if media_type == 'video' else 0
        except (ValueError, SyntaxError, TypeError) as e:
            logger.error(f"Invalid coordinate or distance format: {str(e)}")
            return jsonify({'error': 'Invalid coordinate or distance format'}), 400

        # Validate coordinates
        if not (isinstance(start_line, list) and isinstance(end_line, list) and
                len(start_line) == 2 and len(end_line) == 2 and
                all(isinstance(p, list) and len(p) == 2 for p in start_line + end_line)):
            logger.error("Invalid line points format")
            return jsonify({'error': 'Start and end lines must be [[x1,y1], [x2,y2]]'}), 400

    # Log input parameters for debugging
    logger.debug(f"Upload inputs: media_type={media_type}, entire_screen={entire_screen}, "
                 f"left_line={left_line}, start_line={start_line}, end_line={end_line}, distance={distance}")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    output_filename = f"processed_{file.filename}"
    with lock:
        progress = {
            'status': 'processing',
            'progress': 0,
            'summary': {},
            'live_counts': {},
            'entire_screen': entire_screen
        }

    thread = threading.Thread(
        target=process_media,
        args=(file_path, media_type, left_line, start_line, end_line, distance, output_filename)
    )
    thread.start()

    logger.info(f"Started processing {media_type}: {file.filename}")
    return jsonify({'filename': output_filename})

def process_media(file_path, media_type, left_line, start_line, end_line, distance, output_filename):
    global progress
    try:
        if media_type == 'image':
            summary = process_image(file_path, left_line, start_line, end_line)
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, summary['image'])
            with lock:
                progress = {
                    'status': 'complete',
                    'progress': 100,
                    'summary': summary['counts'],
                    'live_counts': {},
                    'entire_screen': progress['entire_screen']
                }
        else:
            summary = process_video(file_path, left_line, start_line, end_line, distance, output_filename)
            summary['video_writer'].release()
            with lock:
                progress = {
                    'status': 'complete',
                    'progress': 100,
                    'summary': summary['counts'],
                    'live_counts': summary['live_counts'],
                    'entire_screen': progress['entire_screen']
                }
        logger.info(f"Completed processing {media_type}: {file_path}")
    except Exception as e:
        logger.error(f"Processing error for {file_path}: {str(e)}")
        with lock:
            progress = {
                'status': 'error',
                'progress': 0,
                'summary': {'error': str(e)},
                'live_counts': {},
                'entire_screen': False
            }
    finally:
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to unlink file {file_path}: {str(e)}")

def process_image(image_path, left_line, start_line, end_line):
    """Process an image to count vehicles, zone-free for entire screen or with LineZone."""
    with lock:
        entire_screen = progress['entire_screen']
    logger.debug(f"process_image: entire_screen={entire_screen}, left_line={left_line}, "
                 f"start_line={start_line}, end_line={end_line}")

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        raise ValueError('Failed to load image')

    height, width = image.shape[:2]
    results = model(image, verbose=False, device=device)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

    # Apply confidence threshold
    conf_mask = detections.confidence > CONFIDENCE_THRESHOLD
    detections = detections[conf_mask]

    counts = defaultdict(int)

    if entire_screen:
        # Entire screen: Count all detections without zones
        logger.debug("Entire screen mode: Counting all detections without zones")
        for class_id in detections.class_id:
            class_name = model.model.names[class_id]
            counts[class_name] += 1
    else:
        # Specific region: Use LineZone and left_line
        logger.debug("Specific region mode: Creating LineZone objects")
        start_line_zone = LineZone(start=Point(*start_line[0]), end=Point(*start_line[1]))
        end_line_zone = LineZone(start=Point(*end_line[0]), end=Point(*end_line[1]))

        # Filter by left_line
        left_mask = detections.xyxy[:, 0] >= left_line
        detections = detections[left_mask]

        # Count vehicles crossing lines
        processed_ids = set()
        for idx, (xyxy, class_id, confidence) in enumerate(
            zip(detections.xyxy, detections.class_id, detections.confidence)
        ):
            if idx in processed_ids:
                continue
            x1, y1, x2, y2 = xyxy
            centroid = Point((x1 + x2) / 2, (y1 + y2) / 2)
            single_detection = sv.Detections(
                xyxy=np.array([xyxy]),
                confidence=np.array([confidence]),
                class_id=np.array([class_id]),
                tracker_id=None
            )
            if start_line_zone.trigger(detections=single_detection) or end_line_zone.trigger(detections=single_detection):
                class_name = model.model.names[class_id]
                counts[class_name] += 1
                processed_ids.add(idx)

    # Annotate image
    labels = [
        f"{model.model.names[class_id]} {conf:.2f}"
        for class_id, conf in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    # Draw lines only in specific region mode
    if not entire_screen:
        logger.debug(f"Drawing lines: left_line={left_line}, start_line={start_line}, end_line={end_line}")
        annotated_image = line_annotator.annotate(frame=annotated_image, line_counter=start_line_zone)
        annotated_image = line_annotator.annotate(frame=annotated_image, line_counter=end_line_zone)
        if left_line > 0:
            logger.debug(f"Drawing left_line at x={left_line}")
            cv2.line(annotated_image, (int(left_line), 0), (int(left_line), height), (255, 0, 0), 2)
        else:
            logger.debug("left_line is 0, skipping left_line drawing")
    else:
        logger.debug("Entire screen mode: Skipping all line drawing")

    # Draw count summary
    annotated_image = draw_count_summary(annotated_image, counts, width)

    logger.debug(f"Image processed, entire_screen={entire_screen}, counts: {dict(counts)}")
    return {'image': annotated_image, 'counts': {k: {'count': v} for k, v in counts.items()}}

def process_video(video_path, left_line, start_line, end_line, distance, output_filename):
    """Process a video to count vehicles, zone-free for entire screen or with LineZone."""
    global progress
    with lock:
        entire_screen = progress['entire_screen']
    logger.debug(f"process_video: entire_screen={entire_screen}, left_line={left_line}, "
                 f"start_line={start_line}, end_line={end_line}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise ValueError('Failed to open video')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = initialize_tracker()
    tracker.reset()

    counts = defaultdict(lambda: {'count': set(), 'speeds': [], 'cross_times': {}})
    live_counts = defaultdict(int)
    frame_count = 0

    # Only create LineZone objects if not in entire screen mode
    start_line_zone = None
    end_line_zone = None
    if not entire_screen:
        logger.debug("Specific region mode: Creating LineZone objects")
        start_line_zone = LineZone(start=Point(*start_line[0]), end=Point(*start_line[1]))
        end_line_zone = LineZone(start=Point(*end_line[0]), end=Point(*end_line[1]))
    else:
        logger.debug("Entire screen mode: Skipping LineZone creation")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, device=device)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

        # Apply confidence threshold
        conf_mask = detections.confidence > CONFIDENCE_THRESHOLD
        detections = detections[conf_mask]

        detections = tracker.update_with_detections(detections=detections)

        if entire_screen:
            # Entire screen: Count all tracked detections without zones
            logger.debug("Entire screen mode: Counting all detections without zones")
            current_ids = detections.tracker_id.astype(int).tolist()
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
                class_name = model.model.names[class_id]
                counts[class_name]['count'].add(tracker_id)
                live_counts[class_name] = len(counts[class_name]['count'])
        else:
            # Specific region: Use LineZone and left_line
            logger.debug("Specific region mode: Using LineZone and left_line")
            # Filter by left_line
            left_mask = detections.xyxy[:, 0] >= left_line
            detections = detections[left_mask]

            # Update counts and speeds
            current_ids = detections.tracker_id.astype(int).tolist()
            for idx, (class_id, tracker_id, xyxy, confidence) in enumerate(
                zip(detections.class_id, detections.tracker_id, detections.xyxy, detections.confidence)
            ):
                class_name = model.model.names[class_id]
                x1, y1, x2, y2 = xyxy
                track_key = (class_name, tracker_id)

                # Check line crossings
                single_detection = sv.Detections(
                    xyxy=np.array([xyxy]),
                    confidence=np.array([confidence]),
                    class_id=np.array([class_id]),
                    tracker_id=np.array([tracker_id])
                )
                start_crossed = start_line_zone.trigger(detections=single_detection)
                end_crossed = end_line_zone.trigger(detections=single_detection)

                # Record crossing times
                if start_crossed and track_key not in counts[class_name]['cross_times']:
                    counts[class_name]['cross_times'][track_key] = {'start_frame': frame_count}
                    logger.debug(f"Start crossed: {track_key}, frame: {frame_count}")
                if end_crossed and track_key in counts[class_name]['cross_times']:
                    if 'end_frame' not in counts[class_name]['cross_times'][track_key]:
                        counts[class_name]['cross_times'][track_key]['end_frame'] = frame_count
                        logger.debug(f"End crossed: {track_key}, frame: {frame_count}")
                        start_frame = counts[class_name]['cross_times'][track_key].get('start_frame')
                        if start_frame is not None:
                            frames_taken = frame_count - start_frame
                            logger.debug(f"Frames taken: {frames_taken}, distance: {distance}, fps: {fps}")
                            if frames_taken > 0 and distance > 0:
                                speed_mps = distance / (frames_taken / fps)
                                speed_kmph = speed_mps * 3.6
                                counts[class_name]['speeds'].append(speed_kmph)
                                logger.debug(f"Speed calculated: {speed_kmph} kmph for {track_key}")
                            counts[class_name]['count'].add(tracker_id)
                            live_counts[class_name] = len(counts[class_name]['count'])
                            logger.debug(f"Updated count: {class_name}: {live_counts[class_name]}")

        # Annotate frame
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {conf:.2f}"
            for class_id, tracker_id, conf in zip(
                detections.class_id,
                detections.tracker_id,
                detections.confidence
            )
        ]
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # Draw lines only in specific region mode
        if not entire_screen:
            logger.debug(f"Drawing lines: left_line={left_line}, start_line={start_line}, end_line={end_line}")
            annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=start_line_zone)
            annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=end_line_zone)
            if left_line > 0:
                logger.debug(f"Drawing left_line at x={left_line}")
                cv2.line(annotated_frame, (int(left_line), 0), (int(left_line), height), (255, 0, 0), 2)
            else:
                logger.debug("left_line is 0, skipping left_line drawing")
        else:
            logger.debug("Entire screen mode: Skipping all line drawing")

        # Draw count summary
        annotated_frame = draw_count_summary(annotated_frame, live_counts, width, is_video=True)

        out.write(annotated_frame)
        frame_count += 1
        with lock:
            progress['progress'] = min(99, (frame_count / total_frames) * 100)
            progress['live_counts'] = dict(live_counts)

    # Compute final counts and average speeds
    final_counts = {}
    for cls in counts:
        count = len(counts[cls]['count'])
        if entire_screen:
            logger.debug(f"Entire screen mode: Excluding average_speed for {cls}")
            final_counts[cls] = {'count': count}
        else:
            avg_speed = (
                sum(counts[cls]['speeds']) / len(counts[cls]['speeds'])
                if counts[cls]['speeds'] else 0
            )
            logger.debug(f"Specific region mode: Including average_speed for {cls}: {avg_speed}")
            final_counts[cls] = {
                'count': count,
                'average_speed': avg_speed
            }
            logger.debug(f"Final for {cls}: count={count}, avg_speed={avg_speed}, speeds={counts[cls]['speeds']}")

    cap.release()
    logger.debug(f"Video processed, entire_screen={entire_screen}, counts: {final_counts}")
    return {'video_writer': out, 'counts': final_counts, 'live_counts': live_counts}

@app.route('/get-first-frame', methods=['POST'])
def get_first_frame():
    """Extract the first frame of a video for preview."""
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
        logger.error(f"Unsupported file extension: {file.filename}")
        return jsonify({'error': 'Unsupported video format. Use MP4, AVI, MOV, MKV, or WEBM'}), 400

    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False)
        file.save(temp_file.name)
        temp_file.close()

        logger.debug(f"Temporary file saved: {temp_file.name}")

        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {temp_file.name}")
            raise ValueError('Failed to open video file')

        for attempt in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.debug(f"Successfully extracted frame on attempt {attempt + 1}")
                cap.release()
                _, buffer = cv2.imencode('.jpg', frame)
                return send_file(
                    io.BytesIO(buffer),
                    mimetype='image/jpeg',
                    as_attachment=True,
                    download_name='first_frame.jpg'
                )
            time.sleep(0.1)

        cap.release()
        logger.error(f"Failed to extract frame from {temp_file.name} after 3 attempts")
        raise ValueError('Failed to extract frame from video')

    except Exception as e:
        logger.error(f"Error in get_first_frame: {str(e)}")
        return jsonify({'error': f'Failed to extract frame: {str(e)}'}), 500

    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.debug(f"Cleaned up temporary file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")

@app.route('/status')
def status():
    """Return processing status, progress, and live counts."""
    with lock:
        return jsonify(progress)

@app.route('/outputs/<filename>')
def serve_output(filename):
    """Serve processed media files."""
    try:
        return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/get-results/<filename>')
def get_results(filename):
    """Return processing results with summary."""
    with lock:
        if progress['status'] != 'complete':
            logger.error(f"Processing not complete for {filename}")
            return jsonify({'error': 'Processing not complete'}), 400

        summary_list = []
        entire_screen = progress.get('entire_screen', False)
        for vehicle_class, data in progress['summary'].items():
            entry = {
                'vehicle_class': vehicle_class,
                'count': data['count']
            }
            if not entire_screen and 'average_speed' in data:
                entry['average_speed'] = round(data['average_speed'], 1)
                logger.debug(f"Including average_speed in results for {vehicle_class}: {entry['average_speed']}")
            else:
                logger.debug(f"Excluding average_speed in results for {vehicle_class} (entire_screen={entire_screen})")
            summary_list.append(entry)

        response = {
            'processed_media_url': f'/outputs/{filename}',
            'download_url': f'/outputs/{filename}',
            'summary': summary_list
        }
        logger.debug(f"Results for {filename}: {response}")
        return jsonify(response)

@app.route('/download-summary', methods=['POST'])
def download_summary():
    """Generate and download the summary as Excel or PDF."""
    format_type = request.form.get('format')
    if format_type not in ['excel', 'pdf']:
        logger.error(f"Invalid format type: {format_type}")
        return jsonify({'error': 'Invalid format type'}), 400

    with lock:
        if progress['status'] != 'complete':
            logger.error("Processing not complete for summary download")
            return jsonify({'error': 'Processing not complete'}), 400

        summary_list = []
        entire_screen = progress.get('entire_screen', False)
        for vehicle_class, data in progress['summary'].items():
            entry = {
                'Vehicle Class': vehicle_class,
                'Count': data['count']
            }
            if not entire_screen and 'average_speed' in data:
                entry['Average Speed (km/h)'] = round(data['average_speed'], 1)
            summary_list.append(entry)

    # Create DataFrame
    df = pd.DataFrame(summary_list)

    if format_type == 'excel':
        # Generate Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Traffic Summary', index=False, startrow=2)
            # Access the workbook and worksheet to add the heading
            workbook = writer.book
            worksheet = workbook['Traffic Summary']
            worksheet['A1'] = 'Traffic Management Detections'
            worksheet['A1'].font = writer.book.add_format({'bold': True, 'font_size': 16})
            # Adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = max_length + 2
                worksheet.column_dimensions[column_letter].width = adjusted_width

        output.seek(0)
        logger.info("Generated Excel summary for download")
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='traffic_summary.xlsx'
        )

    else:  # format_type == 'pdf'
        # Generate PDF file
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        elements = []

        # Define styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        title_style.alignment = 1  # Center

        # Add heading
        elements.append(Paragraph("Traffic Management Detections", title_style))
        elements.append(Spacer(1, 12))

        # Prepare table data
        table_data = [list(df.columns)]  # Headers
        for _, row in df.iterrows():
            table_data.append(list(row))

        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(table)
        doc.build(elements)

        output.seek(0)
        logger.info("Generated PDF summary for download")
        return send_file(
            output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='traffic_summary.pdf'
        )

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
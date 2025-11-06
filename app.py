from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import json
import os
from datetime import datetime
import numpy as np
import threading
import time

app = Flask(__name__)

# Khởi tạo model YOLO
model = YOLO("model/yolo11n.pt")

# Danh sách các loại xe
vehicles = [
    "car", "motorcycle", "bicycle", "bus", "truck",
    "train", "airplane", "boat", "ship", "scooter",
    "van", "helicopter"
]

# Tạo thư mục lưu ảnh tai nạn
os.makedirs("accident_images", exist_ok=True)

# Đọc hoặc tạo zones.json mặc định
ZONES_FILE = "zones.json"
if not os.path.exists(ZONES_FILE):
    default_zones = {
        "North": {"x": 50, "y": 50, "width": 200, "height": 150, "color": "#FF0000"},
        "South": {"x": 50, "y": 350, "width": 200, "height": 150, "color": "#00FF00"},
        "East": {"x": 450, "y": 50, "width": 200, "height": 150, "color": "#0000FF"},
        "West": {"x": 450, "y": 350, "width": 200, "height": 150, "color": "#FFFF00"}
    }
    with open(ZONES_FILE, "w") as f:
        json.dump(default_zones, f)

# Biến global để lưu zones
zones = {}
with open(ZONES_FILE, "r") as f:
    zones = json.load(f)

# Biến để theo dõi tai nạn đã phát hiện (tránh lưu ảnh trùng)
detected_accidents = {}
accident_cooldown = 3  # Số giây trước khi có thể phát hiện lại tai nạn ở vị trí tương tự

# Video source (có thể thay đổi)
# 0 = webcam
# "path/to/video.mp4" = video file
# "rtsp://..." = IP camera
video_source = "TEST/vidieo/1900-151662242_small.mp4"  # Thay đổi theo đường dẫn video của bạn

# Kiểm tra video source
if isinstance(video_source, str) and not os.path.exists(video_source):
    print(f"⚠️  WARNING: Video file not found: {video_source}")
    print("📹 Switching to webcam (0)")
    video_source = 0

# Biến global để lưu frame hiện tại và thống kê
current_frame = None
current_stats = {"North": 0, "South": 0, "East": 0, "West": 0}
frame_lock = threading.Lock()


def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))


def check_bbox_in_zone(bbox, zone):
    """Kiểm tra bounding box có nằm trong hoặc giao với zone không"""
    x1, y1, x2, y2 = bbox
    zx, zy, zw, zh = zone["x"], zone["y"], zone["width"], zone["height"]
    
    # Kiểm tra giao nhau
    return not (x2 < zx or x1 > zx + zw or y2 < zy or y1 > zy + zh)


def check_collision(bbox1, bbox2, threshold=10):
    """Kiểm tra 2 bounding box có chạm nhau không với threshold"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Mở rộng bbox với threshold để phát hiện va chạm gần
    x1_1 -= threshold
    y1_1 -= threshold
    x2_1 += threshold
    y2_1 += threshold
    
    # Kiểm tra giao nhau
    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)


def get_accident_key(bbox1, bbox2):
    """Tạo key duy nhất cho cặp bounding box"""
    # Sắp xếp để đảm bảo (bbox1, bbox2) và (bbox2, bbox1) tạo cùng key
    box1_center = ((bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2)
    box2_center = ((bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2)
    
    if box1_center < box2_center:
        return f"{box1_center[0]}_{box1_center[1]}_{box2_center[0]}_{box2_center[1]}"
    else:
        return f"{box2_center[0]}_{box2_center[1]}_{box1_center[0]}_{box1_center[1]}"


def clean_old_accidents():
    """Xóa các accident cũ khỏi bộ nhớ"""
    current_time = time.time()
    keys_to_remove = []
    
    for key, timestamp in detected_accidents.items():
        if current_time - timestamp > accident_cooldown:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del detected_accidents[key]


def save_accident_image(frame, bbox1, bbox2):
    """Lưu ảnh tai nạn"""
    accident_key = get_accident_key(bbox1, bbox2)
    current_time = time.time()
    
    # Kiểm tra cooldown
    if accident_key in detected_accidents:
        if current_time - detected_accidents[accident_key] < accident_cooldown:
            return None
    
    detected_accidents[accident_key] = current_time
    
    # Crop vùng chứa 2 bounding box
    x_min = int(min(bbox1[0], bbox2[0]))
    y_min = int(min(bbox1[1], bbox2[1]))
    x_max = int(max(bbox1[2], bbox2[2]))
    y_max = int(max(bbox1[3], bbox2[3]))
    
    # Mở rộng vùng crop thêm 30 pixel
    x_min = max(0, x_min - 30)
    y_min = max(0, y_min - 30)
    x_max = min(frame.shape[1], x_max + 30)
    y_max = min(frame.shape[0], y_max + 30)
    
    cropped = frame[y_min:y_max, x_min:x_max]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"accident_images/accident_{timestamp}.jpg"
    cv2.imwrite(filename, cropped)
    
    # Cleanup
    clean_old_accidents()
    
    return filename


def process_video():
    """Xử lý video trong background thread"""
    global current_frame, current_stats
    
    print(f"📹 Attempting to open video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video source: {video_source}")
        return
    
    print("✅ Video source opened successfully")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️  End of video or read error, restarting...")
            # Nếu là video file, loop lại từ đầu
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Resize frame
        frame = cv2.resize(frame, (800, 600))
        
        # Process every frame (có thể skip frames để tăng tốc: frame_count % 2 == 0)
        frame_count += 1
        
        # Chạy YOLO detection
        results = model(frame, verbose=False, conf=0.3)
        
        # Lấy danh sách các bounding box của xe
        vehicle_boxes = []
        zone_counts = {"North": 0, "South": 0, "East": 0, "West": 0}
        accident_detected = False
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                if class_name in vehicles:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    vehicle_boxes.append([x1, y1, x2, y2])
                    
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Đếm xe trong từng zone
                    for zone_name, zone_data in zones.items():
                        if check_bbox_in_zone([x1, y1, x2, y2], zone_data):
                            zone_counts[zone_name] += 1
        
        # Kiểm tra va chạm giữa các xe
        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                if check_collision(vehicle_boxes[i], vehicle_boxes[j]):
                    accident_detected = True
                    filename = save_accident_image(frame, vehicle_boxes[i], vehicle_boxes[j])
                    
                    if filename:
                        print(f"⚠️ ACCIDENT DETECTED! Saved to {filename}")
                    
                    # Vẽ cảnh báo trên frame
                    cv2.putText(frame, "ACCIDENT DETECTED!", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.2, (0, 0, 255), 3)
                    
                    # Vẽ X đỏ quanh vùng va chạm
                    x_center = int((vehicle_boxes[i][0] + vehicle_boxes[j][0] + 
                                   vehicle_boxes[i][2] + vehicle_boxes[j][2]) / 4)
                    y_center = int((vehicle_boxes[i][1] + vehicle_boxes[j][1] + 
                                   vehicle_boxes[i][3] + vehicle_boxes[j][3]) / 4)
                    
                    # Vẽ dấu X
                    cv2.line(frame, (x_center - 40, y_center - 40), 
                            (x_center + 40, y_center + 40), (0, 0, 255), 4)
                    cv2.line(frame, (x_center + 40, y_center - 40), 
                            (x_center - 40, y_center + 40), (0, 0, 255), 4)
                    cv2.circle(frame, (x_center, y_center), 60, (0, 0, 255), 3)
        
        # Vẽ các zones
        for zone_name, zone_data in zones.items():
            x, y, w, h = zone_data["x"], zone_data["y"], zone_data["width"], zone_data["height"]
            color_bgr = hex_to_bgr(zone_data["color"])
            
            # Vẽ zone với độ trong suốt
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color_bgr, -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Vẽ viền zone
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 3)
            
            # Vẽ tên zone
            cv2.putText(frame, f"{zone_name}: {zone_counts[zone_name]}", 
                       (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color_bgr, 2)
        
        # Hiển thị số lượng ở góc phải
        y_offset = 30
        for zone_name in ["North", "South", "East", "West"]:
            count = zone_counts[zone_name]
            color_bgr = hex_to_bgr(zones[zone_name]["color"])
            
            text = f"{zone_name}: {count}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Vẽ background cho text
            cv2.rectangle(frame, 
                         (frame.shape[1] - text_width - 20, y_offset - text_height - 5),
                         (frame.shape[1] - 10, y_offset + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, text, 
                       (frame.shape[1] - text_width - 15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
            y_offset += 35
        
        # Lưu frame và stats
        with frame_lock:
            current_frame = frame.copy()
            current_stats = zone_counts.copy()
        
        # Giảm CPU usage
        time.sleep(0.01)
    
    cap.release()


def generate_frames():
    """Generator để stream video"""
    while True:
        # Đợi cho đến khi có frame
        timeout = 0
        while current_frame is None and timeout < 50:
            time.sleep(0.1)
            timeout += 1
        
        if current_frame is None:
            # Tạo frame đen với text error
            error_frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Waiting for video...", (250, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
        else:
            with frame_lock:
                frame = current_frame.copy()
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_zones')
def get_zones():
    return jsonify(zones)


@app.route('/get_stats')
def get_stats():
    """API để lấy thống kê real-time"""
    with frame_lock:
        return jsonify(current_stats)


@app.route('/update_zones', methods=['POST'])
def update_zones():
    global zones
    zones = request.json
    with open(ZONES_FILE, "w") as f:
        json.dump(zones, f, indent=2)
    return jsonify({"status": "success"})


if __name__ == '__main__':
    # Khởi động video processing trong background thread
    video_thread = threading.Thread(target=process_video, daemon=True)
    video_thread.start()
    
    print("🚀 Starting Traffic Monitoring System...")
    print("📹 Video processing started in background")
    print("🌐 Open browser: http://localhost:5000")
    print("⚠️  Press Ctrl+C to stop the server")
    
    # Chạy Flask server
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
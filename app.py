from ultralytics import YOLO
import cv2
import time

def main():
    # Load model YOLOv11n
    model = YOLO("yolo11n.pt")  # hoặc "yolo11s.pt" nếu muốn model chính xác hơn

    # Mở camera (0 = webcam mặc định)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return

    prev_time = 0

    print("Đang chạy YOLOv11 realtime — nhấn 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chạy dự đoán (inference)
        results = model(frame, verbose=False)

        # Vẽ bounding box trực tiếp
        annotated_frame = results[0].plot()

        # Tính FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị ảnh có box
        cv2.imshow("YOLOv11 Camera", annotated_frame)

        # Thoát bằng phím q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
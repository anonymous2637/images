import cv2
import numpy as np
import tensorflow as tf
import threading
import time
import json
# from db import save_to_db
# from excel import save_to_excel
from queue import Queue

# Load ROI coordinates from JSON
with open("roi_coordinates.json", "r") as f:
    roi_coordinates = json.load(f)

def is_inside_roi(x, y):

    if len(roi_coordinates) != 4:
        return False
    pts = np.array(roi_coordinates, np.int32)
    return cv2.pointPolygonTest(pts, (x, y), False) >= 0

def draw_roi(frame):
    if len(roi_coordinates) == 4:
        pts = np.array(roi_coordinates, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="D:/project/1/tflite_model/1.tflite",
    num_threads=4
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
PERSON_CLASS_ID = 0
SAVE_INTERVAL = 10

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                try:
                    self.ret, self.frame = self.cap.read()
                    if not self.ret:
                        time.sleep(0.1)
                except cv2.error as e:
                    print(f"[OpenCV Error] {e}")
                    time.sleep(0.1)
            else:
                print("[Stream Closed] Ending update loop.")
                break

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def preprocess(frame):
    input_shape = input_details[0]['shape'][1:3]
    image = cv2.resize(frame, tuple(input_shape))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if input_details[0]['dtype'] == np.uint8:
        image = np.expand_dims(image, axis=0).astype(np.uint8)
    else:
        image = np.expand_dims(image / 255.0, axis=0).astype(np.float32)

    return image


def run_inference(frame):
    input_data = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    return boxes, class_ids, scores

def apply_nms(boxes, scores, conf_thresh, iou_thresh):
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
    if isinstance(indices, np.ndarray):
        return indices.flatten().tolist()
    elif isinstance(indices, list):
        return [int(i) if isinstance(i, (np.integer, int)) else int(i[0]) for i in indices]
    else:
        return []

def process_output(boxes, class_ids, scores, frame_shape):
    h, w, _ = frame_shape
    results = []

    for i in range(len(scores)):
        if scores[i] > CONF_THRESHOLD and int(class_ids[i]) == PERSON_CLASS_ID:
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            results.append([x1, y1, x2, y2, float(scores[i])])

    return results

def io_worker(queue):
    while True:
        count = queue.get()
        if count is None:
            break
        # save_to_db(count)
        # save_to_excel(count)
        queue.task_done()

def process_frames():
    stream_url = "rtsp://admin:admin123@192.168.1.213/cam/realmonitor?channel=1&subtype=0&rtsp_transport=tcp&buffer_size=1024"
    stream = VideoStream(stream_url)
    io_queue = Queue()
    threading.Thread(target=io_worker, args=(io_queue,), daemon=True).start()

    prev_time = time.time()
    fps = 0.0
    detection_counter = 0

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        curr_time = time.time()
        delta_time = curr_time - prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 0.0
        prev_time = curr_time

        boxes, class_ids, scores = run_inference(frame)
        detections = process_output(boxes, class_ids, scores, frame.shape)


        boxes, scores = [], []
        for x1, y1, x2, y2, score in detections:
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)

        indices = apply_nms(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        final_detections = [detections[i] for i in indices]

        person_count = 0
        for x1, y1, x2, y2, score in final_detections:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if is_inside_roi(cx, cy):
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                label = f"person: {int(score * 100)}%"
                font_scale, thickness = 0.7, 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                label_x = x1
                label_y = max(y1 - 10, label_height + 10)
                cv2.rectangle(frame,
                              (label_x, label_y - label_height - baseline),
                              (label_x + label_width, label_y + baseline),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (label_x, label_y),
                            font, font_scale, (0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)

        if person_count > 0:
            detection_counter += 1
            if detection_counter >= SAVE_INTERVAL:
                io_queue.put(person_count)
                detection_counter = 0

        draw_roi(frame)

        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(frame, f"Persons: {person_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.namedWindow("Pedestrian Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Pedestrian Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Pedestrian Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    io_queue.put(None)
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frames()

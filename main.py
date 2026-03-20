import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import threading
import json
from datetime import datetime
import time

# 變數區
model = YOLO("yolo11n.pt")

video_source = None
is_paused = False
roi_points = []
drawing = False
drawing_line = False

line_start = None
line_end = None

track_history = {}
violated_ids = set()
CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

cap = None
fps_video = 30  # 預設值

#function區

def select_file():
    global video_source
    file_path = filedialog.askopenfilename(
        filetypes=[("MP4 files", "*.mp4")]
    )
    if file_path:
        video_source = file_path
        print("選擇影片:", file_path)

def open_rtsp_window():
    rtsp_window = tk.Toplevel(root)
    rtsp_window.title("輸入 RTSP")
    rtsp_window.geometry("300x120")

    tk.Label(rtsp_window, text="RTSP URL:").pack(pady=5)

    entry = tk.Entry(rtsp_window, width=40)
    entry.pack(pady=5)

    def confirm():
        global video_source
        rtsp = entry.get()
        if rtsp:
            video_source = rtsp
            print("RTSP:", rtsp)
        rtsp_window.destroy()  # 關閉小視窗

    tk.Button(rtsp_window, text="確認", command=confirm).pack(pady=10)

def open_manual_window():
    manual_window = tk.Toplevel(root)
    manual_window.title("使用說明")
    manual_window.geometry("400x300")  # 可調整大小

    # 可以放 Label 或 Text
    # 如果文字很多，用 Text 方便滾動
    text_widget = tk.Text(manual_window, wrap="word")
    text_widget.pack(expand=True, fill="both", padx=10, pady=10)

    # 你自己填寫說明內容
    manual_text = """
    YOLO 偵測系統使用說明：

    1. 點選「選擇 MP4」或「設定 RTSP」選擇影片來源
    2. 點選「開始偵測」開始即時車輛追蹤
    3. 可用「暫停」按鈕暫停或繼續播放
    4. 可使用「往前5秒」「往後5秒」調整影片位置
    5. 可用滑鼠左鍵畫 ROI(綠框)，右鍵畫行進方向線(藍線)
    6. 違規車輛會自動記錄在 violations.json
    """
    text_widget.insert("1.0", manual_text)
    text_widget.config(state="disabled")  # 不可編輯


def toggle_pause():
    global is_paused, btn_pause
    is_paused = not is_paused

    if is_paused:
        btn_pause.config(text="繼續")
    else:
        btn_pause.config(text="暫停")

def forward_5s():
    global cap, fps_video, is_paused

    if str(video_source).startswith("rtsp://"):
        print("RTSP 不支援跳轉")
        return

    if cap is None:
        return

    is_paused = True  # 先暫停

    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    new_frame = current_frame + int(fps_video * 5)

    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

    time.sleep(0.1)  #  等解碼穩定

    is_paused = False  #  再恢復
def backward_5s():
    global cap, fps_video, is_paused

    if str(video_source).startswith("rtsp://"):
        print("RTSP 不支援跳轉")
        return

    if cap is None:
        return
    
    is_paused = True  # 先暫停

    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    new_frame = current_frame - int(fps_video * 5)

    if new_frame < 0:
        new_frame = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

    time.sleep(0.1)  #  等解碼穩定

    is_paused = False  #  再恢復

def run_detection():
    global video_source, is_paused, cap

    if not video_source:
        print("請先選擇來源")
        return

    cap = cv2.VideoCapture(video_source)
    global fps_video

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video == 0:
        fps_video = 30  # 避免除以0

    if not cap.isOpened():
        print("無法開啟來源")
        return
    
    cv2.namedWindow("YOLO Detection")
    cv2.setMouseCallback("YOLO Detection", draw_roi)
    prev_time = 0

    while True:
        # 🔹 暫停控制
        if is_paused:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)


        if not ret:
            print("影片結束")
            break

        # YOLO偵測
        results = model.track(frame, persist=True)
        frame = results[0].plot()

        # ROI框
        if len(roi_points) == 2:
            cv2.rectangle(frame, roi_points[0], roi_points[1], (0,255,0), 2)

        # 方向線
        if line_start is not None and line_end is not None:
            cv2.arrowedLine(frame, line_start, line_end, (255,0,0), 2, tipLength=0.1)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        for r in results:
            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:
                cls = int(box.cls[0])
                label = CLASS_NAMES.get(cls, "unknown")

                # 只抓車類
                if cls not in [2, 3, 5, 7]:
                    continue

                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # ROI限制
                if len(roi_points) == 2:
                    x_min = min(roi_points[0][0], roi_points[1][0])
                    x_max = max(roi_points[0][0], roi_points[1][0])
                    y_min = min(roi_points[0][1], roi_points[1][1])
                    y_max = max(roi_points[0][1], roi_points[1][1])

                    if not (x_min < cx < x_max and y_min < cy < y_max):
                        continue

                # 紀錄軌跡
                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append((cx, cy))

                if len(track_history[track_id]) > 10:
                    track_history[track_id].pop(0)

                # 判斷方向
                pts = track_history[track_id]

                if len(pts) >= 2 and line_start is not None and line_end is not None:
                    line_vec = (line_end[0]-line_start[0], line_end[1]-line_start[1])
                    obj_vec = (pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1])

                    # 計算點積
                    dot_product = line_vec[0]*obj_vec[0] + line_vec[1]*obj_vec[1]
                    
                    # 如果點積 < 0 表示方向相反 → 逆向
                    if dot_product < 0:
                        cv2.putText(frame, "WRONG WAY!", (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                        if track_id not in violated_ids:
                            violated_ids.add(track_id)
                            save_violation(track_id, label)

            # FPS 計算
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def start_detection_thread():
    t = threading.Thread(target=run_detection)
    t.daemon = True
    t.start()

def draw_roi(event, x, y, flags, param):
    global roi_points, drawing
    global line_start, line_end, drawing_line

    # 左鍵按下：開始畫框
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if len(roi_points) == 1:
            roi_points.append((x, y))
        else:
            roi_points[1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    # 右鍵按下：開始畫線
    if event == cv2.EVENT_RBUTTONDOWN:
        line_start = (x, y)
        line_end = (x, y)
        drawing_line = True

    # 滑鼠移動：更新線終點
    elif event == cv2.EVENT_MOUSEMOVE and drawing_line:
        line_end = (x, y)

    # 右鍵放開：完成畫線
    elif event == cv2.EVENT_RBUTTONUP:
        line_end = (x, y)
        drawing_line = False

def save_violation(track_id, label):
    data = {
        "event": "wrong_way",
        "track_id": track_id,
        "label": label,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open("violations.json", "r", encoding="utf-8") as f:
            file_data = json.load(f)
    except:
        file_data = []

    file_data.append(data)

    with open("violations.json", "w", encoding="utf-8") as f:
        json.dump(file_data, f, indent=4, ensure_ascii=False)

    print(f"已記錄違規: {track_id}, {label}")
    
# GUI
root = tk.Tk()
root.title("YOLO 偵測系統")
root.geometry("400x300")

btn_manual = tk.Button(root, text="使用說明", command=open_manual_window)
btn_manual.pack(pady=10)

top_frame = tk.Frame(root)
top_frame.pack(pady=10)

btn_file = tk.Button(top_frame, text="選擇 MP4", command=select_file)
btn_file.pack(side=tk.LEFT, padx=5)

label_or = tk.Label(top_frame, text="或")
label_or.pack(side=tk.LEFT, padx=5)

btn_rtsp = tk.Button(top_frame, text="設定 RTSP", command=open_rtsp_window)
btn_rtsp.pack(side=tk.LEFT, padx=5)

btn_start = tk.Button(root, text="開始偵測", command=start_detection_thread)
btn_start.pack(pady=10)

btn_pause = tk.Button(root, text="暫停", command=toggle_pause)
btn_pause.pack(pady=10)

btn_back = tk.Button(root, text="往前5秒", command=backward_5s)
btn_back.pack(pady=5)

btn_forward = tk.Button(root, text="往後5秒", command=forward_5s)
btn_forward.pack(pady=5)


root.mainloop()
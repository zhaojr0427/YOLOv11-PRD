from ultralytics import YOLO
import cv2

# 1. 加载训练好的模型
model = YOLO(r'/tmp/pycharm_project_413/ultralytics/cfg/models/4 0.852/weights/best.pt')  # 替换为你的权重文件路径

# 2. 设置输入源（图片、视频、摄像头或文件夹）
source = r'/tmp/pycharm_project_413/1/test/images'  # 可以是图片、视频、摄像头（0）或文件夹路径

# 3. 进行推理
results = model.predict(
    source=source,  # 输入源
    conf=0.5,       # 置信度阈值
    iou=0.45,       # IOU阈值
    show=True,      # 显示结果
    save=True,      # 保存结果
    save_txt=False, # 是否保存检测结果为txt文件
    save_conf=True, # 保存结果时是否包含置信度
    save_crop=False, # 是否保存裁剪的检测目标
    show_labels=True, # 显示标签
    show_conf=True,  # 显示置信度
    line_width=2,    # 边界框线宽
    project='/tmp/pycharm_project_413/runs/detect',
    name='my_experiment'
)

# 4. 处理结果（可选）
for result in results:
    # 获取边界框信息
    boxes = result.boxes  # 边界框对象
    for box in boxes:
        # 获取边界框坐标（xyxy格式）
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        # 获取置信度
        confidence = box.conf[0].item()
        # 获取类别ID和名称
        class_id = box.cls[0].item()
        class_name = model.names[class_id]

        print(f"Detected {class_name} with confidence {confidence:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# 5. 如果是视频或摄像头，逐帧显示
if source == '0' or source.endswith('.mp4') or source.endswith('.avi'):
    cap = cv2.VideoCapture(source if source != '0' else 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 对每一帧进行推理
        results = model.predict(frame, conf=0.5, iou=0.45)

        # 显示结果
        for result in results:
            annotated_frame = result.plot()  # 绘制边界框和标签
            cv2.imshow('YOLOv8 Detection', annotated_frame)

        # 按下 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
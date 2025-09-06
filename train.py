# 模型配置文件
#model_yaml_path = "/tmp/pycharm_project_413/ultralytics/cfg/models/11/yolo11_4.yaml"
model_yaml_path = "/tmp/pycharm_project_413/ultralytics/cfg/models/models_test/yolo_plus3.yaml"
# 数据集配置文件
data_yaml_path = r'/tmp/pycharm_project_413/datasets/5/archive/weldqualityinspectionv9/data.yaml'
#data_yaml_path = r'/tmp/pycharm_project_413/datasets/1/data.yaml'
# 预训练模型
pre_model_name = '/tmp/pycharm_project_413/yolo11s.pt'
#pre_model_name = r'/tmp/pycharm_project_413/ultralytics/cfg/models/4 0.852/weights/best.pt'
# python /tmp/pycharm_project_413/train.py ... 2>&1 | tee train_log_yolo11_2.txt

#plus_4 11m ShapeIou

import warnings


warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model_yaml_path)

    model.load(pre_model_name)  # 是否加载预训练权重
    model.train(data=data_yaml_path,
                cache=False,
                imgsz=640,
                epochs=100,
                single_cls=False,  # 是否是单类别检测
                batch=16 ,
                close_mosaic=0,
                workers=16,
                device='0',
                optimizer='SGD',
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                augment=True,  # 设置是否使用数据增，True表示使用纹据增强
                #lr0=0.01,
                project='/tmp/pycharm_project_413/runs/train5',
                name='yolo11',
                )


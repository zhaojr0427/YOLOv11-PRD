YOLO-PRD: Ablation Study Codebase

This repository contains the source code for the paper YOLO-PRD, an improved YOLOv11-based model designed for steel weld defect detection.
The project mainly focuses on ablation experiments that investigate the contribution of different architectural modifications and dataset switching strategies.

🔑 Key Components of the Ablation Study

The ablation experiments are organized into four main parts, each addressing model structure adjustments or dataset variations:

Model Configuration Files (YOLOv11 Baseline & Improved Versions)

Path: ultralytics/cfg/models/11/

Purpose: Contains configuration files for both the baseline YOLOv11 and the proposed YOLOv11-PRD models.
These configs define architectural changes such as:

Adding the P2 detection head to improve small-object detection.

Introducing DynamicHead for adaptive feature aggregation.

Other incremental modifications tested during ablation.

Source Domain Dataset

Path: datasets/1/

Purpose: Serves as the dataset for pre-training in cross-domain transfer learning experiments.
Models trained here are later adapted to the target domain.

Target Domain Dataset

Path: datasets/5/

Purpose: Used for fine-tuning and final evaluation.
This dataset represents the actual downstream weld defect detection task.

Ablation Code Modification Points

Location: ultralytics/cfg/models/11/

Purpose: Different configuration files within this folder represent individual ablation settings.
For example, selectively disabling or replacing modules (P2 head, DynamicHead, transfer learning) allows us to verify the independent contribution of each component."# YOLOv11-PRD" 
"# YOLOv11-PRD" 
# YOLOv11-PRD

# C3 Intelligent Perception

面向全国海洋航行器设计与制作大赛 C3 智能感知赛题的三模态目标检测工程。

本工程基于已经整理完成的三模态框标注检测数据集：

- `RGB` 可见光
- `IR` 红外
- `Radar` 雷达点迹投影图
- `YOLO` 风格多模态融合检测器

## 1. 工程目标

- 提供去雾增强 + RGB/IR/Radar 多模态融合检测完整代码
- 提供训练、验证、推理、可视化、报告模板
- 面向比赛场景兼顾精度、鲁棒性与实时性

## 2. 目录结构

```text
ship_game/
├─ configs/
│  └─ c3_multimodal_yolo.yaml
├─ docs/
│  └─ C3_智能感知_算法方案报告.md
├─ processed/
│  └─ c3_bbox_dataset/
├─ src/
│  ├─ data/
│  ├─ engine/
│  ├─ models/
│  └─ utils/
├─ tools/
│  └─ train_server.py
├─ train.py
├─ infer.py
└─ requirements.txt
```

## 3. 服务器运行流程

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

### 3.2 数据集结构

```text
processed/c3_bbox_dataset/
├─ train/
│  ├─ rgb/
│  ├─ ir/
│  ├─ radar/
│  └─ labels/
├─ val/
├─ test/
├─ dataset_meta.json
└─ dataset.yaml
```

### 3.3 训练

```bash
python train.py --config configs/c3_multimodal_yolo.yaml
```

### 3.4 验证

```bash
python train.py --config configs/c3_multimodal_yolo.yaml --evaluate-only --checkpoint runs/c3_multimodal_yolo/best.pt
```

### 3.5 推理

```bash
python infer.py \
  --config configs/c3_multimodal_yolo.yaml \
  --checkpoint runs/c3_multimodal_yolo/best.pt \
  --split test \
  --save-dir runs/c3_multimodal_yolo/infer_test
```

## 4. 关键说明

- 本工程的训练、验证、推理统一读取 `processed/c3_bbox_dataset`
- 每个样本由 `rgb / ir / radar / labels` 四部分组成，文件名一一对应
- 标签格式为标准 `YOLO txt`

## 5. 方案概述

- `RGB` 分支带轻量去雾增强模块
- `IR` 分支提供弱光与夜间轮廓补充
- `Radar` 分支将点迹 CSV 投影为图像对齐多通道特征图
- `QG-CMF` 质量感知门控融合模块自适应抑制失效模态
- `YOLO` 风格 FPN/PAN 多尺度检测头兼顾精度与实时性

## 6. 文档

完整比赛报告已保存至：

- [docs/C3_智能感知_算法方案报告.md](c:\Users\lenovo\Desktop\ship_game\docs\C3_智能感知_算法方案报告.md)

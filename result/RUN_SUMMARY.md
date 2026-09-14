# YOLOv5 Box Detection — Run Evidence Summary

วันที่รวบรวมหลักฐาน: 2026-09-14

## สรุปผล

รวบรวมผลการรันสำคัญ 4 รอบ โดยทุก run ใช้ `yolov5n.pt`, image size 640, batch size 2 และ CPU เนื่องจาก GPU + AMP configuration ที่ใช้ก่อนหน้านี้ให้ค่า loss เป็น NaN

| Run | Dataset (train/val/test) | Epochs | Best epoch | Precision | Recall | mAP50 | mAP50-95 | สถานะ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `smoke_1epoch_tensorboard` | 14/3/3 | 1 | 0 | 0.0051 | 0.1667 | 0.0037 | 0.0019 | pipeline และ TensorBoard tags ผ่าน |
| `baseline_5epochs` | 14/3/3 | 5 | 0 | 0.0051 | 0.1667 | 0.0037 | 0.0019 | รันผ่าน แต่ยังไม่ converge |
| `dataset350_smoke_retry` | 250/50/50 | 1 | 0 | 0.1959 | 0.2172 | 0.1503 | 0.0433 | expanded-dataset smoke test ผ่าน |
| `dataset350_baseline30` | 250/50/50 | 30 | 28 | 0.9965 | 1.0000 | 0.9950 | 0.7999 | ผลดีที่สุด |

ค่าของแต่ละ run เลือกจาก epoch ที่มี `metrics/mAP_0.5:0.95` สูงที่สุดใน `results.csv` ไม่ใช่เพียง epoch สุดท้าย รายละเอียดแบบ machine-readable อยู่ใน `metrics.csv`

## ข้อสรุปสำคัญ

- การเพิ่ม dataset จาก 20 เป็น 350 ภาพทำให้ผล validation ดีขึ้นอย่างชัดเจน
- `dataset350_baseline30` ให้ผลดีที่สุดที่ epoch 28: precision 0.99648, recall 1.0, mAP50 0.995 และ mAP50-95 0.79994
- validation มี 50 ภาพ จึงน่าเชื่อถือกว่ารอบเดิมที่มีเพียง 3 ภาพ แต่ภาพมาจากลำดับเฟรมที่สัมพันธ์กัน จึงยังไม่ควรสรุป generalization ไปยังสภาพแวดล้อมอื่นโดยไม่มี external test set
- ควรใช้ `yolov5/runs/f09_box/dataset350_baseline30/weights/best.pt` สำหรับ evaluation และ inference ขั้นถัดไป

## ภาพหลักฐาน

### ภาพแคป TensorBoard

- [TensorBoard — completed runs overview](screenshots/tensorboard_all_runs_scalars.png)
- [TensorBoard — dataset350 baseline 30 epochs](screenshots/tensorboard_dataset350_baseline30.png)

หมายเหตุ: overview อาจแสดง event directories จาก smoke attempts ที่หยุดก่อน training (`dataset350_smoke` และ `dataset350_smoke2`) แต่ตารางสรุปนับเฉพาะ 4 runs ที่มี `results.csv` สมบูรณ์

### กราฟราย run

- [Smoke test 1 epoch](screenshots/01_smoke_1epoch_results.png)
- [Baseline 5 epochs](screenshots/02_baseline_5epochs_results.png)
- [Expanded dataset smoke test](screenshots/03_dataset350_smoke_results.png)
- [Expanded dataset baseline 30 epochs](screenshots/04_dataset350_baseline30_results.png)

### หลักฐานของโมเดลที่ดีที่สุด

- [Confusion matrix](screenshots/05_dataset350_baseline30_confusion_matrix.png)
- [Precision–Recall curve](screenshots/06_dataset350_baseline30_PR_curve.png)
- [F1 curve](screenshots/07_dataset350_baseline30_F1_curve.png)
- [Dataset label distribution](screenshots/08_dataset350_labels.jpg)
- [Validation predictions batch 1](screenshots/09_dataset350_validation_predictions_1.jpg)
- [Validation predictions batch 2](screenshots/10_dataset350_validation_predictions_2.jpg)
- [Validation predictions batch 3](screenshots/11_dataset350_validation_predictions_3.jpg)

## Training configuration

ค่าร่วมของทั้ง 4 runs:

```text
weights: yolov5n.pt
batch_size: 2
imgsz: 640
device: cpu
data: ../datasets/f09_box/data.yaml
```

จำนวน epochs แตกต่างกันตามวัตถุประสงค์ของ smoke test และ baseline ดังที่ระบุในตาราง

## ที่มาของข้อมูล

- Metrics: `yolov5/runs/f09_box/<run>/results.csv`
- Settings: `yolov5/runs/f09_box/<run>/opt.yaml`
- TensorBoard: event files ภายในแต่ละ run
- Curves และ prediction evidence: artifacts ที่ YOLOv5 สร้างใน `dataset350_baseline30`

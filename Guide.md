---
type: assignment
instructor: P.Fern
date: 2026-08-31
due:
status: complete
tags: [cv, tool, pytorch]
aliases: [F09 Part 1 YOLO-Box]
---

# Part 1 — YOLOv5 Box Detection

> [!success] สถานะการทดลองล่าสุด — เสร็จเมื่อ 2026-09-15
> รันใหม่ครบ 4 ขนาด 30/60/150/350 ที่ 30 epochs ด้วย YOLOv5n และ seed 2569 แล้ว ตรวจ TensorBoard iteration tags ครบ 330/630/1,620/3,750 จุดโดยไม่มี NaN/Inf ประเมิน `best.pt` ทุกตัวบน common test set 50 ภาพเดียวกัน และเก็บสรุป ตาราง metrics, TensorBoard UI screenshot, กราฟ และ inference 3 ภาพไว้ใน `result/` ผลหลักดูที่ `result/RUN_SUMMARY.md`
> ภาพแยกแต่ละ run อยู่ใน `result/screenshots/per_run/` โดยมีทั้ง dashboard จาก TensorBoard event data และกราฟ `results.png` ต้นฉบับของ YOLOv5 สำหรับขนาด 30/60/150/350
> รายงานฉบับพร้อมส่งที่รวมตาราง คำอธิบาย และภาพทั้งหมดอยู่ที่ `result/FINAL_REPORT.md`; weight distribution ถูกพักไว้เพราะเป็นส่วนเสริมที่อาจารย์ไม่ได้กำหนด

> แผนนี้ขยาย Part 1 จาก [[P.Fern/Assignments/F09_31-08-69/instruction]] เป้าหมายคือ train YOLOv5n ตรวจจับ bounding box ด้วยภาพ 20 รูป และเก็บกราฟที่โจทย์กำหนดใน TensorBoard

> [!note] รอบผลที่ใช้ส่ง — 2026-09-15
> ผลเก่าถูกล้างก่อนเริ่มรอบนี้ ผลหลักฐานปัจจุบันมาจาก run ใหม่ `size30_e30_seed2569`, `size60_e30_seed2569`, `size150_e30_seed2569` และ `size350_e30_seed2569` เท่านั้น

## สถานะเทียบกับสไลด์ต้นฉบับ — ตรวจเมื่อ 2026-09-15

ข้อกำหนดอ้างอิงอยู่ใน `Slides-n5-241-353 TensorBoard 2569.pdf` หน้า 17–19

| ข้อกำหนด | หลักฐานปัจจุบัน | สถานะ |
| --- | --- | --- |
| ใช้ pretrained YOLOv5 รุ่น n หรือ s | ใช้ `yolov5n.pt` | เสร็จ |
| ปรับ `data.yaml` ให้ตรงกับ dataset | `datasets/f09_box/data.yaml` ใช้ portable path และมี 3 classes | เสร็จ |
| คำนวณ loss อย่างน้อย 30 iterations | 330/630/1,620/3,750 iterations | เสร็จ |
| มีโมเดลที่ train แล้ว | ทุก run มี `best.pt` และ `last.pt` | เสร็จ |
| total loss และ LR เทียบกับ iteration | event counts ครบและค่าทั้งหมด finite | เสร็จ |
| box/objectness/classification loss เทียบกับ iteration | Step 6A บันทึกครบทุก iteration | เสร็จ |
| evaluation metrics ใน TensorBoard | event 30 epochs และกราฟใน `result/screenshots/` | เสร็จ |
| evaluation บน test split | common test ครบ 4 โมเดลบน 50 ภาพ/351 objects | เสร็จ |
| inference ตัวอย่างจาก test split 3 ภาพ | `05_inference_first.jpg`–`07_inference_last.jpg` | เสร็จ |
| ภาพ TensorBoard สำหรับรายงาน | UI screenshot และกราฟ export อยู่ใน `result/screenshots/` | เสร็จ |
| คำตอบว่า network converge อย่างไร | บันทึกข้อสรุปใน `result/RUN_SUMMARY.md` | เสร็จ |

สไลด์หน้า 17 ระบุ split เป็น train 100/test 100 แต่ dataset ปัจจุบันเป็น train 250/validation 50/test 50 ก่อนส่งต้องทำอย่างใดอย่างหนึ่งให้มีหลักฐานชัดเจน:

1. ใช้ split ปัจจุบันและเขียนในรายงานว่าเพิ่ม validation set และใช้ข้อมูลมากกว่าตัวอย่างในสไลด์ โดย test 50 รูปไม่ถูกใช้เลือกโมเดล หรือ
2. หากอาจารย์กำหนดจำนวน 100/100 แบบตายตัว ให้สร้าง dataset สำหรับส่งใหม่และ train/evaluate ใหม่ทั้งหมดด้วย split นั้น

อย่าผสมผลจากคนละ split ในตารางหรือข้อสรุปเดียวกัน

## Dataset subsets สำหรับทดลองขนาดข้อมูล — สร้างเมื่อ 2026-09-15

สร้างจาก `datasets/f09_box` ด้วย seed `2569` แบบ deterministic และ nested ภายในแต่ละ split ชุดเล็กเป็นส่วนหนึ่งของชุดใหญ่ จึงสร้างซ้ำและเปรียบเทียบลำดับการเพิ่มข้อมูลได้

| Dataset | Train | Validation | Test | Iterations/epoch เมื่อ batch=2 | 30 epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `datasets/f09_box_30` | 22 | 4 | 4 | 11 | 330 |
| `datasets/f09_box_60` | 42 | 9 | 9 | 21 | 630 |
| `datasets/f09_box_150` | 108 | 21 | 21 | 54 | 1,620 |
| `datasets/f09_box_350` | 250 | 50 | 50 | 125 | 3,750 |

แต่ละโฟลเดอร์มี `images/{train,val,test}`, `labels/{train,val,test}`, `manifest.csv`, `data.yaml` และ `README.md` ครบ สคริปต์สร้างซ้ำอยู่ที่ `create_f09_box_subsets.py` โดยตั้งใจให้หยุดพร้อม error หากโฟลเดอร์ปลายทางมีอยู่แล้ว เพื่อไม่เขียนทับ dataset โดยไม่ตั้งใจ

จำนวน 30/60/150/350 ในชื่อนี้หมายถึงจำนวนภาพรวมทุก split หากต้องการเปรียบเทียบความสามารถของโมเดลอย่างยุติธรรม ให้รายงาน validation ของแต่ละชุดแยกกัน และประเมิน `best.pt` ทุกตัวซ้ำบน test 50 ภาพชุดเดียวจาก `f09_box_350` เพราะคะแนนจาก test 4, 9, 21 และ 50 ภาพเปรียบเทียบตรงกันไม่ได้

run เก่าที่สร้างก่อน Step 6A ยังใช้ weights หรือ epoch-level metrics เดิมได้ แต่ไม่สามารถเติม component-loss history ราย iteration ย้อนหลังได้ หากต้องการกราฟตามสไลด์และเปรียบเทียบทั้งสี่ขนาด ต้อง train แต่ละชุดใหม่หลังใช้โค้ด Step 6A โดยคง model, seed, image size, batch size, epochs และ hyperparameters ให้เหมือนกัน

- [x] สร้าง dataset 30/60/150/350 ครบ
- [x] จำนวน image/label ตรงกันทุก split
- [x] ตรวจว่า dataset เล็ก nested อยู่ใน dataset ใหญ่
- [x] `data.yaml` ทั้งสี่ชุด resolve ผ่าน YOLOv5
- [x] รัน training ชุดเปรียบเทียบ 30/60/150/350 ครบ 30 epochs เมื่อ 2026-09-15

## แผนการทดลอง 4 ขนาด: 30/60/150/350

### เป้าหมายและตัวแปรควบคุม

ทดลองว่าปริมาณข้อมูลสัมพันธ์กับ loss, convergence และ test metrics อย่างไร โดยเปลี่ยนเฉพาะ dataset และคงค่าต่อไปนี้เหมือนกันทุก run:

```text
model: yolov5n.pt
epochs: 30
batch_size: 2
imgsz: 640
device: cpu
workers: 0
seed: 2569
hyperparameters: data/hyps/hyp.scratch-low.yaml (default)
```

ใช้ 30 epochs เท่ากันเพื่อให้แต่ละโมเดลเห็นข้อมูลของตนจำนวนรอบเท่ากัน จำนวน optimizer iterations จะแตกต่างตามจำนวน train images และต้องรายงานความแตกต่างนี้ด้วย ห้ามใช้ `--exist-ok` เพราะอาจนำ event จากหลาย execution มาปนกัน

### ลำดับการทำงาน

1. ตรวจว่า `yolov5/runs/` ยังไม่มีผลเก่า และ `.venv`, `yolov5n.pt`, dataset YAML ทั้งสี่ชุดพร้อม
2. รัน preflight ด้วย dataset 30 เพียง 1 epoch
3. ตรวจว่า loss ไม่มี NaN และ TensorBoard มี iteration tags ทั้งห้าครบ 11 steps
4. รัน full training ตามลำดับ 30 → 60 → 150 → 350 เพื่อพบปัญหาเร็วและประเมินเวลาจากชุดเล็กก่อน
5. ตรวจแต่ละ run ทันทีหลังจบ ห้ามเริ่ม run ถัดไปหาก event tags ไม่ครบหรือมี NaN
6. ประเมิน `best.pt` ทั้งสี่ตัวบน common test set 50 ภาพเดียวกันจาก `f09_box_350`
7. เปิด TensorBoard จาก parent directory เดียวเพื่อเปรียบเทียบทั้งสี่ runs
8. สร้างตารางสรุปและเลือกโมเดล final จาก common-test mAP50-95 โดยใช้ qualitative inference ประกอบ

### Phase A — Preflight

ทำจากโฟลเดอร์ `yolov5/`:

```powershell
python train.py --img 640 --batch-size 2 --epochs 1 --data '..\datasets\f09_box_30\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --seed 2569 --project runs\f09_box_compare --name preflight_size30
```

เกณฑ์ผ่าน:

- `results.csv` มี 1 epoch
- มี `best.pt`, `last.pt` และ TensorBoard event file
- `train/box_loss_iter`, `train/obj_loss_iter`, `train/cls_loss_iter`, `train/total_loss`, `train/learning_rate` มีอย่างละ 11 steps ตั้งแต่ 0–10
- ทุก loss เป็น finite และ `total_loss` เท่ากับผลรวมสาม component losses ภายใน floating-point tolerance

### Phase B — Full training

รันทีละคำสั่ง รอให้คำสั่งก่อนหน้าจบและตรวจผ่านก่อนเริ่มคำสั่งถัดไป:

```powershell
python train.py --img 640 --batch-size 2 --epochs 30 --data '..\datasets\f09_box_30\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --seed 2569 --project runs\f09_box_compare --name size30_e30_seed2569

python train.py --img 640 --batch-size 2 --epochs 30 --data '..\datasets\f09_box_60\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --seed 2569 --project runs\f09_box_compare --name size60_e30_seed2569

python train.py --img 640 --batch-size 2 --epochs 30 --data '..\datasets\f09_box_150\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --seed 2569 --project runs\f09_box_compare --name size150_e30_seed2569

python train.py --img 640 --batch-size 2 --epochs 30 --data '..\datasets\f09_box_350\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --seed 2569 --project runs\f09_box_compare --name size350_e30_seed2569
```

จำนวน iteration tags ที่คาดหวัง:

| Run | Steps ที่ต้องมีต่อ tag | ช่วง global step |
| --- | ---: | ---: |
| `size30_e30_seed2569` | 330 | 0–329 |
| `size60_e30_seed2569` | 630 | 0–629 |
| `size150_e30_seed2569` | 1,620 | 0–1,619 |
| `size350_e30_seed2569` | 3,750 | 0–3,749 |

เวลาบน CPU จากความเร็วที่เคยวัดเป็นเพียงค่าประมาณ: size 30 ราว 3–5 นาที, size 60 ราว 6–10 นาที, size 150 ราว 12–18 นาที และ size 350 ราว 25–35 นาที เวลาจริงขึ้นกับโหลดของเครื่อง

### Phase C — ตรวจผลหลังแต่ละ run

ตรวจสิ่งต่อไปนี้ก่อนรันชุดถัดไป:

- process exit code เป็น 0 และ `results.csv` มี 30 แถวสำหรับ epochs 0–29 โดยไม่ซ้ำ
- ไม่มี NaN/Inf ใน loss หรือ metrics
- มี `weights/best.pt`, `weights/last.pt`, `opt.yaml` และ event file อย่างละชุด
- iteration tags ทั้งห้ามีจำนวนตามตาราง Phase B และมี step ต่อเนื่อง
- epoch-level tags มี 30 steps: component losses, validation losses, precision, recall, mAP50 และ mAP50-95
- จดเวลาที่ใช้, best epoch และ validation metrics ที่ best epoch

### Phase D — Common test evaluation

validation split ภายใน dataset มีขนาดต่างกัน จึงใช้เพื่อดูการเรียนรู้ของ run นั้นเท่านั้น การเปรียบเทียบสุดท้ายต้องประเมินทุกโมเดลบน test 50 ภาพเดียวกันจาก `f09_box_350`:

```powershell
python val.py --weights 'runs\f09_box_compare\size30_e30_seed2569\weights\best.pt' --data '..\datasets\f09_box_350\data.yaml' --task test --img 640 --batch-size 2 --device cpu --workers 0 --project runs\f09_box_compare_test --name size30_common_test
python val.py --weights 'runs\f09_box_compare\size60_e30_seed2569\weights\best.pt' --data '..\datasets\f09_box_350\data.yaml' --task test --img 640 --batch-size 2 --device cpu --workers 0 --project runs\f09_box_compare_test --name size60_common_test
python val.py --weights 'runs\f09_box_compare\size150_e30_seed2569\weights\best.pt' --data '..\datasets\f09_box_350\data.yaml' --task test --img 640 --batch-size 2 --device cpu --workers 0 --project runs\f09_box_compare_test --name size150_common_test
python val.py --weights 'runs\f09_box_compare\size350_e30_seed2569\weights\best.pt' --data '..\datasets\f09_box_350\data.yaml' --task test --img 640 --batch-size 2 --device cpu --workers 0 --project runs\f09_box_compare_test --name size350_common_test
```

บันทึก precision, recall, mAP50 และ mAP50-95 จากแถว `all` ของแต่ละ run ห้ามเปรียบเทียบคะแนน test 4/9/21/50 ภาพจาก YAML แต่ละชุดเป็นผลหลัก

### Phase E — TensorBoard และตารางสรุป

```powershell
tensorboard --logdir runs\f09_box_compare --port 6006
```

แคปกราฟโดยเลือก runs ทั้งสี่และตั้ง Horizontal Axis เป็น Step แยกภาพสำหรับ:

1. `train/total_loss`
2. `train/box_loss_iter`
3. `train/obj_loss_iter`
4. `train/cls_loss_iter`
5. `train/learning_rate`
6. epoch-level validation losses และ evaluation metrics

ตารางสรุปสุดท้าย:

| Dataset | Train/Val/Test | Iterations | เวลา | Best epoch | Common-test P | R | mAP50 | mAP50-95 | หมายเหตุ |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 30 | 22/4/4 | 330 | 28 | 0.141 | 0.318 | 0.212 | 0.0706 | ผ่าน | ต่ำสุด |
| 60 | 42/9/9 | 630 | 18 | 0.365 | 0.708 | 0.546 | 0.2490 | ผ่าน | ดีขึ้นจาก 30 |
| 150 | 108/21/21 | 1,620 | 28 | 0.886 | 0.987 | 0.984 | 0.6520 | ผ่าน | เพิ่มขึ้นมาก |
| 350 | 250/50/50 | 3,750 | 29 | 0.990 | 0.990 | 0.989 | 0.7250 | ผ่าน | final model |

เกณฑ์เลือก final model คือ common-test mAP50-95 สูงสุด โดยตรวจว่า precision/recall ไม่ผิดปกติ, loss ไม่มี NaN, validation ไม่แย่ลงต่อเนื่อง และ inference บน test images สมเหตุสมผล หากคะแนนใกล้กันให้เลือกโมเดลที่เสถียรกว่าและอธิบายข้อจำกัดจากข้อมูลที่เป็นลำดับเฟรม

- [x] Phase A preflight ผ่าน: 11 steps ต่อ tag, finite และ total loss ตรงกับผลรวม component losses
- [x] Full training 30/60/150/350 ผ่านครบ
- [x] Common test evaluation ครบสี่โมเดลบน 50 ภาพ/351 objects
- [x] TensorBoard screenshots ครบห้า training tags และ evaluation metrics
- [x] ตารางเปรียบเทียบ ข้อสรุป และ inference 3 ภาพเสร็จใน `result/`

## ผลลัพธ์ที่ต้องได้

- YOLO detection dataset ที่แปลงจาก CVAT XML แล้ว
- โมเดล `best.pt` จาก pretrained `yolov5n.pt`
- TensorBoard plots ของ total, objectness, classification, box regression loss และ learning rate
- evaluation metrics plot
- inference ของ test images 3 รูป
- ข้อสรุปว่า network converge หรือยัง พร้อมหลักฐานจากกราฟ

## Step 0 — กำหนดชุดข้อมูลร่วม

ใช้รูปจาก `data/400-449_completed/` เท่านั้น เพราะ `400-449_completed/` และ `450-499_completed/` เป็นข้อมูลสำเนาเดียวกันทั้งรูปและ `annotations.xml`

ใช้ split นี้ร่วมกับ Part 2 และ Part 3:

- **train 14 รูป:** `0400, 0403, 0405, 0408, 0410, 0413, 0416, 0418, 0421, 0423, 0426, 0429, 0431, 0433`
- **val 3 รูป:** `0436, 0439, 0442`
- **test 3 รูป:** `0445, 0447, 0449`

เลขข้างบนหมายถึงเลขหลัง `frame_` เช่น `0400` คือ `frame_0400_f10374.jpg`

ชุดที่คัดแล้วอยู่ที่ [[dataset-20/README|Dataset 20 — ชุดข้อมูลร่วม]]

- [x] จดรายชื่อไฟล์จริงทั้ง 20 รูปไว้ใน `manifest.csv`
- [x] ตรวจว่าแต่ละชื่อมี `<image name="...">` ใน `annotations.xml`
- [x] ใช้เฉพาะข้อมูลจาก `400-449_completed/` ไม่ปนโฟลเดอร์สำเนา

## Step 1 — เตรียม working directory นอก vault

repo และ virtual environment ต้องอยู่นอก Obsidian vault:

```text
D:\CoE Y.4 T.1\241-353\code\F09-TensorBoard\
├── yolov5\
├── datasets\f09_box\
└── notes\
```

เก็บเฉพาะรายงาน รูปผลลัพธ์ และไฟล์ที่ต้องส่งกลับมาในโฟลเดอร์งานนี้ ห้าม clone repo ลง vault

> [!success] Step 1 เสร็จแล้ว — 2026-09-14
> สร้าง `D:\CoE Y.4 T.1\241-353\code\F09-TensorBoard\` พร้อม `datasets/f09_box/` และ `notes/` แล้ว ส่วน `yolov5/` จะเกิดจาก `git clone` ใน Step 3 จึงยังไม่สร้างโฟลเดอร์เปล่าไว้

## Step 2 — แปลง CVAT XML เป็น YOLO detection format

กำหนด class ID ให้คงที่ทุก Part:

| class ID | class        |
| -------: | ------------ |
|        0 | `lane`       |
|        1 | `track-line` |
|        2 | `sideway`    |

ใน `annotations.xml` มี `<box>` สำหรับ `lane` และ `track-line` อยู่แล้ว แต่ `sideway` มีเฉพาะ `<polygon>` ดังนั้น converter ใช้กติกานี้:

1. อ่านเฉพาะ `<image>` ของ 20 รูปที่เลือก
2. ใช้ `<box>` เดิมสำหรับ `lane` และ `track-line`
3. สำหรับ `sideway` ให้หากรอบเล็กสุดที่ครอบ polygon ด้วย `xmin`, `ymin`, `xmax`, `ymax`
4. อย่าแปลง polygon ของ `lane` ซ้ำเป็น box เพราะมี `<box label="lane">` อยู่แล้ว
5. แปลงแต่ละ box เป็น YOLO format:

   ```text
   class_id x_center/W y_center/H box_width/W box_height/H
   ```

6. clamp พิกัดให้อยู่ในขอบภาพ 1280×720 และค่าที่ normalize แล้วต้องอยู่ใน `[0,1]`

โครงสร้างปลายทาง:

```text
datasets/f09_box/
├── images/
│   ├── train/  # 14 images
│   ├── val/    # 3 images
│   └── test/   # 3 images
├── labels/
│   ├── train/  # 14 txt files
│   ├── val/    # 3 txt files
│   └── test/   # 3 txt files
└── data.yaml
```

`data.yaml`:

```yaml
path: D:/CoE Y.4 T.1/241-353/code/F09-TensorBoard/datasets/f09_box
train: images/train
val: images/val
test: images/test
names:
  0: lane
  1: track-line
  2: sideway
```

> [!success] Step 2 เสร็จแล้ว — 2026-09-14
> สร้าง YOLO detection dataset ที่ `D:\CoE Y.4 T.1\241-353\code\F09-TensorBoard\datasets\f09_box\` แล้ว มีภาพ/labels 14/3/3, รวม 20 คู่และ 140 objects พร้อม `data.yaml`, `manifest.csv` และ contact sheet ที่ `notes/f09_box_overlay_contact_sheet.jpg`

### ตรวจ dataset ก่อน train

- [x] มีภาพ 14/3/3 ตรงตาม split
- [x] ทุกภาพมี label `.txt` ชื่อเดียวกัน
- [x] class ID มีเฉพาะ `0`, `1`, `2`
- [x] ทุกพิกัดอยู่ใน `[0,1]`
- [x] วาด bounding box ทับภาพทั้ง 20 รูปแล้วตรวจด้วยตา
- [x] ไม่มีภาพหรือ label ซ้ำข้าม split

## Step 3 — เตรียม YOLOv5 และ environment

เปิด PowerShell แล้วทำใน `code/`:

```powershell
Set-Location 'D:\CoE Y.4 T.1\241-353\code'
New-Item -ItemType Directory -Force 'F09-TensorBoard'
Set-Location 'F09-TensorBoard'
git clone https://github.com/ultralytics/yolov5.git
Set-Location 'yolov5'
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install tensorboard
```

> [!success] Clone repo แล้ว — 2026-09-14
> `yolov5/` ถูก clone แบบ shallow จาก `https://github.com/ultralytics/yolov5.git` ที่ branch `master`, commit `35b48237aef6d71ca9de2c5dea345d7536eb7fa7` ส่วน `.venv` และ dependencies ยังไม่ได้ติดตั้ง

ใช้ clean environment นี้และอย่าตั้ง API key ของ Comet ML หรือ ClearML หาก logger ใดทำงานอัตโนมัติ ให้ปิด integration นั้นก่อน train รอบจริง

- [x] `python --version` ใช้ interpreter จาก `.venv`
- [x] ตรวจ device แล้วและบังคับใช้ CPU ทุก run ด้วย `--device cpu`
- [x] TensorBoard 2.21.0 เปิด event logs ได้และบันทึก UI screenshot แล้ว

## Step 4 — Smoke test 1 epoch

YOLOv5 รุ่นปัจจุบันบันทึก standard metrics เข้า TensorBoard อยู่แล้ว จึงทดสอบของเดิมก่อนแก้ `train.py`:

```powershell
python train.py --img 640 --batch-size 2 --epochs 1 --data '..\datasets\f09_box\data.yaml' --weights yolov5n.pt --project runs\f09_box --name smoke --exist-ok
tensorboard --logdir runs\f09_box
```

เปิด URL ที่ TensorBoard แสดง แล้วตรวจหน้า **Scalars**:

- [x] เห็น training losses
- [x] เห็น validation/evaluation metrics
- [x] เห็น learning rate
- [x] step บนแกน x เพิ่มตาม iteration หรือ epoch ตามชนิด metric

## Step 5 — เติม TensorBoard logging เท่าที่ขาด

ตรวจ tag ที่มีอยู่ก่อนแก้โค้ด เป้าหมายคือให้ได้ plot pane แยกดังนี้:

```text
train/total_loss
train/box_loss
train/obj_loss
train/cls_loss
train/learning_rate
```

แนวทางแก้แบบสั้นที่สุด:

1. ใช้ TensorBoard writer/logger ที่ YOLOv5 สร้างไว้อยู่แล้ว
2. เพิ่มเฉพาะ scalar ที่ยังไม่มี โดยเฉพาะ `total_loss`
3. ใช้ global iteration เป็น `global_step` สำหรับ training loss
4. log ค่า LR จริงจาก `optimizer.param_groups[0]['lr']`
5. อย่าสร้าง logger class ใหม่หรือแก้หลายไฟล์ถ้าแก้ใน `train.py` จุดเดียวพอ
6. ปิด writer ตอน train จบ หากสร้าง `SummaryWriter` เพิ่มเอง

หลังแก้ ให้ลบเฉพาะโฟลเดอร์ `runs/f09_box/smoke` หรือเปลี่ยน `--name` เพื่อไม่ให้ event จากคนละรอบปนกัน

## Step 6 — สร้าง baseline ใหม่หลังล้างผล

สร้าง baseline ใหม่ครบทั้งสี่ขนาดแล้วใน `yolov5/runs/f09_box_compare/` โดยใช้ run names ตาม Phase B และเลือก `size350_e30_seed2569/weights/best.pt` เป็น final model จาก common-test mAP50-95 สูงสุด

- [x] run จบโดยไม่มี NaN loss
- [x] มี `best.pt` และ `last.pt`
- [x] มี TensorBoard event file
- [x] บันทึก device, batch size, epochs และ image size ใน `opt.yaml`
- [x] มีอย่างน้อย 30 iterations

### Step 6A — ทำ component losses ระดับ iteration เมื่อใช้เกณฑ์ตามตัวอักษรของสไลด์

โค้ด Step 6A ถูก merge เข้า `main` แล้วที่ commit `708f301` และ smoke test ก่อนล้างผลเคยยืนยันว่า iteration tags ทั้งห้าทำงานครบ แต่ event และ output ของ smoke test ถูกลบตามคำสั่งล้างผล การ train ครั้งใหม่จะบันทึก component losses, total loss และ learning rate ระดับ iteration โดยอัตโนมัติ

แนวทางแก้:

1. ใน training loop ส่ง `loss_items` ของ batch ปัจจุบันเข้า `on_train_batch_end` แทน `mloss` ซึ่งเป็น running mean ภายใน epoch
2. ใน TensorBoard logger บันทึก `vals[0]`, `vals[1]`, `vals[2]` เป็น `train/box_loss_iter`, `train/obj_loss_iter`, `train/cls_loss_iter`
3. ใช้ `ni` เป็น global step เช่นเดียวกับ `train/total_loss` และ `train/learning_rate`
4. คง epoch-level tags เดิมไว้เพื่อใช้ดูแนวโน้มที่ smooth กว่า
5. smoke test ก่อน แล้วตรวจว่า iteration tags มีจำนวน steps ตรงกับจำนวน batches
6. รัน final ด้วยชื่อใหม่และไม่ใช้ `--exist-ok`

จุดแก้ใน `yolov5/train.py` ภายใน callback `on_train_batch_end`:

```python
# เดิม
list(mloss),

# ใหม่: raw loss ของ batch ปัจจุบัน
list(loss_items.detach()),
```

เพิ่มใน `on_train_batch_end` ของ `yolov5/utils/loggers/__init__.py` ภายใต้เงื่อนไข `if self.tb:`:

```python
self.tb.add_scalar("train/box_loss_iter", vals[0], ni)
self.tb.add_scalar("train/obj_loss_iter", vals[1], ni)
self.tb.add_scalar("train/cls_loss_iter", vals[2], ni)
self.tb.add_scalar("train/total_loss", sum(vals), ni)
self.tb.add_scalar("train/learning_rate", learning_rate, ni)
```

ใช้คำสั่ง Phase A และ Phase B ในหัวข้อ `แผนการทดลอง 4 ขนาด: 30/60/150/350` เพื่อให้ชื่อ run และ dataset ตรงกับแผนเปรียบเทียบล่าสุด

ตรวจ smoke event ก่อนเริ่ม 30 epochs:

```powershell
python -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; import glob; p=glob.glob(r'runs\f09_box_compare\preflight_size30\events.out.tfevents.*')[0]; a=EventAccumulator(p); a.Reload(); print({t: len(a.Scalars(t)) for t in a.Tags()['scalars'] if t.endswith('_iter') or t in ('train/total_loss','train/learning_rate')})"
```

ผล preflight ที่ผ่านต้องมีทั้งห้า tags อย่างละ 11 steps สำหรับ dataset 30 ห้ามเริ่ม full training หากจำนวน steps ไม่เท่ากันหรือ loss มี NaN

## Step 7 — Evaluation และ inference บน test split

ทำ Phase D ให้ครบก่อน แล้วเลือก final run จาก common-test results ทำงานจากโฟลเดอร์ `yolov5/` และกำหนด `$finalRun` เป็นชื่อผู้ชนะเพียงตัวเดียว ตัวอย่างตั้งต้นด้านล่างใช้ size 350 แต่ต้องเปลี่ยนหาก size อื่นชนะ

### Step 7A — Standalone test evaluation

```powershell
$finalRun = 'size350_e30_seed2569'
$weights = Join-Path 'runs\f09_box_compare' "$finalRun\weights\best.pt"
New-Item -ItemType Directory -Force '..\result' | Out-Null
python val.py --weights $weights --data '..\datasets\f09_box_350\data.yaml' --task test --img 640 --batch-size 2 --device cpu --workers 0 --project runs\f09_box_final --name "${finalRun}-test-eval" 2>&1 | Tee-Object '..\result\final_test_metrics.txt'
```

ตรวจและบันทึกค่าจากแถว `all`: precision, recall, mAP50 และ mAP50-95 ห้ามนำ validation metrics มาแทน common-test metrics ให้เก็บ confusion matrix/curves ของ test run แยกชื่อจาก training evidence

- [x] test evaluation จบโดยไม่มี error
- [x] บันทึก precision, recall, mAP50 และ mAP50-95 ของ test split
- [x] ระบุจำนวน test images 50 ภาพและ 351 objects
- [x] เก็บ output และกราฟไว้ภายใต้ `result/`

### Step 7B — Test inference และเลือกตัวอย่าง 3 ภาพ

```powershell
python detect.py --weights $weights --source '..\datasets\f09_box_350\images\test' --img 640 --device cpu --save-txt --save-conf --project runs\f09_box_final --name "${finalRun}-test-inference"
```

รัน inference ทั้ง 50 test images แล้วเลือกตัวอย่างที่อยู่ห่างกันในลำดับเฟรม 3 ภาพ เพื่อไม่ให้รายงานมีภาพแทบเหมือนกัน แนะนำเริ่มตรวจจาก:

- `frame_0700_f18154.jpg`
- `frame_0725_f18803.jpg`
- `frame_0749_f19425.jpg`

คัดภาพ output เป็น `result/screenshots/12_test_inference_1.jpg` ถึง `14_test_inference_3.jpg` และตรวจด้วยตาว่า class อ่านได้, confidence แสดงครบ, กรอบไม่หลุดขอบภาพ และมีตัวอย่างข้อผิดพลาดของโมเดลหากพบ ห้ามใช้ `val_batch*_pred.jpg` แทน test inference

```powershell
New-Item -ItemType Directory -Force '..\result\screenshots' | Out-Null
Copy-Item "runs\f09_box_final\${finalRun}-test-inference\frame_0700_f18154.jpg" '..\result\screenshots\12_test_inference_1.jpg'
Copy-Item "runs\f09_box_final\${finalRun}-test-inference\frame_0725_f18803.jpg" '..\result\screenshots\13_test_inference_2.jpg'
Copy-Item "runs\f09_box_final\${finalRun}-test-inference\frame_0749_f19425.jpg" '..\result\screenshots\14_test_inference_3.jpg'
```

- [x] มี inference output ของ test split ครบ 50 ภาพ
- [x] คัดตัวอย่าง test 3 ภาพเข้า `result/screenshots/`
- [x] เขียนคำอธิบายสั้น ๆ ใน `result/RUN_SUMMARY.md`

## Step 8 — เก็บภาพ TensorBoard สำหรับรายงาน

```powershell
tensorboard --logdir runs\f09_box_compare --port 6006
```

เลือก final run ที่ชนะเมื่อต้องการภาพสำหรับส่ง หรือเลือกทั้งสี่ runs เมื่อต้องการภาพเปรียบเทียบ ปิด smoothing หรือระบุค่า smoothing ในคำอธิบายภาพ และตั้ง Horizontal Axis เป็น Step

ต้องถ่ายภาพโดยขยาย card จริง ไม่ใช่แสดงเพียงชื่อกลุ่มที่พับอยู่:

1. training losses: total, objectness, classification และ box regression
2. evaluation metrics: precision, recall, mAP50 และ mAP50-95
3. scheduled learning rate

ในทุกภาพต้องเห็นชื่อ run, tag/ชื่อกราฟ, แกน x และเส้นกราฟชัดเจน ควรแบ่งเป็น 2–3 screenshots แทนการย่อทุก card ลงในภาพเดียว แล้วเก็บเป็น:

```text
result/screenshots/15_tensorboard_training_losses.png
result/screenshots/16_tensorboard_evaluation_metrics.png
result/screenshots/17_tensorboard_learning_rate.png
```

- [x] training loss plots ครบ 4 ค่า
- [x] evaluation metrics plots ครบ 4 ค่า
- [x] learning-rate plot ชัดเจน
- [x] แกน x และชื่อ run มองเห็นได้
- [x] weight distribution ไม่จัดทำในรอบนี้ เพราะเป็นส่วนเสริมที่อาจารย์ไม่ได้กำหนด

## Step 9 — เขียนข้อสรุป convergence

ตอบโดยอ้างหลักฐานอย่างน้อยสามส่วนร่วมกัน:

1. แนวโน้ม training และ validation losses ตลอด 30 epochs
2. precision, recall, mAP50 และ mAP50-95 ว่าเพิ่มขึ้นและเริ่ม plateau หรือยัง
3. คุณภาพเชิงภาพของ test inference 3 รูปและ standalone test metrics

ข้อสรุปจากผลจริง:

> โมเดล size-350 มีแนวโน้ม converge เพราะ box, objectness, classification และ total losses ลดลงต่อเนื่อง ขณะที่ precision/recall/mAP เพิ่มขึ้นและเริ่มคงที่ช่วงท้าย ผล validation ดีที่สุดที่ epoch 29 คือ precision 0.99625, recall 1.0, mAP50 0.995 และ mAP50-95 0.81224 ส่วน common test 50 ภาพได้ precision 0.990, recall 0.990, mAP50 0.989 และ mAP50-95 0.725 ภาพ inference ต้น/กลาง/ท้ายตรวจพบ lane, track-line และ sideway ได้สอดคล้องกับวัตถุหลัก อย่างไรก็ตาม ภาพมาจากลำดับเฟรมที่สัมพันธ์กัน จึงยังใช้ยืนยัน generalization ต่อสภาพแวดล้อมใหม่ไม่ได้

อย่าเขียนว่า converge จากค่า mAP สูงเพียงค่าเดียว และอย่าเรียก validation predictions ว่า test inference

- [x] เติม test metrics ลงในคำตอบ
- [x] อ้างภาพ TensorBoard ที่เกี่ยวข้อง
- [x] อธิบายข้อจำกัดของข้อมูลที่เป็นลำดับเฟรม

## Step 10 — จัดชุดหลักฐานและรายงานส่ง

โครงสร้างขั้นต่ำที่ต้องมี:

```text
result/
├── RUN_SUMMARY.md
├── metrics.csv
├── dataset350_test_metrics.txt
└── screenshots/
    ├── 12_test_inference_1.jpg
    ├── 13_test_inference_2.jpg
    ├── 14_test_inference_3.jpg
    ├── 15_tensorboard_training_losses.png
    ├── 16_tensorboard_evaluation_metrics.png
    └── 17_tensorboard_learning_rate.png
```

อัปเดต `RUN_SUMMARY.md` ให้มี test metrics, ลิงก์ inference 3 ภาพ, คำตอบ convergence และ training settings ของ final run จากนั้นนำเนื้อหาและภาพเหล่านี้ไปวางในรายงานส่ง Microsoft Teams

## Checklist จบ Part 1

- [x] YOLO detection dataset ผ่าน structural และ visual checks
- [x] training มีอย่างน้อย 30 iterations
- [x] มี `best.pt` จาก pretrained `yolov5n.pt`
- [x] TensorBoard event มี total loss, component losses, metrics และ LR
- [x] ใช้ split ปัจจุบัน 250/50/50 และระบุว่าเพิ่ม validation set พร้อมใช้ common test 50 ภาพที่ไม่ใช้ train
- [x] สร้าง final runs ที่ log component losses ระดับ iteration
- [x] standalone test evaluation เสร็จและบันทึก metrics แล้ว
- [x] inference จาก test split 3 รูปเสร็จ
- [x] ภาพ TensorBoard แสดง training losses, evaluation metrics และ LR ครบ
- [x] เขียนคำตอบ convergence โดยอ้างกราฟและผล test
- [x] อัปเดต `result/RUN_SUMMARY.md` และจัดเนื้อหาเข้า report
- [x] ตรวจผลหลักว่า common test ใช้ split เดียวกันและ final inference ใช้ size-350 run

## แหล่งอ้างอิง

- [YOLOv5 custom training](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)
- [YOLOv5 repository](https://github.com/ultralytics/yolov5)
- [PyTorch TensorBoard](https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)

## Progress update — 2026-09-14

- Step 4 smoke test passed on CPU without NaN loss; the successful run is `yolov5/runs/train/exp6`.
- The original TensorBoard output contains train/validation losses, evaluation metrics, and `x/lr0`–`x/lr2`.
- Step 5 is verified by `yolov5/runs/f09_box/smoke_tags`: `train/total_loss` and `train/learning_rate` were each recorded for all 7 training batches using global iteration as the step.
- GPU training with PyTorch `2.11.0+cu128` and AMP produced NaN losses on the GTX 1660 SUPER. Use CPU for the recorded run unless GPU training is revalidated with AMP disabled.

## Documentation policy

From 2026-09-14 onward, record every important code change, experiment result, decision, and work milestone in `REPORT.md`, and update the corresponding status or instructions in this guide.

Detailed file-by-file changes, before/after behavior, rationale, and verification evidence for the Step 5 TensorBoard modification are recorded in the `รายละเอียดการแก้ไขโค้ด` and `การตรวจสอบหลังแก้ไข` sections of `REPORT.md`.

Generated Python caches, YOLO run outputs, TensorBoard events, downloaded model weights, and dataset index caches are excluded through the repository `.gitignore`. The implementation and documentation are committed separately; see `แผนการแบ่ง Git commits` in `REPORT.md`.

The completed commits and their hashes are recorded in the `Git commits ที่สร้างแล้ว` section of `REPORT.md`.

### Five-epoch run review — 2026-09-14

The first `exp20` directory is not suitable for final evidence: `results.csv` contains epoch 0 twice, showing that `--exist-ok` mixed multiple executions in one directory. Its losses are finite and all required TensorBoard tags are present, but the metrics peak early and then decline, so the run has not converged. Use the updated Step 6 command with the unused name `exp20_clean` and without `--exist-ok`.

### Clean run convergence review — 2026-09-14

`exp20_clean` contains one event file and exactly five CSV rows for epochs 0–4. The loss is not expected to decrease smoothly: the run has only 35 iterations, while YOLOv5 enforces at least 100 warmup iterations, so the entire run remains in warmup. In addition, the custom `train/total_loss` currently plots the within-epoch running mean (`mloss`) at every global step; that mean resets each epoch and can create visible discontinuities. Epoch losses fluctuate and metrics decline after an early peak, so the supported conclusion is that the network has **not converged within five epochs**.

To inspect convergence beyond warmup, run one controlled 30-epoch experiment (about 210 iterations) without changing the other hyperparameters:

```powershell
python train.py --img 640 --batch-size 2 --epochs 30 --data '..\datasets\f09_box\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --project runs\f09_box --name exp20_30e
```

Do not use `--exist-ok`. Judge the result from epoch-level train and validation losses together with precision, recall, mAP50, and mAP50-95. A decreasing train loss accompanied by worsening validation results indicates overfitting, not successful generalization.

### 30-epoch result — 2026-09-14

The completed run is named `exp30_epoch`. It contains 30 unique CSV epochs and both `best.pt` and `last.pt`. From epoch 0 to 29, train box loss decreased from `0.10221` to `0.076912`, and classification loss decreased from `0.043962` to `0.03722`, while objectness loss remained noisy and ended at `0.074991`. Epoch 29 achieved precision `0.30842`, recall `0.46296`, mAP50 `0.35436`, and mAP50-95 `0.11719`. The supported conclusion is that the network has learned and started to converge after warmup, but has not reached a stable plateau; the three-image validation set is too small for a strong generalization claim.

### Parameter tuning note — 2026-09-14

Parameter tuning may improve validation metrics, but the 14/3 train/validation split is too small for reliable search. The deleted `exp30_epoch` remains a historical baseline through its recorded metrics; regenerate a clean 30-epoch baseline before any future tuning comparison. Change only one variable per run, and never use the three test images to select parameters. The first controlled candidates are 50 epochs with otherwise identical settings and a separate 30-epoch run with `--freeze 10`. Prefer adding diverse labeled images over extensive tuning; higher metrics on three validation images do not establish generalization. Commands and comparison rules are recorded in `REPORT.md` under `แนวทางปรับ parameters`.

### Run cleanup — 2026-09-14

ข้อความนี้เป็นสถานะก่อนสร้าง expanded-dataset runs โดยตอนนั้นเหลือเพียง `smoke_1epoch_tensorboard` และ `baseline_5epochs` ต่อมามีการสร้าง `dataset350_smoke_retry` และ `dataset350_baseline30` สำเร็จแล้ว ปัจจุบัน Step 7–10 จึงอ้างอิง `dataset350_baseline30` และไม่ใช้ five-epoch baseline เป็นผลส่ง

### Extended dataset verification — 2026-09-14

The expanded dataset contains 250 train, 50 validation, and 50 test image/label pairs (350 JPEG images at 1280×720, 2,455 objects). Label structure, class IDs, normalized coordinates, image readability, image/label pairing, and cross-split SHA-256 duplication checks all pass. Before training, change the stale D-drive path in `datasets/f09_box/data.yaml` to the portable `path: ../datasets/f09_box`. Use a new one-epoch CPU smoke run (`dataset350_smoke`) first; if it has finite losses and produces the required TensorBoard tags, use `dataset350_baseline30` for a controlled 30-epoch baseline. Full counts and commands are recorded in `REPORT.md`.

The initial expanded-dataset smoke command failed before training because the stale D-drive path had not yet been changed. `datasets/f09_box/data.yaml` now uses `path: ../datasets/f09_box`; validate this resolution and rerun the one-epoch smoke test. The failed launch produced no training result and should not be treated as a completed smoke run.

### Evidence package — 2026-09-14

The important completed runs, best-epoch metrics, TensorBoard captures, YOLO result plots, confusion matrix, PR/F1 curves, label distribution, and validation prediction examples are collected separately under `result/`. Start with `result/RUN_SUMMARY.md`; machine-readable metrics are in `result/metrics.csv`, and all images are in `result/screenshots/`. The strongest run is `dataset350_baseline30` at epoch 28 with precision `0.99648`, recall `1.0`, mAP50 `0.995`, and mAP50-95 `0.79994`.

Dataset scan caches ending in both `.cache` and `.cache.npy` are generated artifacts and are excluded by `.gitignore`.

### Part 1 completion audit — 2026-09-15

ตรวจสไลด์ต้นฉบับหน้า 17–19 เทียบกับผลรอบใหม่แล้ว งาน training, standalone common-test evaluation, test inference 3 ภาพ, TensorBoard screenshots, component losses ระดับ iteration และคำตอบ convergence เสร็จครบ หลักฐานอยู่ใน `result/` และเลือก size-350 เป็น final model จาก common-test mAP50-95 สูงสุด ทั้งนี้รายงานใช้ split ปัจจุบัน 250/50/50 พร้อมระบุว่าแตกต่างจากตัวอย่าง train 100/test 100 ในสไลด์

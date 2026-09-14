---
type: assignment
instructor: P.Fern
date: 2026-08-31
due:
status: todo
tags: [cv, tool, pytorch]
aliases: [F09 Part 1 YOLO-Box]
---

# Part 1 — YOLOv5 Box Detection

> แผนนี้ขยาย Part 1 จาก [[P.Fern/Assignments/F09_31-08-69/instruction]] เป้าหมายคือ train YOLOv5n ตรวจจับ bounding box ด้วยภาพ 20 รูป และเก็บกราฟที่โจทย์กำหนดใน TensorBoard

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

- [ ] `python --version` ใช้ interpreter จาก `.venv`
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` แสดงสถานะ GPU
- [ ] `tensorboard --help` ทำงาน

## Step 4 — Smoke test 1 epoch

YOLOv5 รุ่นปัจจุบันบันทึก standard metrics เข้า TensorBoard อยู่แล้ว จึงทดสอบของเดิมก่อนแก้ `train.py`:

```powershell
python train.py --img 640 --batch-size 2 --epochs 1 --data '..\datasets\f09_box\data.yaml' --weights yolov5n.pt --project runs\f09_box --name smoke --exist-ok
tensorboard --logdir runs\f09_box
```

เปิด URL ที่ TensorBoard แสดง แล้วตรวจหน้า **Scalars**:

- [ ] เห็น training losses
- [ ] เห็น validation/evaluation metrics
- [ ] เห็น learning rate
- [ ] step บนแกน x เพิ่มตาม iteration หรือ epoch ตามชนิด metric

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

## Step 6 — Train รอบเก็บผล

ใช้ `batch-size=2` และ 5 epochs เป็นค่าเริ่มต้น เพราะ 14 train images จะได้ประมาณ 7 iterations ต่อ epoch รวมประมาณ 35 iterations ซึ่งผ่านเงื่อนไขอย่างน้อย 30 iterations

```powershell
python train.py --img 640 --batch-size 2 --epochs 5 --data '..\datasets\f09_box\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --project runs\f09_box --name exp20_clean
```

หาก GPU memory ไม่พอ ให้ลด batch size และเพิ่ม epochs จนจำนวน iterations รวมยังไม่น้อยกว่า 30 อย่าเปลี่ยนหลาย hyperparameters พร้อมกัน

- [ ] run จบโดยไม่มี NaN loss
- [ ] มี `runs/f09_box/exp20/weights/best.pt`
- [ ] มี TensorBoard event file
- [ ] บันทึก device, batch size, epochs, image size และเวลาที่ใช้

## Step 7 — Evaluation และ inference

ใช้ `best.pt` และ test split 3 รูป:

```powershell
python val.py --weights runs\f09_box\exp20\weights\best.pt --data '..\datasets\f09_box\data.yaml' --task test --img 640
python detect.py --weights runs\f09_box\exp20\weights\best.pt --source '..\datasets\f09_box\images\test' --img 640 --save-txt --save-conf --project runs\f09_box --name predict20 --exist-ok
```

- [ ] เก็บภาพ prediction ครบ 3 รูป
- [ ] ตรวจว่ากรอบไม่หลุดขอบภาพและ class อ่านได้
- [ ] จด precision, recall, mAP50 และ mAP50-95 ที่โปรแกรมรายงาน
- [ ] ใช้ภาพ test ชุดเดิมต่อใน Part 2 และ Part 3

## Step 8 — เก็บหลักฐานจาก TensorBoard

```powershell
tensorboard --logdir runs\f09_box\exp20
```

ถ่ายภาพหรือ export plot โดยให้เห็นชื่อ run, ชื่อแกน และค่าครบ:

- [ ] total loss
- [ ] objectness loss
- [ ] classification loss
- [ ] box regression loss
- [ ] learning rate
- [ ] evaluation metrics
- [ ] weight distribution over time ถ้าทำส่วนเสริม

## Step 9 — ตอบเรื่อง convergence

อย่าตอบจาก training loss เส้นเดียว ให้พิจารณาพร้อมกัน:

1. training losses ลดลงแล้วเริ่มนิ่งหรือไม่
2. validation loss/metrics ดีขึ้นแล้วเริ่ม plateau หรือไม่
3. train ดีขึ้นแต่ validation แย่ลงหรือไม่ ซึ่งบ่งชี้ overfitting
4. prediction บน test 3 รูปสมเหตุสมผลหรือไม่

ชุดข้อมูลมีเพียง 20 รูป จึงสรุปได้เฉพาะว่า run นี้เริ่ม converge หรือ overfit อย่างไร ห้ามอ้างว่าโมเดล generalize ได้ดีจากตัวอย่างขนาดเล็กนี้

## Step 10 — คัดผลกลับเข้า vault

คัดเฉพาะสิ่งที่ใช้ส่งกลับมายังโฟลเดอร์งาน เช่น:

```text
results/part1-box/
├── tensorboard/
├── predictions/
├── metrics.txt
└── train-settings.md
```

ไม่คัด repo, `.venv`, weights ทุก epoch หรือ cache เข้ามาใน vault

## Checklist จบ Part 1

- [ ] dataset 20 รูปผ่าน visual check
- [ ] training มีอย่างน้อย 30 iterations
- [ ] TensorBoard มีทุก plot ที่โจทย์กำหนด
- [ ] evaluation metrics ถูกบันทึก
- [ ] inference test 3 รูปเสร็จ
- [ ] ร่างคำตอบเรื่อง convergence แล้ว
- [ ] ผลลัพธ์ที่ใช้ทำรายงานถูกคัดกลับเข้า vault

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

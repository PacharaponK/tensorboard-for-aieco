# F09 Part 1 — YOLOv5 Box Detection Report

## บันทึกการดำเนินงาน

### 2026-09-14 — Smoke test และ TensorBoard

- ทดสอบ YOLOv5n ด้วยข้อมูล train 14 ภาพ, validation 3 ภาพ และ 3 classes (`lane`, `track-line`, `sideway`)
- การรันบน NVIDIA GeForce GTX 1660 SUPER ด้วย PyTorch `2.11.0+cu128` และ AMP ให้ค่า `box_loss`, `obj_loss` และ `cls_loss` เป็น `NaN`
- การรันบน CPU สำเร็จโดยไม่มี `NaN`; ผล smoke test อยู่ที่ `yolov5/runs/train/exp6`
- ค่าจาก smoke test: `box_loss=0.1071`, `obj_loss=0.06238`, `cls_loss=0.04415`, `precision=0.00506`, `recall=0.167`, `mAP50=0.00360` และ `mAP50-95=0.00192`
- TensorBoard เดิมมี training/validation losses, precision, recall, mAP และ learning-rate groups แต่ไม่มี `train/total_loss` และ `train/learning_rate`
- ปรับ TensorBoard logging ให้บันทึก `train/total_loss` และ `train/learning_rate` ทุก training iteration โดยใช้ global iteration เป็น step และอ่าน learning rate จริงจาก `optimizer.param_groups[0]["lr"]`

## สถานะปัจจุบัน

- Step 4 smoke test: ผ่านบน CPU
- Step 5 TensorBoard logging: ผ่านการยืนยันด้วย `yolov5/runs/f09_box/smoke_tags`; ทั้ง `train/total_loss` และ `train/learning_rate` มี 7 steps ตรงกับ 7 training batches
- การตรวจ syntax ของ `train.py` และ `utils/loggers/__init__.py`: ผ่าน
- ขั้นถัดไป: เปิด TensorBoard ตรวจกราฟของ `smoke_tags` แล้ว train รอบเก็บผล 5 epochs

## รายละเอียดการแก้ไขโค้ด

### `yolov5/train.py`

- ตำแหน่ง: ฟังก์ชัน `train()` ภายใน training loop ส่วน `# Log` หลังคำนวณค่าเฉลี่ยสะสม `mloss`
- จุดที่แก้: การเรียก callback `on_train_batch_end`
- ก่อนแก้: callback ได้รับเฉพาะ model, global iteration (`ni`), batch data และค่าเฉลี่ย `box_loss`, `obj_loss`, `cls_loss`
- หลังแก้: ส่ง `optimizer.param_groups[0]["lr"]` เพิ่มเป็น argument สุดท้าย เพื่อให้ logger ได้ learning rate ที่ optimizer ใช้จริงใน iteration นั้น
- เหตุผล: ค่า learning rate เดิม (`x/lr0`–`x/lr2`) ถูกบันทึกระดับ epoch และชื่อ tag ไม่ตรง `train/learning_rate` ตามข้อกำหนด ส่วนการส่งค่าจาก training loop ทำให้ logger ไม่ต้องเข้าถึง optimizer โดยตรง
- ผลกระทบ: ไม่มีการเปลี่ยน forward pass, loss calculation, backward pass หรือ optimizer step; เปลี่ยนเฉพาะข้อมูลที่ส่งไป logging callback

โค้ดก่อนแก้:

```python
callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
```

โค้ดหลังแก้:

```python
callbacks.run(
    "on_train_batch_end",
    model,
    ni,
    imgs,
    targets,
    paths,
    list(mloss),
    optimizer.param_groups[0]["lr"],
)
```

### `yolov5/utils/loggers/__init__.py`

- ตำแหน่ง: class `Loggers`, method `on_train_batch_end()`
- จุดที่แก้: เพิ่ม parameter `learning_rate` ให้สอดคล้องกับ callback จาก `train.py`
- เพิ่ม `self.tb.add_scalar("train/total_loss", sum(vals), ni)` เพื่อบันทึกผลรวมของค่าเฉลี่ย `box_loss + obj_loss + cls_loss`
- เพิ่ม `self.tb.add_scalar("train/learning_rate", learning_rate, ni)` เพื่อบันทึก learning rate จริงของ optimizer parameter group แรก
- ใช้ `ni` เป็น TensorBoard step ซึ่งเป็น global iteration (`batch_index + batches_per_epoch × epoch`) ทำให้แกน X ต่อเนื่องข้าม epoch
- ทำงานเฉพาะเมื่อ TensorBoard writer (`self.tb`) พร้อมใช้งาน จึงไม่สร้าง writer หรือ logger class ซ้ำ
- logging เดิมระดับ epoch เช่น `train/box_loss`, `metrics/*`, `val/*` และ `x/lr0`–`x/lr2` ยังคงเดิม

### `Guide.md`

- เปลี่ยนคำสั่ง train รอบเก็บผล 5 epochs ให้ระบุ `--device cpu --workers 0`
- บันทึกเหตุผลว่า GPU + AMP บน GTX 1660 SUPER และ PyTorch `2.11.0+cu128` ให้ค่า loss เป็น `NaN` แต่ CPU ให้ค่าปกติ
- บันทึกผลยืนยัน Step 5 ว่า tag ใหม่ทั้งสองมี 7 steps จาก 7 training batches
- เพิ่มนโยบายให้งานสำคัญครั้งถัดไปอัปเดตทั้ง `REPORT.md` และ `Guide.md`

### `REPORT.md`

- สร้างบันทึกผล smoke test, ปัญหา GPU/AMP, metrics, สถานะของแต่ละขั้น และรายละเอียดการแก้ไขรายไฟล์

### `.gitignore`

- เพิ่มกฎ ignore สำหรับ virtual environments (`.venv/`), Python bytecode/cache (`__pycache__/`, `*.py[cod]`), YOLO run outputs (`runs/`), model weights (`*.pt`), TensorBoard event files (`events.out.tfevents.*`) และ dataset index cache (`*.cache`)
- เพิ่มกฎสำหรับไฟล์ metadata ของระบบปฏิบัติการ (`.DS_Store`, `Thumbs.db`)
- ไม่ ignore รูปภาพ dataset, YOLO label `.txt`, `data.yaml`, source code หรือเอกสารที่เป็นข้อมูลต้นฉบับและผลส่งงาน
- หลังเพิ่มกฎ `datasets/f09_box/labels/train.cache` และ `val.cache` จะไม่ปรากฏเป็นไฟล์รอ commit

## แผนการแบ่ง Git commits

1. `chore: ignore generated training artifacts` — เฉพาะ `.gitignore`
2. `feat: log total loss and learning rate` — เฉพาะ `yolov5/train.py` และ `yolov5/utils/loggers/__init__.py`
3. `docs: record TensorBoard logging changes` — `REPORT.md` และ `Guide.md`

การแก้ไขเดิมใน `datasets/f09_box/data.yaml` ไม่เกี่ยวข้องกับงานนี้และจะไม่ถูกรวมใน commits ข้างต้น

## Git commits ที่สร้างแล้ว

- `0bceee9 chore: ignore generated training artifacts` — เพิ่ม `.gitignore`
- `8f25f49 feat: log total loss and learning rate` — เพิ่ม TensorBoard logging ใน training callback และ logger
- `c1bcacd docs: record TensorBoard logging changes` — บันทึกรายงาน รายละเอียดการแก้ไข และอัปเดตคู่มือ
- Git ใช้ committer identity ที่ตั้งอัตโนมัติเป็น `Student <student@lab.coe.psu.ac.th>`; ไม่มีการแก้ global Git configuration

## การตรวจสอบหลังแก้ไข

- `python -m py_compile train.py utils/loggers/__init__.py`: ผ่าน
- รัน CPU smoke test 1 epoch ที่ `yolov5/runs/f09_box/smoke_tags`: ผ่านและไม่มี `NaN`
- ตรวจ event file ด้วย TensorBoard event accumulator: พบ `train/total_loss` จำนวน 7 steps และ `train/learning_rate` จำนวน 7 steps
- training/validation tags เดิมยังอยู่ครบ ได้แก่ train losses, validation losses, precision, recall, mAP50, mAP50-95 และ learning-rate groups เดิม

### 2026-09-14 — ตรวจผล train 5 epochs (`exp20`)

- TensorBoard แสดง metrics ครบ 5 กลุ่มในหมวด train รวมทั้ง `train/total_loss` และ `train/learning_rate`; total loss และ learning rate มี step ระดับ iteration ตามที่กำหนด
- เส้นสีอ่อนใน TensorBoard คือค่าดิบ และเส้นสีเข้มคือค่าหลัง smoothing; แกน Y ถูกปรับช่วงแคบอัตโนมัติ จึงทำให้ความผันผวนดูรุนแรงกว่าค่าจริง
- ตรวจ `yolov5/runs/f09_box/exp20/results.csv` พบ epoch `0` ซ้ำสองแถว จึงสรุปว่าโฟลเดอร์ `exp20` มีผลจากมากกว่าหนึ่ง run ปะปนกัน เนื่องจากใช้ `--exist-ok`
- run นี้มี `best.pt` และ `last.pt` แต่ไม่ควรใช้กราฟเป็นหลักฐานสุดท้าย เพราะ event และ CSV ปะปนกับการรันก่อนหน้า
- แนวโน้มค่าจริงยังไม่ converge: `train/obj_loss` เพิ่มจาก `0.063952` เป็น `0.071624`; mAP50 สูงสุด `0.0056994` ที่ epoch 2 แล้วลดเป็น `0.0025514`; mAP50-95 ลดจาก `0.0018802` เป็น `0.00079441`
- ข้อมูลมีเพียง 14 training images และทดสอบเพียง 5 epochs จึงรายงานได้เฉพาะว่า run ยังไม่ converge ห้ามสรุปความสามารถในการ generalize
- การแก้ไข: รันใหม่ด้วยชื่อ `exp20_clean` และไม่ใช้ `--exist-ok` เพื่อให้ event file, CSV และ weights แยกจากทุก run ก่อนหน้า

### 2026-09-14 — วิเคราะห์กราฟ loss ของ `exp20_clean`

- ยืนยันว่า run สะอาด: `results.csv` มี epoch 0–4 อย่างละหนึ่งแถว และมี TensorBoard event file เพียงหนึ่งไฟล์
- loss โดยหลักควรมีแนวโน้มลดลงเมื่อ optimization ดำเนินไปเพียงพอ แต่ไม่จำเป็นต้องลดทุก batch หรือทุก epoch
- run นี้มีเพียง 14 training images × 5 epochs = 35 iterations ขณะที่ `train.py` กำหนด warmup ขั้นต่ำด้วย `nw = max(round(hyp["warmup_epochs"] * nb), 100)` จึงทำให้ทั้ง 35 iterations ยังอยู่ภายใน warmup 100 iterations
- learning rate ยังเปลี่ยนตลอด run และโมเดลยังไม่เข้าสู่ช่วง optimization หลัง warmup จึงไม่ควรคาดหวังเส้น loss ลู่ลงอย่างชัดเจนจากรอบ 5 epochs นี้
- `train/total_loss` ปัจจุบันคำนวณจาก `sum(mloss)` โดย `mloss` เป็นค่าเฉลี่ยสะสมภายใน epoch และถูกรีเซ็ตเมื่อต้น epoch รูปร่างระดับ iteration จึงอาจเกิดรอยต่อหรือแกว่งเมื่อเปลี่ยน epoch และไม่ใช่ raw per-batch loss
- ค่า epoch-level ยืนยันว่า `box_loss` แกว่ง, `obj_loss` เพิ่ม และ metrics สูงสุดช่วง epoch 1–2 ก่อนลดลง ดังนั้นข้อสรุปสำหรับรายงานคือ network **ยังไม่ converge** ภายใน 5 epochs
- ผลนี้ไม่ใช่ runtime error และใช้ตอบคำถามเรื่อง convergence ได้ตามขอบเขตของ Guide แต่หากต้องการกราฟ total loss แบบ raw per-batch ที่ตีความตรงกว่าเดิม ต้องเปลี่ยน logger ให้รับ `loss_items` ของ batch ปัจจุบันแทน `mloss`

## นโยบายการบันทึก

ตั้งแต่ 2026-09-14 เป็นต้นไป การเปลี่ยนแปลงโค้ด ผลการทดลอง การตัดสินใจ และการทำงานสำคัญของงานนี้ต้องบันทึกใน `REPORT.md` พร้อมอัปเดตสถานะที่เกี่ยวข้องใน `Guide.md`

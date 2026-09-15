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

### 2026-09-14 — แผนทดลองเพื่อดูแนวโน้ม convergence

- คง `batch-size=2`, image size 640, pretrained `yolov5n.pt` และ hyperparameters เดิม เพื่อเปลี่ยนตัวแปรหลักเพียงจำนวน epochs
- เพิ่มจาก 5 เป็น 30 epochs: 14 training images ให้ 7 batches ต่อ epoch รวมประมาณ 210 iterations ซึ่งผ่าน warmup ขั้นต่ำ 100 iterations และเหลือช่วงหลัง warmup ให้สังเกตแนวโน้ม
- ใช้ run ใหม่ `exp20_30e` โดยไม่ใช้ `--exist-ok` เพื่อป้องกัน event/CSV ปะปน
- ใช้ CPU เนื่องจาก GPU+AMP configuration ปัจจุบันเคยให้ NaN
- ประเมิน convergence จาก epoch-level train losses ร่วมกับ validation losses, precision, recall, mAP50 และ mAP50-95 ไม่ใช้ `total_loss` หรือ training loss เพียงกราฟเดียว
- หาก train loss ลดลงแต่ validation loss เพิ่มหรือ metrics ลดลง ให้สรุปว่าเริ่ม overfit แทนที่จะสรุปว่า generalize ดี เนื่องจาก dataset มีเพียง 20 ภาพ
- การเพิ่ม epochs ช่วยให้มีข้อมูลมากพอสำหรับดูแนวโน้ม แต่ไม่รับประกันว่า loss จะลดแบบ monotonic หรือ metrics จะดีขึ้น

คำสั่งทดลอง:

```powershell
python train.py --img 640 --batch-size 2 --epochs 30 --data '..\datasets\f09_box\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --project runs\f09_box --name exp20_30e
```

### 2026-09-14 — ผลทดลอง 30 epochs (`exp30_epoch`)

- run จบครบ 30 epochs และมี `best.pt` กับ `last.pt` ที่ `yolov5/runs/f09_box/exp30_epoch/weights/`
- `results.csv` มี 30 records สำหรับ epoch 0–29 ไม่มี epoch ซ้ำ
- จาก epoch 0 ถึง epoch 29: `train/box_loss` ลดจาก `0.10221` เป็น `0.076912` และ `train/cls_loss` ลดจาก `0.043962` เป็น `0.03722`
- `train/obj_loss` เพิ่มจาก `0.063952` เป็น `0.074991` และยังแกว่ง จึงยังไม่ใช่การลู่ลงพร้อมกันของ loss ทุกองค์ประกอบ
- ผล epoch 29: precision `0.30842`, recall `0.46296`, mAP50 `0.35436` และ mAP50-95 `0.11719`; เป็นค่าดีที่สุดของ run ตาม mAP50
- validation ที่ epoch 29: box loss `0.044287`, objectness loss `0.055044` และ classification loss `0.02095`
- เมื่อเทียบกับช่วงต้น metrics เพิ่มขึ้นอย่างมีนัยสำคัญ ขณะที่ box/classification losses มีแนวโน้มลดลง จึงสรุปว่า network เรียนรู้และ **เริ่ม converge บางส่วน** หลังพ้น warmup
- ยังไม่ควรเรียกว่า converge สมบูรณ์ เพราะ objectness loss และ recall ผันผวน, curves ยังไม่ plateau อย่างนิ่ง และ validation set มีเพียง 3 ภาพ
- กราฟ `train/total_loss` ระดับ iteration ยังคงแกว่งจาก batch composition และการใช้ within-epoch running mean; ใช้ epoch-level component losses และ validation metrics เป็นหลักในการสรุป

### 2026-09-14 — แนวทางปรับ parameters

- การปรับ parameters อาจเพิ่ม metrics ได้ แต่ไม่รับประกันผล และ validation เพียง 3 ภาพทำให้ค่าผันผวนสูงมาก
- ให้เก็บ `exp30_epoch` เป็น baseline แล้วทดลองเปลี่ยนครั้งละหนึ่งตัวแปร ภายใต้ split และ seed เดิม เพื่อให้เปรียบเทียบได้
- ลำดับที่แนะนำสำหรับชุดข้อมูลขนาดเล็ก: (1) เพิ่มจำนวนภาพและความหลากหลายของ annotations, (2) ทดลองเพิ่ม epochs พร้อมติดตาม validation, (3) ทดลอง freeze pretrained backbone, และ (4) จึงค่อยทดลอง learning rate/optimizer
- การทดลองที่มีความเสี่ยงต่ำคือ 50 epochs ด้วยค่าที่เหลือเหมือน baseline และชื่อ run ใหม่; หาก train loss ลดแต่ validation metrics แย่ลงให้หยุดและรายงาน overfitting
- อีกการทดลองหนึ่งคือ 30 epochs พร้อม `--freeze 10` เพื่อ train ส่วน detection head เป็นหลัก ลดจำนวน parameters ที่ต้องเรียนรู้จากข้อมูลเพียง 14 ภาพ
- เปลี่ยนทีละตัวแปร: ห้ามเปลี่ยน epochs, optimizer, learning rate, augmentation และ freeze พร้อมกัน เพราะจะระบุไม่ได้ว่าผลต่างเกิดจากอะไร
- ใช้ validation split เปรียบเทียบการทดลองเท่านั้น และเก็บ test split 3 ภาพไว้ประเมินโมเดลที่เลือกแล้ว ห้ามใช้ test metrics เลือก parameters
- metrics ที่สูงขึ้นบน validation 3 ภาพยังไม่เพียงพอสำหรับสรุป generalization; วิธีปรับปรุงที่น่าเชื่อถือที่สุดคือเพิ่มข้อมูล train/validation

ตัวอย่าง controlled experiments:

```powershell
# เปลี่ยนเฉพาะจำนวน epochs
python train.py --img 640 --batch-size 2 --epochs 50 --data '..\datasets\f09_box\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --project runs\f09_box --name exp50_epoch

# เปลี่ยนเฉพาะการ freeze backbone โดยเทียบที่ 30 epochs
python train.py --img 640 --batch-size 2 --epochs 30 --data '..\datasets\f09_box\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --freeze 10 --project runs\f09_box --name exp30_freeze10
```

### 2026-09-14 — ล้างและเปลี่ยนชื่อผลการเทรน

- เก็บ `yolov5/runs/f09_box/smoke_tags` และเปลี่ยนชื่อเป็น `yolov5/runs/f09_box/smoke_1epoch_tensorboard` เพื่อระบุว่าเป็น smoke test 1 epoch ที่ใช้ยืนยัน TensorBoard tags
- เก็บ `yolov5/runs/f09_box/exp20_clean` และเปลี่ยนชื่อเป็น `yolov5/runs/f09_box/baseline_5epochs` เพื่อระบุว่าเป็น baseline 5 epochs ที่ไม่มี run ปะปน
- ลบถาวร: `f09_box/exp20`, `f09_box/exp30_epoch`, `f09_box/smoke`, และ `train/exp` ถึง `train/exp7`
- ลบโฟลเดอร์ `yolov5/runs/train` หลังจากไม่มี run เหลือ
- หลัง cleanup เหลือผลการเทรนเพียง `baseline_5epochs` และ `smoke_1epoch_tensorboard`
- ผล generated เหล่านี้ถูก ignore ด้วย `.gitignore` จึงไม่มี weights, event files หรือภาพผลลัพธ์ถูก commit
- metrics ของการทดลอง 30 epochs ยังคงอยู่ในรายงานเพื่อเป็นหลักฐานทางข้อความ แต่ไฟล์ `best.pt`, `last.pt`, TensorBoard event และ artifacts ของ run นั้นถูกลบแล้วและกู้คืนจาก workspace ไม่ได้
- อัปเดตคำสั่ง Step 7 และ Step 8 ใน `Guide.md` ให้ใช้ `baseline_5epochs/weights/best.pt` และระบุ `--device cpu`

### 2026-09-14 — ตรวจ dataset ที่ขยายเป็น 350 ภาพ

- split ใหม่: train 250 ภาพ/250 labels, validation 50 ภาพ/50 labels และ test 50 ภาพ/50 labels; ไม่มี label ว่าง
- ภาพทั้งหมด 350 รูปเป็น JPEG ขนาด 1280×720 และเปิดตรวจด้วย Pillow ได้โดยไม่มีไฟล์เสีย
- จำนวน objects: train 1,754, validation 350 และ test 351 รวม 2,455 objects
- class distribution รวม: class 0 `lane` 700 objects, class 1 `track-line` 1,053 objects และ class 2 `sideway` 702 objects
- ทุก label มี 5 fields ตาม YOLO detection format, class ID อยู่ใน `{0,1,2}`, normalized coordinates อยู่ใน `[0,1]` และ width/height มากกว่า 0
- ชื่อ stem ของภาพและ label จับคู่ครบทุก split ไม่มี orphan label หรือภาพที่ไม่มี label
- ตรวจ SHA-256 ของภาพทั้งหมดแล้วไม่พบไฟล์ภาพซ้ำข้าม train/val/test
- `manifest.csv` มี header และ 350 records สอดคล้องกับจำนวนภาพ
- ข้อผิดพลาดที่ต้องแก้ก่อน train: `datasets/f09_box/data.yaml` ยังระบุ `path: 'D:/CoE Y.4 T.1/241-353/code/F09-TensorBoard/datasets/f09_box'` แต่ workspace ปัจจุบันอยู่ไดรฟ์ C ให้เปลี่ยนเป็น portable path `path: ../datasets/f09_box` ซึ่ง YOLOv5 resolve จากโฟลเดอร์ repository
- smoke test ที่เหมาะสม: CPU, batch size 2, 1 epoch = 125 training iterations ซึ่งผ่าน warmup ขั้นต่ำ 100 iterations และตรวจ pipeline ทั้งชุดได้
- baseline ที่แนะนำหลัง smoke test ผ่าน: 30 epochs = ประมาณ 3,750 training iterations ใช้ชื่อ run ใหม่และไม่ใช้ `--exist-ok`

คำสั่ง smoke test:

```powershell
python train.py --img 640 --batch-size 2 --epochs 1 --data '..\datasets\f09_box\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --project runs\f09_box --name dataset350_smoke
```

คำสั่ง baseline 30 epochs:

```powershell
python train.py --img 640 --batch-size 2 --epochs 30 --data '..\datasets\f09_box\data.yaml' --weights yolov5n.pt --device cpu --workers 0 --project runs\f09_box --name dataset350_baseline30
```

### 2026-09-14 — แก้ Dataset not found ของ expanded dataset

- smoke test ครั้งแรกหยุดก่อนสร้าง dataloader พร้อมข้อความว่าไม่พบ `D:\CoE Y.4 T.1\241-353\code\F09-TensorBoard\datasets\f09_box\images\val`
- สาเหตุคือ `datasets/f09_box/data.yaml` ยังคงใช้ absolute path จาก workspace เดิมบนไดรฟ์ D ขณะที่ repository ปัจจุบันอยู่บนไดรฟ์ C
- แก้ `path` จาก `D:/CoE Y.4 T.1/241-353/code/F09-TensorBoard/datasets/f09_box` เป็น `../datasets/f09_box`
- relative path ถูก resolve โดย YOLOv5 จาก repository root ทำให้ใช้ได้กับ workspace ปัจจุบันและย้าย repository ได้ง่ายกว่า absolute path
- ตรวจด้วย `utils.general.check_dataset()` แล้ว resolve เป็น `C:\Users\student\ai-eco\tensorboard-for-aieco\datasets\f09_box` และพบ train/val/test paths กับ class names ครบ
- การรันที่ล้มเหลวไม่ได้เริ่ม training และไม่ถือเป็นผล smoke test; ต้องรันคำสั่ง `dataset350_smoke` ซ้ำหลังตรวจ path ผ่าน

### 2026-09-14 — รวบรวมหลักฐานผลการรันใน `result/`

- สร้าง `result/RUN_SUMMARY.md` เป็นรายงานแยกสำหรับผลการรันสำคัญ 4 รอบ พร้อม settings, best metrics, ข้อสรุป และลิงก์ภาพหลักฐาน
- สร้าง `result/metrics.csv` เป็นตาราง machine-readable โดยเลือก best epoch จาก mAP50-95 ของแต่ละ run
- สร้าง `result/screenshots/` และคัดลอกกราฟราย run, confusion matrix, PR/F1 curves, label distribution และ validation predictions ด้วยชื่อไฟล์ที่สื่อความหมาย
- เปิด TensorBoard 2.21.0 บน localhost และใช้ Microsoft Edge headless แคป overview ของ runs กับหน้าของ `dataset350_baseline30` โดยเฉพาะ
- run สำคัญที่รวม: `smoke_1epoch_tensorboard`, `baseline_5epochs`, `dataset350_smoke_retry` และ `dataset350_baseline30`
- best result คือ `dataset350_baseline30` epoch 28: precision `0.99648`, recall `1.0`, mAP50 `0.995` และ mAP50-95 `0.79994`
- เพิ่ม `.gitignore` pattern `*.cache.npy` หลังพบ `train.cache.npy` และ `val.cache.npy` ที่สร้างระหว่าง dataset scan เพื่อไม่ให้ generated cache เข้า commit

## นโยบายการบันทึก

ตั้งแต่ 2026-09-14 เป็นต้นไป การเปลี่ยนแปลงโค้ด ผลการทดลอง การตัดสินใจ และการทำงานสำคัญของงานนี้ต้องบันทึกใน `REPORT.md` พร้อมอัปเดตสถานะที่เกี่ยวข้องใน `Guide.md`

### 2026-09-15 — ตรวจความครบถ้วนของ Part 1 เทียบสไลด์ต้นฉบับ

- ตรวจ `Slides-n5-241-353 TensorBoard 2569.pdf` หน้า 17–19 เทียบกับ dataset, run artifacts, TensorBoard event และหลักฐานใน `result/`
- ยืนยันว่า `dataset350_baseline30` มี 30 epochs หรือ 3,750 training iterations, มี `best.pt`, และ TensorBoard มี `train/total_loss` กับ `train/learning_rate` อย่างละ 3,750 steps
- พบว่า `train/box_loss`, `train/obj_loss` และ `train/cls_loss` ของ run ปัจจุบันมี 30 steps ระดับ epoch จึงเพิ่มทางเลือกสำหรับสร้าง iteration-level tags และรัน final ใหม่ไว้ใน `Guide.md`
- พบว่ายังไม่มี standalone evaluation บน test split และไม่มี inference samples จาก test split 3 ภาพ ภาพ prediction ที่เก็บอยู่เดิมเป็น validation batches
- พบว่าภาพ TensorBoard เดิมพับกลุ่ม training losses, metrics และ LR อยู่ จึงยังใช้เป็นหลักฐานกราฟครบทุกข้อในรายงานไม่ได้
- เพิ่มคำสั่ง test evaluation, test inference, หลักเกณฑ์เลือกตัวอย่าง 3 ภาพ, รายการ screenshots ที่ต้องเก็บ, โครงคำตอบ convergence และ final checklist ใน `Guide.md`
- บันทึก decision gate เรื่อง split ปัจจุบัน 250/50/50 ซึ่งต่างจากข้อความ train 100/test 100 ในสไลด์ เพื่อป้องกันการผสมผลจากคนละ split ในรายงาน

### 2026-09-15 — สร้าง dataset subsets ขนาด 30/60/150/350

- เพิ่ม `create_f09_box_subsets.py` เพื่อสร้าง physical YOLO datasets แบบ deterministic nested subsets จาก `datasets/f09_box` ด้วย seed `2569`
- สร้าง `datasets/f09_box_30` เป็น train/val/test 22/4/4, `f09_box_60` เป็น 42/9/9, `f09_box_150` เป็น 108/21/21 และ `f09_box_350` เป็น 250/50/50
- แต่ละชุดมี images, labels, `manifest.csv`, portable `data.yaml` และ `README.md` ครบ
- ตรวจแล้ว image/label stems จับคู่ตรงกันทุก split, manifest มี 30/60/150/350 records, ชุดเล็กเป็น subset ของชุดใหญ่ และ YOLOv5 `check_dataset()` resolve YAML ทั้งสี่ชุดผ่าน
- ยังไม่ได้เริ่ม training ชุดเปรียบเทียบตามคำสั่งผู้ใช้ ผล `dataset350_final_iterloss` ที่เริ่มก่อนคำสั่งหยุดถูกยุติระหว่าง epoch 4 และลบไปพร้อม worktree จึงห้ามนับเป็น final run
- smoke run ของ Step 6A จบครบ 125 iterations และยืนยันว่า iteration tags ทั้งห้ามี steps 0–124, ไม่มี NaN และ `total_loss` เท่ากับผลรวมสาม component losses ภายใน tolerance ของ float
- commit `708f301` (`feat: log detection losses per iteration`) ถูก fast-forward เข้า `main` แล้ว จากนั้นลบ feature worktree และ branch `feat/tensorboard-iteration-losses` ตามคำสั่งผู้ใช้ โดยคัดลอก smoke evidence กลับมาไว้ที่ `yolov5/runs/f09_box/dataset350_iterloss_smoke` ก่อนลบ

### 2026-09-15 — ล้างผลการ train เพื่อเริ่มการทดลองใหม่

- ลบ `yolov5/runs/` ทั้งหมด รวม run directories, `best.pt`, `last.pt`, TensorBoard event files, plots, validation images และ artifacts จาก smoke/final attempts เดิม
- ลบ `result/` ทั้งหมด รวม `RUN_SUMMARY.md`, `metrics.csv` และ screenshots เดิม
- ลบ `train.cache`, `val.cache`, `train.cache.npy` และ `val.cache.npy` ใต้ `datasets/f09_box/labels/`
- คง source code, virtual environment, pretrained `yolov5n.pt` และ dataset folders 30/60/150/350 ไว้
- ผลและ metrics ที่บันทึกก่อนหัวข้อนี้เป็นประวัติการทำงานเท่านั้น ต้องไม่ใช้เป็นหลักฐานของการทดลองรอบใหม่

### 2026-09-15 — จัดทำแผน train เปรียบเทียบ dataset 30/60/150/350

- กำหนดให้ทั้งสี่ runs ใช้ YOLOv5n, 30 epochs, batch size 2, image size 640, CPU, workers 0, seed 2569 และ hyperparameters เดียวกัน โดยเปลี่ยนเฉพาะ dataset
- กำหนดลำดับ preflight ด้วย dataset 30 ก่อน full training ตามลำดับ 30 → 60 → 150 → 350 พร้อม expected iteration steps 330/630/1,620/3,750
- เพิ่มคำสั่ง train ครบสี่ชุด, เกณฑ์ตรวจ NaN/epochs/event tags/weights หลังแต่ละ run และประมาณเวลาบน CPU
- กำหนดให้ประเมิน `best.pt` ทั้งสี่ตัวบน common test set 50 ภาพจาก `f09_box_350` เพื่อไม่เปรียบเทียบคะแนนจาก test sets ที่มีขนาดต่างกัน
- เพิ่มแผน TensorBoard comparison, ตารางบันทึกผล และเกณฑ์เลือก final model จาก common-test mAP50-95 ร่วมกับความเสถียรของ loss และ qualitative inference
- การแก้ครั้งนี้เป็นการจัดทำแผนเท่านั้น ยังไม่ได้เริ่ม training

### 2026-09-15 — รันการทดลอง dataset 30/60/150/350 และจัดทำหลักฐาน

- รัน preflight บน `f09_box_30` 1 epoch ผ่าน: iteration tags ทั้งห้ามีอย่างละ 11 steps (0–10), ทุกค่า finite และ `total_loss` ตรงกับผลรวม component losses ภายใน floating-point tolerance
- รัน YOLOv5n ใหม่ครบ 30 epochs ด้วย batch size 2, image size 640, CPU, workers 0, seed 2569 และ default `hyp.scratch-low.yaml` โดยเปลี่ยนเฉพาะ dataset ตามลำดับ 30 → 60 → 150 → 350
- ยืนยัน event counts ของ full runs เป็น 330/630/1,620/3,750 จุดต่อ tag, step ต่อเนื่อง, ไม่มี NaN/Inf, `results.csv` มี 30 epochs และทุก run มี `best.pt`/`last.pt`
- best validation mAP50-95 ของขนาด 30/60/150/350 เท่ากับ 0.096851/0.33475/0.70766/0.81224 ที่ epoch 28/18/28/29 ตามลำดับ
- ประเมิน `best.pt` ทั้งสี่ตัวบน common test split ของ `f09_box_350` เดียวกัน 50 ภาพ/351 objects ได้ mAP50-95 เท่ากับ 0.0706/0.249/0.652/0.725 และ mAP50 เท่ากับ 0.212/0.546/0.984/0.989
- เลือก `size350_e30_seed2569/weights/best.pt` เป็น final model จาก common-test mAP50-95 สูงสุด และรัน inference บน test set ครบ 50 ภาพ ก่อนคัดภาพต้น/กลาง/ท้าย 3 ภาพไว้เป็นหลักฐาน
- สร้าง `result/RUN_SUMMARY.md`, `result/metrics.csv`, raw common-test logs, TensorBoard UI screenshot, กราฟ iteration losses/validation metrics/LR/common-test comparison และ inference screenshots 3 ภาพ
- เพิ่ม `create_experiment_results.py` เพื่อสร้าง metrics, ตรวจ event invariants และ export กราฟจาก TensorBoard event data ซ้ำได้
- บันทึกข้อจำกัดว่า dataset เป็นลำดับเฟรมที่สัมพันธ์กัน จึงยังไม่เพียงพอสำหรับยืนยัน generalization ต่อสภาพแวดล้อมใหม่

### 2026-09-15 — เพิ่ม screenshots แยกตามรอบการเทรน

- เพิ่ม `result/screenshots/per_run/size30_tensorboard_dashboard.png`, `size60_tensorboard_dashboard.png`, `size150_tensorboard_dashboard.png` และ `size350_tensorboard_dashboard.png`
- dashboard แต่ละภาพแสดง box/objectness/classification/total loss ระดับ iteration, scheduled learning rate และ validation metrics ของ run นั้นโดยไม่รวมเส้นจาก run อื่น
- คัดลอกกราฟ `results.png` ต้นฉบับของ YOLOv5 แยกเป็น `size30_yolov5_results.png`, `size60_yolov5_results.png`, `size150_yolov5_results.png` และ `size350_yolov5_results.png`
- ตรวจเปิด dashboard ทั้งสี่ภาพแล้ว ชื่อ run, จำนวน iterations และแกนกราฟอ่านได้ครบ

### 2026-09-15 — จัดทำรายงานฉบับพร้อมส่ง

- เพิ่ม `result/FINAL_REPORT.md` ภาษาไทย พร้อมวัตถุประสงค์, dataset splits, controlled settings, event verification, ตาราง validation/common-test metrics และเหตุผลเลือก final model
- ฝังภาพ TensorBoard, กราฟเปรียบเทียบ, dashboard แยกทั้งสี่ run และ inference ต้น/กลาง/ท้าย พร้อมคำอธิบายภาพ
- ระบุชัดเจนว่าใช้ split 250/50/50 ซึ่งต่างจากตัวอย่าง 100/100 ในสไลด์ และใช้ common test 50 ภาพที่ไม่ถูกใช้ train
- เพิ่มคำตอบ convergence พร้อมค่าจริงและข้อจำกัดจากข้อมูลที่เป็นลำดับเฟรม
- พัก weight distribution ไว้เนื่องจากเป็นส่วนเสริมและผู้ใช้ยืนยันว่าอาจารย์ไม่ได้กำหนด
- สร้าง `result/FINAL_REPORT.html` และ `result/FINAL_REPORT.pdf` จาก Markdown โดย PDF ฝังตารางและภาพไว้ในไฟล์เดียวสำหรับอัปโหลดส่งงาน
- ตรวจ render หน้าแรกของรายงานแล้ว ภาษาไทย ตาราง และรูปแบบหัวข้อแสดงผลถูกต้อง พร้อมลบ browser profile และภาพ preview ชั่วคราวออกจาก `result/`

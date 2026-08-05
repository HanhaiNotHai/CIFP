# CIFP dataset audit

This report is generated read-only. No sample was deleted, moved, or filtered.

## ForenSynths

- Root: `/data/zhy/CNNDetection/dataset`
- Images: 728,119
- Image headers checked: 1,000 (complete=False)
- Smaller than crop: 0
- Corrupt/unreadable: 0
- Symlinks: 0
- Duplicate hard-linked/resolved paths: 0

### Bounded tree

```text
./
test/
test/CNN_synth_testset.zip
test/download_testset.sh
train/
train/download_trainset.sh
train/airplane/
train/bicycle/
train/bird/
train/boat/
train/bottle/
train/bus/
train/car/
train/cat/
train/chair/
train/cow/
train/diningtable/
train/dog/
train/horse/
train/motorbike/
train/person/
train/pottedplant/
train/sheep/
train/sofa/
train/train/
train/tvmonitor/
val/
val/download_valset.sh
val/airplane/
val/bicycle/
val/bird/
val/boat/
val/bottle/
val/bus/
val/car/
val/cat/
val/chair/
val/cow/
val/diningtable/
val/dog/
val/horse/
val/motorbike/
val/person/
val/pottedplant/
val/sheep/
val/sofa/
val/train/
val/tvmonitor/
```

### Extension counts

```json
{
  ".png": 728119
}
```

### Per-group counts

```json
{
  "train/airplane/0_real": 18003,
  "train/airplane/1_fake": 18003,
  "train/bicycle/0_real": 18003,
  "train/bicycle/1_fake": 18003,
  "train/bird/0_real": 18003,
  "train/bird/1_fake": 18003,
  "train/boat/0_real": 18002,
  "train/boat/1_fake": 18003,
  "train/bottle/0_real": 18003,
  "train/bottle/1_fake": 18003,
  "train/bus/0_real": 18003,
  "train/bus/1_fake": 18003,
  "train/car/0_real": 18003,
  "train/car/1_fake": 18003,
  "train/cat/0_real": 18003,
  "train/cat/1_fake": 18003,
  "train/chair/0_real": 18003,
  "train/chair/1_fake": 18003,
  "train/cow/0_real": 18003,
  "train/cow/1_fake": 18003,
  "train/diningtable/0_real": 18003,
  "train/diningtable/1_fake": 18003,
  "train/dog/0_real": 18003,
  "train/dog/1_fake": 18003,
  "train/horse/0_real": 18003,
  "train/horse/1_fake": 18003,
  "train/motorbike/0_real": 18003,
  "train/motorbike/1_fake": 18003,
  "train/person/0_real": 18003,
  "train/person/1_fake": 18003,
  "train/pottedplant/0_real": 18003,
  "train/pottedplant/1_fake": 18003,
  "train/sheep/0_real": 18003,
  "train/sheep/1_fake": 18003,
  "train/sofa/0_real": 18003,
  "train/sofa/1_fake": 18003,
  "train/train/0_real": 18003,
  "train/train/1_fake": 18003,
  "train/tvmonitor/0_real": 18003,
  "train/tvmonitor/1_fake": 18003,
  "val/airplane/0_real": 200,
  "val/airplane/1_fake": 200,
  "val/bicycle/0_real": 200,
  "val/bicycle/1_fake": 200,
  "val/bird/0_real": 200,
  "val/bird/1_fake": 200,
  "val/boat/0_real": 200,
  "val/boat/1_fake": 200,
  "val/bottle/0_real": 200,
  "val/bottle/1_fake": 200,
  "val/bus/0_real": 200,
  "val/bus/1_fake": 200,
  "val/car/0_real": 200,
  "val/car/1_fake": 200,
  "val/cat/0_real": 200,
  "val/cat/1_fake": 200,
  "val/chair/0_real": 200,
  "val/chair/1_fake": 200,
  "val/cow/0_real": 200,
  "val/cow/1_fake": 200,
  "val/diningtable/0_real": 200,
  "val/diningtable/1_fake": 200,
  "val/dog/0_real": 200,
  "val/dog/1_fake": 200,
  "val/horse/0_real": 200,
  "val/horse/1_fake": 200,
  "val/motorbike/0_real": 200,
  "val/motorbike/1_fake": 200,
  "val/person/0_real": 200,
  "val/person/1_fake": 200,
  "val/pottedplant/0_real": 200,
  "val/pottedplant/1_fake": 200,
  "val/sheep/0_real": 200,
  "val/sheep/1_fake": 200,
  "val/sofa/0_real": 200,
  "val/sofa/1_fake": 200,
  "val/train/0_real": 200,
  "val/train/1_fake": 200,
  "val/tvmonitor/0_real": 200,
  "val/tvmonitor/1_fake": 200
}
```

### Candidates

- Splits: ['test', 'train', 'val']
- Real/fake labels: ['0_real', '1_fake']
- Generator/source candidates: []
- Semantic class candidates: ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

### Problem paths

```json
{
  "small_images": [],
  "corrupt_images": [],
  "corrupt_image_errors": {},
  "duplicate_paths": [],
  "symlinks": []
}
```

## GenImage

- Root: `/data/zhy/GenImage`
- Images: 2,681,167
- Image headers checked: 1,000 (complete=False)
- Smaller than crop: 0
- Corrupt/unreadable: 0
- Symlinks: 0
- Duplicate hard-linked/resolved paths: 0

### Bounded tree

```text
./
ADM/
ADM/train/
ADM/val/
BigGAN/
BigGAN/train/
BigGAN/val/
Midjourney/
Midjourney/train/
Midjourney/val/
VQDM/
VQDM/train/
VQDM/val/
glide/
glide/train/
glide/val/
stable_diffusion_v_1_4/
stable_diffusion_v_1_4/train/
stable_diffusion_v_1_4/val/
stable_diffusion_v_1_5/
stable_diffusion_v_1_5/train/
stable_diffusion_v_1_5/val/
wukong/
wukong/train/
wukong/val/
```

### Extension counts

```json
{
  ".jpeg": 1331167,
  ".png": 1350000
}
```

### Per-group counts

```json
{
  "ADM/train/ai": 162000,
  "ADM/train/nature": 157453,
  "ADM/val/ai": 6000,
  "ADM/val/nature": 6000,
  "BigGAN/train/ai": 162000,
  "BigGAN/train/nature": 162000,
  "BigGAN/val/ai": 6000,
  "BigGAN/val/nature": 6000,
  "Midjourney/train/ai": 162000,
  "Midjourney/train/nature": 161701,
  "Midjourney/val/ai": 6000,
  "Midjourney/val/nature": 6000,
  "VQDM/train/ai": 162000,
  "VQDM/train/nature": 162000,
  "VQDM/val/ai": 6000,
  "VQDM/val/nature": 6000,
  "glide/train/ai": 162000,
  "glide/train/nature": 162000,
  "glide/val/ai": 6000,
  "glide/val/nature": 6000,
  "stable_diffusion_v_1_4/train/ai": 162000,
  "stable_diffusion_v_1_4/train/nature": 162000,
  "stable_diffusion_v_1_4/val/ai": 6000,
  "stable_diffusion_v_1_4/val/nature": 6000,
  "stable_diffusion_v_1_5/train/ai": 166000,
  "stable_diffusion_v_1_5/train/nature": 153274,
  "stable_diffusion_v_1_5/val/ai": 8000,
  "stable_diffusion_v_1_5/val/nature": 8000,
  "wukong/train/ai": 162000,
  "wukong/train/nature": 160739,
  "wukong/val/ai": 6000,
  "wukong/val/nature": 6000
}
```

### Candidates

- Splits: ['train', 'val']
- Real/fake labels: ['ai', 'nature']
- Generator/source candidates: ['ADM', 'BigGAN', 'Midjourney', 'VQDM', 'glide', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'wukong']
- Semantic class candidates: []

### Problem paths

```json
{
  "small_images": [],
  "corrupt_images": [],
  "corrupt_image_errors": {},
  "duplicate_paths": [],
  "symlinks": []
}
```

## Self-Synthesis

- Root: `/data/zhy/GANGen-Detection`
- Images: 36,000
- Image headers checked: 1,000 (complete=False)
- Smaller than crop: 0
- Corrupt/unreadable: 0
- Symlinks: 0
- Duplicate hard-linked/resolved paths: 0

### Bounded tree

```text
./
./tarx.sh
AttGAN/
AttGAN/0_real/
AttGAN/0_real/10001.png
AttGAN/0_real/10020.png
AttGAN/0_real/10021.png
AttGAN/0_real/10024.png
AttGAN/0_real/10029.png
AttGAN/1_fake/
AttGAN/1_fake/182638_0.png
AttGAN/1_fake/182638_1.png
AttGAN/1_fake/182638_10.png
AttGAN/1_fake/182638_11.png
AttGAN/1_fake/182638_12.png
BEGAN/
BEGAN/0_real/
BEGAN/0_real/000001_crop.png
BEGAN/0_real/000002_crop.png
BEGAN/0_real/000005_crop.png
BEGAN/0_real/000006_crop.png
BEGAN/0_real/000007_crop.png
BEGAN/1_fake/
BEGAN/1_fake/BEGAN_00000000.png
BEGAN/1_fake/BEGAN_00000001.png
BEGAN/1_fake/BEGAN_00000002.png
BEGAN/1_fake/BEGAN_00000003.png
BEGAN/1_fake/BEGAN_00000004.png
CramerGAN/
CramerGAN/0_real/
CramerGAN/0_real/000008.png
CramerGAN/0_real/000021.png
CramerGAN/0_real/000034.png
CramerGAN/0_real/000049.png
CramerGAN/0_real/000070.png
CramerGAN/1_fake/
CramerGAN/1_fake/CRAMER_00000000.png
CramerGAN/1_fake/CRAMER_00000001.png
CramerGAN/1_fake/CRAMER_00000002.png
CramerGAN/1_fake/CRAMER_00000003.png
CramerGAN/1_fake/CRAMER_00000004.png
InfoMaxGAN/
InfoMaxGAN/0_real/
InfoMaxGAN/0_real/000008.png
InfoMaxGAN/0_real/000021.png
InfoMaxGAN/0_real/000034.png
InfoMaxGAN/0_real/000049.png
InfoMaxGAN/0_real/000070.png
InfoMaxGAN/1_fake/
InfoMaxGAN/1_fake/InfoMaxGAN_00000000.png
InfoMaxGAN/1_fake/InfoMaxGAN_00000001.png
InfoMaxGAN/1_fake/InfoMaxGAN_00000002.png
InfoMaxGAN/1_fake/InfoMaxGAN_00000003.png
InfoMaxGAN/1_fake/InfoMaxGAN_00000004.png
MMDGAN/
MMDGAN/0_real/
MMDGAN/0_real/000008.png
MMDGAN/0_real/000021.png
MMDGAN/0_real/000034.png
MMDGAN/0_real/000049.png
MMDGAN/0_real/000070.png
MMDGAN/1_fake/
MMDGAN/1_fake/MMD_00000000.png
MMDGAN/1_fake/MMD_00000001.png
MMDGAN/1_fake/MMD_00000002.png
MMDGAN/1_fake/MMD_00000003.png
MMDGAN/1_fake/MMD_00000004.png
RelGAN/
RelGAN/0_real/
RelGAN/0_real/0.png
RelGAN/0_real/1.png
RelGAN/0_real/10.png
RelGAN/0_real/100.png
RelGAN/0_real/1000.png
RelGAN/1_fake/
RelGAN/1_fake/RelGAN_00000000.png
RelGAN/1_fake/RelGAN_00000001.png
RelGAN/1_fake/RelGAN_00000002.png
RelGAN/1_fake/RelGAN_00000003.png
RelGAN/1_fake/RelGAN_00000004.png
S3GAN/
S3GAN/0_real/
S3GAN/0_real/n01440764_20434.png
S3GAN/0_real/n01440764_4360.png
S3GAN/0_real/n01440764_6443.png
S3GAN/0_real/n01440764_6468.png
S3GAN/0_real/n01440764_7004.png
S3GAN/1_fake/
S3GAN/1_fake/S3GAN_00000000.png
S3GAN/1_fake/S3GAN_00000001.png
S3GAN/1_fake/S3GAN_00000002.png
S3GAN/1_fake/S3GAN_00000003.png
S3GAN/1_fake/S3GAN_00000004.png
SNGAN/
SNGAN/0_real/
SNGAN/0_real/000008.png
SNGAN/0_real/000021.png
SNGAN/0_real/000034.png
SNGAN/0_real/000049.png
SNGAN/0_real/000070.png
SNGAN/1_fake/
SNGAN/1_fake/SNGAN_00000000.png
SNGAN/1_fake/SNGAN_00000001.png
SNGAN/1_fake/SNGAN_00000002.png
SNGAN/1_fake/SNGAN_00000003.png
SNGAN/1_fake/SNGAN_00000004.png
STGAN/
STGAN/0_real/
STGAN/0_real/000001.png
STGAN/0_real/000002.png
STGAN/0_real/000003.png
STGAN/0_real/000004.png
STGAN/0_real/000005.png
STGAN/1_fake/
STGAN/1_fake/000000_Bangs-Brown_Hair-Bushy_Eyebrows.png
STGAN/1_fake/000000_Black_Hair-No_Beard-Brown_Hair.png
STGAN/1_fake/000000_Brown_Hair-Eyeglasses-Young.png
STGAN/1_fake/000000_Eyeglasses-Brown_Hair-Pale_Skin.png
STGAN/1_fake/000000_Male-Bald-Blond_Hair.png
```

### Extension counts

```json
{
  ".png": 36000
}
```

### Per-group counts

```json
{
  "AttGAN/0_real": 2000,
  "AttGAN/1_fake": 2000,
  "BEGAN/0_real": 2000,
  "BEGAN/1_fake": 2000,
  "CramerGAN/0_real": 2000,
  "CramerGAN/1_fake": 2000,
  "InfoMaxGAN/0_real": 2000,
  "InfoMaxGAN/1_fake": 2000,
  "MMDGAN/0_real": 2000,
  "MMDGAN/1_fake": 2000,
  "RelGAN/0_real": 2000,
  "RelGAN/1_fake": 2000,
  "S3GAN/0_real": 2000,
  "S3GAN/1_fake": 2000,
  "SNGAN/0_real": 2000,
  "SNGAN/1_fake": 2000,
  "STGAN/0_real": 2000,
  "STGAN/1_fake": 2000
}
```

### Candidates

- Splits: []
- Real/fake labels: ['0_real', '1_fake']
- Generator/source candidates: ['AttGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN', 'SNGAN', 'STGAN']
- Semantic class candidates: []

### Problem paths

```json
{
  "small_images": [],
  "corrupt_images": [],
  "corrupt_image_errors": {},
  "duplicate_paths": [],
  "symlinks": []
}
```

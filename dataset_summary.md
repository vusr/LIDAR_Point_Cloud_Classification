# Dataset Summary

## Overview

The dataset consists of pre-segmented LiDAR point cloud clusters across four classes: **background**, **bicyclist**, **car**, and **pedestrian**. Data is split into a training set and a test set.

| Metric | Value |
|---|---|
| Total files (train + test) | 52,630 |
| Total point cloud files (train) | 42,090 |
| Total point cloud files (test) | 10,540 |
| Total points (train) | 8,905,627 |
| Total points (test) | 2,140,190 |
| Classes | background, bicyclist, car, pedestrian |

---

## File Counts per Class

| Class | Train | Test | Total |
|---|---|---|---|
| background | 32,520 | 8,135 | 40,655 |
| bicyclist | 1,305 | 330 | 1,635 |
| car | 7,000 | 1,755 | 8,755 |
| pedestrian | 1,265 | 320 | 1,585 |
| **Total** | **42,090** | **10,540** | **52,630** |

---

## Points per File Statistics

### Train Set

| Class | Min | Max | Mean | Median | Total Points |
|---|---|---|---|---|---|
| background | 6 | 16,022 | 173.3 | 94 | 5,636,022 |
| bicyclist | 7 | 3,790 | 299.5 | 72 | 390,813 |
| car | 6 | 11,130 | 372.5 | 148 | 2,607,575 |
| pedestrian | 6 | 2,449 | 214.4 | 93 | 271,217 |

### Test Set

| Class | Min | Max | Mean | Median | Total Points |
|---|---|---|---|---|---|
| background | 6 | 8,346 | 170.9 | 90 | 1,390,647 |
| bicyclist | 7 | 3,868 | 304.5 | 63 | 100,500 |
| car | 6 | 8,956 | 337.1 | 150 | 591,690 |
| pedestrian | 8 | 2,890 | 179.2 | 71 | 57,353 |

---

## Class Distribution

The class distribution is heavily imbalanced and is nearly identical across both splits, suggesting a consistent stratified structure.

### Train (42,090 total files)

| Class | Files | Share |
|---|---|---|
| background | 32,520 | 77.3% |
| car | 7,000 | 16.6% |
| bicyclist | 1,305 | 3.1% |
| pedestrian | 1,265 | 3.0% |

### Test (10,540 total files)

| Class | Files | Share |
|---|---|---|
| background | 8,135 | 77.2% |
| car | 1,755 | 16.7% |
| bicyclist | 330 | 3.1% |
| pedestrian | 320 | 3.0% |

> **Note:** The dataset is heavily class-imbalanced. Background accounts for ~77% of all samples in both splits, while bicyclist and pedestrian together make up only ~6%.

---

## Spatial Statistics (x, y, z ranges)

Sampled from the first 20 files per class. Coordinates are in meters, centered/normalized around the object origin.

### Train Set

| Class | x range | y range | z range |
|---|---|---|---|
| background | [-1.19, 0.92] | [-1.16, 1.04] | [-1.69, 1.44] |
| bicyclist | [-0.58, 0.72] | [-0.74, 1.01] | [-0.95, 1.98] |
| car | [-2.93, 4.46] | [-2.28, 2.06] | [-1.03, 1.15] |
| pedestrian | [-0.79, 0.74] | [-0.66, 1.02] | [-0.95, 0.90] |

### Test Set

| Class | x range | y range | z range |
|---|---|---|---|
| background | [-1.77, 2.10] | [-1.66, 2.47] | [-1.53, 1.36] |
| bicyclist | [-0.93, 0.77] | [-0.66, 0.95] | [-1.29, 1.68] |
| car | [-1.18, 3.03] | [-3.18, 2.77] | [-0.89, 1.09] |
| pedestrian | [-0.78, 0.68] | [-0.80, 0.74] | [-0.94, 0.87] |

**Key observations:**
- **Car** clusters span the largest spatial extent, especially along x and y, consistent with the larger physical size of vehicles.
- **Pedestrian** and **bicyclist** clusters are comparably compact in x and y, but bicyclists tend to be taller (greater z range).
- **Background** clusters vary widely, reflecting noise, vegetation, or other non-object elements.

---

## Optional Challenge Data

The optional challenge dataset contains 10 sequential LiDAR scene frames (no pre-segmentation). Each point has 5 channels: **x, y, z, intensity, ring**.

| Metric | Value |
|---|---|
| Scenes (files) | 10 |
| Min points per scene | 183,642 |
| Max points per scene | 184,922 |
| Mean points per scene | 184,385.5 |
| Total points | 1,843,855 |
| Channels per point | 5 (x, y, z, intensity, ring) |

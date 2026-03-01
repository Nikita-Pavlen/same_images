# Similar Image Detection System — Plan

> **Approach**: Queue-based image categorization. The script picks images
> one by one from `py-images/`, compares each against existing category
> embeddings (held in a local numpy cache for speed), and assigns it to
> the best-matching category. If no category scores above the threshold,
> a new category is created with the current image as its representative.
> All data is persisted to Milvus. Matched images are copied to `py-result-images/{category_id}/`.

---

## Step 0 — Start Milvus [DONE]

Make sure all three containers are running:

```powershell
docker-compose up -d
```

Verify health:

```powershell
curl http://localhost:9091/healthz
```

---

## Step 1 — Set Up the Python Environment [DONE]

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install pymilvus open-clip-torch torch torchvision Pillow numpy tqdm
```

For GPU support, reinstall PyTorch with CUDA:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Model**: CLIP ViT-B/32 (512-dim embeddings, free, MIT license).

---

## Step 2 — Create Milvus Collections [DONE]

Two collections, one script each:

### `categories` — one row per category (searched against)

Script: `02_create_categories_collection.py`

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 | Primary key, auto-id |
| `category_id` | VARCHAR(36) | UUID — unique category identifier |
| `embedding` | FLOAT_VECTOR(512) | CLIP embedding of the main image |

Index: **IVF_FLAT**, metric **IP**, nlist=1024.

### `images` — every processed image

Script: `01_create_images_collection.py`

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 | Primary key, auto-id |
| `filepath` | VARCHAR(512) | Relative path to the image |
| `category_id` | VARCHAR(36) | UUID — links to `categories.category_id` |
| `embedding` | FLOAT_VECTOR(512) | CLIP embedding |

Index: **IVF_FLAT**, metric **IP**, nlist=1024.

---

## Step 3 — Categorize Images (queue processor) [DONE]

Script: `03_categorize.py`

This is the core step. Works as a **queue**: picks images one by one from
`py-images/`, compares each against category embeddings, and decides
whether to assign it to an existing category or create a new one.
Also copies each image into `py-result-images/{category_id}/`.

### Algorithm

```
queue = sorted list of all files in py-images/
skip already-processed files (from checkpoint.json)

for each batch of 64 images:
    1. Batch-compute CLIP embeddings (GPU/CPU)
    2. L2-normalize all embeddings
    3. For each image in the batch:
       a. Search local numpy cache for top-1 nearest category
       b. IF cache is empty OR best score < THRESHOLD (0.70):
            → generate new category_id (uuid4)
            → add embedding to local cache
            → stage for insert into `categories` collection
          ELSE:
            → use matched category_id
       c. Stage for insert into `images` collection
       d. Copy image file to py-result-images/{category_id}/
    4. Batch-insert new categories into Milvus
    5. Batch-insert all images into Milvus
    6. Update checkpoint.json
```

### Key design decisions

| Concern | Approach |
|---|---|
| Queue order | Sorted file list for deterministic, resumable processing |
| One-by-one | Each image is categorized individually — a new category created by image N is immediately available for image N+1 |
| Resumability | `checkpoint.json` stores the index of the last processed file; on restart, existing categories are loaded from Milvus into the local cache |
| Embedding speed | Batch-embed 64 images at once, then categorize them one-by-one; uses fp16 on GPU |
| Fast search | All category embeddings stay in a local numpy array; similarity = single matrix multiply (< 1ms even for 100K categories). Milvus is only used for persistence. |
| Corrupt files | try/except around image loading; skip and log to `errors.log` |
| File output | Images are copied to `py-result-images/{category_id}/` during categorization — no separate organize step needed |

---

## Step 4 — Tune & Iterate

| What to tune | Why |
|---|---|
| **THRESHOLD** (currently 0.70) | Too low → unrelated images grouped; too high → too many single-image categories. Inspect results and adjust. |
| **Model** | CLIP ViT-B/32 is the baseline. If results are poor, try ViT-L/14 (768-dim, more accurate but slower). |

> **Tip**: Before running on 1M images, test on a small subset (500-1000) to calibrate the threshold.
> To re-run with a different threshold: reset collections (run scripts 01 + 02), reset checkpoint, delete `py-result-images/`, then run `03_categorize.py`.

---

## Summary of Scripts

| # | Script | Purpose | Status |
|---|--------|---------|--------|
| 1 | `01_create_images_collection.py` | Create `images` collection + index | DONE |
| 2 | `02_create_categories_collection.py` | Create `categories` collection + index | DONE |
| 3 | `03_categorize.py` | Queue: embed images, match/create categories, insert into Milvus, copy to output dir | DONE |

## Performance (798 test images)

| Device | Time | Speed |
|---|---|---|
| CPU | ~5 min | ~2.7 img/s |
| GPU (CUDA) | ~3.5 min | ~3.6 img/s |

> GPU speed is close to CPU here because the bottleneck is image loading and file
> copying on disk, not model inference. For 1M images with optimized data loading
> (DataLoader + workers), GPU would be significantly faster.

## Important Tips

- **GPU recommended** for large runs. Install PyTorch with CUDA for GPU support.
- **Disk space**: embeddings in Milvus at 512-dim float32 ≈ 2 GB for 1M images — well within standalone capacity.
- **Checkpoint**: progress is tracked in `checkpoint.json` so the pipeline can be stopped and resumed at any point.
- **Errors**: corrupt or unreadable images are logged to `errors.log` and skipped.

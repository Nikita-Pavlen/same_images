import os
import json
import uuid
import glob
import shutil
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import open_clip

from pymilvus import connections, Collection

# ── Config ───────────────────────────────────────────────────────────────
IMAGES_DIR = "py-images"
OUTPUT_DIR = "py-result-images"
CHECKPOINT_FILE = "checkpoint.json"
ERROR_LOG = "errors.log"

THRESHOLD = 0.70
EMBED_BATCH_SIZE = 256
NUM_WORKERS = 4
COPY_THREADS = 4
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ── Checkpoint ───────────────────────────────────────────────────────────
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    step3 = data.get("steps", {}).get("step_3_categorize_images", {})
    return step3.get("last_processed_index", 0)


def save_checkpoint(last_index, total, categories_count):
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"steps": {}, "config": {}}

    data["steps"]["step_3_categorize_images"] = {
        "status": "in_progress" if last_index < total else "done",
        "script": "03_categorize.py",
        "last_processed_index": last_index,
        "total_images": total,
        "categories_created": categories_count,
    }
    data["config"]["output_dir"] = OUTPUT_DIR
    data["config"]["threshold"] = THRESHOLD

    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Local category cache (numpy search — faster than Milvus round-trip) ─
class CategoryCache:
    """
    Keeps all category embeddings in RAM so each similarity lookup is a
    single numpy matmul (~<1 ms even for 100K categories).
    Milvus is only used for persistence.
    """

    def __init__(self):
        self.ids: list[str] = []
        self._matrix: np.ndarray | None = None  # (N, dim) lazily built

    def load_from_milvus(self, col: Collection):
        col.load()
        if col.num_entities == 0:
            return
        rows = col.query(expr="id >= 0", output_fields=["category_id", "embedding"])
        for row in rows:
            self.ids.append(row["category_id"])
        if rows:
            self._matrix = np.array(
                [r["embedding"] for r in rows], dtype=np.float32
            )
        print(f"  Loaded {len(self.ids)} existing categories from Milvus")

    def add(self, category_id: str, embedding: np.ndarray):
        self.ids.append(category_id)
        emb = embedding.reshape(1, -1)
        if self._matrix is None:
            self._matrix = emb.copy()
        else:
            self._matrix = np.vstack([self._matrix, emb])

    def search(self, embedding: np.ndarray) -> tuple[str | None, float]:
        if self._matrix is None or len(self.ids) == 0:
            return None, 0.0
        scores = self._matrix @ embedding  # cosine sim (vectors are L2-normed)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= THRESHOLD:
            return self.ids[best_idx], best_score
        return None, best_score

    def __len__(self):
        return len(self.ids)


# ── Dataset for DataLoader (parallel image loading) ─────────────────────
class ImagePathDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.preprocess(img)
            return tensor, idx, True
        except Exception as e:
            log.warning("skip %s: %s", path, e)
            return torch.zeros(3, 224, 224), idx, False


def collate_fn(batch):
    tensors, indices, valids = zip(*batch)
    return torch.stack(tensors), list(indices), list(valids)


# ── CLIP helpers ─────────────────────────────────────────────────────────
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    return model, preprocess, device


def embed_batch_tensor(model, device, batch_tensor):
    """Embed a pre-built tensor batch. Returns (N, 512) float32 L2-normed."""
    batch_tensor = batch_tensor.to(device)
    if device == "cuda":
        batch_tensor = batch_tensor.half()

    with torch.no_grad():
        embs = model.encode_image(batch_tensor)

    embs = embs.float().cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-8, None)
    return embs


# ── Background file copier ──────────────────────────────────────────────
def copy_file(src, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src, dest_dir)


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    # 1. Gather sorted image list
    all_files = sorted(
        f
        for f in glob.glob(os.path.join(IMAGES_DIR, "*"))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    total = len(all_files)
    print(f"Found {total} images in {IMAGES_DIR}/")
    if total == 0:
        return

    # 2. Resume from checkpoint
    start = load_checkpoint()
    if start >= total:
        print("All images already processed.")
        return
    if start > 0:
        print(f"Resuming from index {start} ({start}/{total} done)")

    # 3. Connect to Milvus
    print("Connecting to Milvus...")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    cat_col = Collection("categories")
    img_col = Collection("images")

    # 4. Load CLIP
    print("Loading CLIP ViT-B/32...")
    model, preprocess, device = load_clip()
    print(f"  Device: {device}")

    # 5. Build local cache of existing categories
    cache = CategoryCache()
    cache.load_from_milvus(cat_col)

    # 6. Prepare output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 7. Build DataLoader for parallel image loading
    remaining = all_files[start:]
    dataset = ImagePathDataset(remaining, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=EMBED_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        prefetch_factor=2,
    )

    processed = start
    n_categories = len(cache)
    pbar = tqdm(total=total, initial=start, desc="Categorizing", unit="img")

    # 8. Process batches with background file copying
    with ThreadPoolExecutor(max_workers=COPY_THREADS) as copier:
        for batch_tensor, batch_indices, batch_valids in loader:
            # Filter to valid images only
            valid_mask = [i for i, v in enumerate(batch_valids) if v]
            if not valid_mask:
                processed += len(batch_indices)
                pbar.update(len(batch_indices))
                continue

            valid_tensor = batch_tensor[valid_mask]
            valid_orig_indices = [batch_indices[i] for i in valid_mask]

            embeddings = embed_batch_tensor(model, device, valid_tensor)

            new_cat_ids, new_cat_embs = [], []
            img_filepaths, img_cat_ids, img_embs = [], [], []

            for j, orig_idx in enumerate(valid_orig_indices):
                emb = embeddings[j]
                path = remaining[orig_idx]
                rel_path = os.path.relpath(path).replace("\\", "/")

                matched_id, _ = cache.search(emb)

                if matched_id is None:
                    cat_id = str(uuid.uuid4())
                    cache.add(cat_id, emb)
                    new_cat_ids.append(cat_id)
                    new_cat_embs.append(emb.tolist())
                    n_categories += 1
                else:
                    cat_id = matched_id

                img_filepaths.append(rel_path)
                img_cat_ids.append(cat_id)
                img_embs.append(emb.tolist())

                dest_dir = os.path.join(OUTPUT_DIR, cat_id)
                copier.submit(copy_file, path, dest_dir)

            if new_cat_ids:
                cat_col.insert([new_cat_ids, new_cat_embs])
                cat_col.flush()
            if img_filepaths:
                img_col.insert([img_filepaths, img_cat_ids, img_embs])
                img_col.flush()

            processed += len(batch_indices)
            pbar.update(len(batch_indices))
            save_checkpoint(processed, total, n_categories)

    pbar.close()

    print(f"\nDone!")
    print(f"  Images processed: {processed}")
    print(f"  Categories: {n_categories}")
    print(f"  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

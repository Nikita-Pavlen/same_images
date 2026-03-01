import os
import csv
import random
import io
import requests
from PIL import Image

OUTPUT_DIR = "py-images"
COUNT = 1000
CSV_URL = (
    "https://storage.googleapis.com/openimages/2018_04/"
    "train/train-images-boxable-with-rotation.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading image list CSV ...")
resp = requests.get(CSV_URL)
resp.raise_for_status()
reader = csv.DictReader(io.StringIO(resp.text))
rows = list(reader)
random.shuffle(rows)

print(f"Got {len(rows)} entries, will download {COUNT} images.")

downloaded = 0
for row in rows:
    if downloaded >= COUNT:
        break

    url = row.get("OriginalURL") or row.get("Thumbnail300KURL")
    image_id = row.get("ImageID", f"{downloaded:06d}")
    if not url:
        continue

    try:
        img_resp = requests.get(url, timeout=10)
        img_resp.raise_for_status()
        img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
        name = f"{image_id}.jpg"
        img.save(os.path.join(OUTPUT_DIR, name), "JPEG", quality=95)
        downloaded += 1
        print(f"Saved {downloaded}/{COUNT}: {name}")
    except Exception as e:
        print(f"Skipped {image_id}: {e}")

print(f"Done — {downloaded} images saved to {OUTPUT_DIR}/")

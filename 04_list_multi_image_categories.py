import os

OUTPUT_DIR = "py-result-images"
REPORT_FILE = "multi_image_categories.txt"

categories = []
for name in os.listdir(OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.isdir(path):
        continue
    count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    if count > 1:
        categories.append((count, name))

categories.sort(reverse=True)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(f"Categories with more than 1 image: {len(categories)}\n\n")
    for count, name in categories:
        files = sorted(os.listdir(os.path.join(OUTPUT_DIR, name)))
        f.write(f"{count} images - {name}\n")
        for fname in files:
            f.write(f"  {fname}\n")
        f.write("\n")

print(f"Written {len(categories)} categories to {REPORT_FILE}")

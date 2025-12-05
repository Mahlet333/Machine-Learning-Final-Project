import csv
import os
import requests
from serpapi import GoogleSearch
from PIL import Image
from io import BytesIO

SERPAPI_KEY = "YOUR API KEY"

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CSV_FILE = os.path.join(PROJECT_ROOT, "data", "metadata", "Hero_list.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "images", "hero")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_and_crop(image_url, out_path):
    """Downloads image and saves a square mugshot crop."""
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Make a center square crop (mugshot-style)
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left+side, top+side))
        img = img.resize((256, 256))

        img.save(out_path)
        return True

    except Exception:
        return False


not_found = []

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["character_name"]
        anime = row["anime"]

        query = f"{name} {anime} anime official portrait headshot"

        print(f"\n Google searching: {query}")

        params = {
            "q": query,
            "tbm": "isch",
            "api_key": SERPAPI_KEY
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "images_results" not in results:
            print(" No images found")
            not_found.append(name)
            continue

        # Try grabbing the best result
        success = False
        for img in results["images_results"][:5]:   # Try top 5 images
            url = img.get("original")
            if not url:
                continue

            out_path = os.path.join(OUTPUT_DIR, f"{name}.jpg")
            if download_and_crop(url, out_path):
                print(f" Saved mugshot → {out_path}")
                success = True
                break

        if not success:
            print(f" Could not process any images for {name}")
            not_found.append(name)

print("\n==============================")
print(" Missing Characters ")
print("==============================")
for c in not_found:
    print("", c)
print("==============================")
print(f"Total missing: {len(not_found)}")

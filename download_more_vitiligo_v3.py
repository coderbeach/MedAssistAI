import os
import urllib.request
import urllib.parse
import json
import ssl
import time
import random
import hashlib
from PIL import Image

ssl_context = ssl._create_unverified_context()

DATA_DIR = "./data/external_images"
TRAIN_DIR = os.path.join(DATA_DIR, "train", "Vitiligo")
VAL_DIR = os.path.join(DATA_DIR, "val", "Vitiligo")

VITILIGO_TAGS = [
    "vitiligo", 
    "vitiligoskin", 
    "vitiligotreatment", 
    "vitiligoawareness", 
    "vitiligopride", 
    "vitiligobeauty", 
    "vitiligosupport", 
    "leucoderma", 
    "depigmentation", 
    "whitepatches"
]

def get_image_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_existing_hashes():
    existing_hashes = set()
    for directory in [TRAIN_DIR, VAL_DIR]:
        if os.path.exists(directory):
            for file_name in os.listdir(directory):
                if file_name.endswith((".jpg", ".png", ".jpeg")):
                    file_path = os.path.join(directory, file_name)
                    try:
                        existing_hashes.add(get_image_md5(file_path))
                    except Exception as e:
                        print(f"Error hashing {file_path}: {e}")
    return existing_hashes

def get_flickr_image_urls(tag, max_results=50):
    encoded_tag = urllib.parse.quote(tag)
    url = f"https://www.flickr.com/services/feeds/photos_public.gne?tags={encoded_tag}&format=json&nojsoncallback=1"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_context) as response:
            res_str = response.read().decode("utf-8", errors="ignore")
            data = json.loads(res_str)
            items = data.get("items", [])
            urls = []
            for item in items:
                media = item.get("media", {})
                img_url = media.get("m")
                if img_url:
                    img_url_large = img_url.replace("_m.jpg", "_b.jpg")
                    urls.append(img_url_large)
            return urls[:max_results]
    except Exception as e:
        print(f"  [Warning] Failed to fetch Flickr feed for tag '{tag}': {e}")
        return []

def main():
    print("=================================================================")
    print(" Vitiligo Dataset Expander: Scrapes new unique Vitiligo images")
    print("=================================================================")
    
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    # 1. Get hashes of existing images to prevent duplicates
    print("Scanning existing Vitiligo images for MD5 hashes...")
    existing_hashes = get_existing_hashes()
    print(f"Found {len(existing_hashes)} existing unique Vitiligo images.")
    
    # 2. Gather image URLs from Flickr across different tags
    unique_urls = []
    seen_urls = set()
    
    for tag in VITILIGO_TAGS:
        print(f"Fetching Flickr images for tag: '{tag}'...")
        urls = get_flickr_image_urls(tag, max_results=40)
        for u in urls:
            if u not in seen_urls:
                seen_urls.add(u)
                unique_urls.append(u)
        time.sleep(0.3)
        
    print(f"Total unique URLs found across all tags: {len(unique_urls)}")
    random.shuffle(unique_urls)
    
    # 3. Download and verify images
    downloaded_count = 0
    duplicate_count = 0
    temp_path = "./temp_download_val.jpg"
    
    # Let's find out where to start indexing for new files
    # We will search for files matching "real_extra_more_*.jpg" or just use a timestamp
    timestamp = int(time.time())
    
    for url in unique_urls:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, context=ssl_context, timeout=8) as response:
                img_data = response.read()
            
            if len(img_data) < 1000:
                continue
                
            # Write to a temp file to verify validity and compute hash
            with open(temp_path, "wb") as f:
                f.write(img_data)
                
            # Verify image format
            with Image.open(temp_path) as img:
                img.verify()
                
            # Compute hash
            img_hash = get_image_md5(temp_path)
            if img_hash in existing_hashes:
                duplicate_count += 1
                os.remove(temp_path)
                continue
                
            # It's a new unique image!
            existing_hashes.add(img_hash)
            
            # Determine split (80% train, 20% validation)
            if downloaded_count % 5 == 0:
                dest_dir = VAL_DIR
                split_name = "validation"
            else:
                dest_dir = TRAIN_DIR
                split_name = "train"
                
            final_filename = f"real_extra_more_{timestamp}_{downloaded_count}.jpg"
            final_path = os.path.join(dest_dir, final_filename)
            
            # Move temp file to final location
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(temp_path, final_path)
            
            downloaded_count += 1
            print(f"  [{downloaded_count}] Saved unique image to {split_name}: {final_filename}")
            
            time.sleep(0.1) # Polite delay
            
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            # Skip errors (timeout, invalid format, etc.)
            continue
            
    print("\n=================================================================")
    print(f"Download complete!")
    print(f"  Successfully added {downloaded_count} new unique Vitiligo images.")
    print(f"  Skipped {duplicate_count} duplicate images.")
    print("=================================================================")

if __name__ == "__main__":
    main()

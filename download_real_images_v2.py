import os
import urllib.request
import urllib.parse
import json
import ssl
import time
import random
from PIL import Image

# Setup SSL context for downloading from HTTPS URLs
ssl_context = ssl._create_unverified_context()

DATA_DIR = "./data/external_images"
os.makedirs(DATA_DIR, exist_ok=True)

# 13 target classes with multiple Flickr query tags to ensure we get up to 40 unique images per category
CLASS_TAG_MAPPING = {
    "Acne": ["acne", "acnevulgaris"],
    "Eczema": ["eczema", "atopicdermatitis"],
    "Psoriasis": ["psoriasis", "plaquepsoriasis"],
    "Ringworm": ["ringworm", "tineacorporis"],
    "Vitiligo": ["vitiligo", "vitiligoskin"],
    "Chickenpox rash": ["chickenpox", "varicella"],
    "Measles rash": ["measles", "measlesrash"],
    "Fungal infection": ["fungal-infection", "tineapedis"],
    "Dermatitis": ["dermatitis", "contactdermatitis"],
    "Suspicious skin lesion": ["skinlesion", "melanoma"],
    "Stye": ["eyestye", "hordeolum"],
    "Conjunctivitis": ["conjunctivitis", "pinkeye"],
    "Normal Eye": ["normaleye", "humaneye"]
}

def get_flickr_image_urls(tag, max_results=30):
    encoded_tag = urllib.parse.quote(tag)
    url = f"https://www.flickr.com/services/feeds/photos_public.gne?tags={encoded_tag}&format=json&nojsoncallback=1"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_context) as response:
            res_str = response.read().decode("utf-8", errors="ignore")
            # Parse response JSON
            data = json.loads(res_str)
            items = data.get("items", [])
            
            urls = []
            for item in items:
                media = item.get("media", {})
                img_url = media.get("m")
                if img_url:
                    # Upgrade to large resolution (_b.jpg instead of medium _m.jpg)
                    img_url_large = img_url.replace("_m.jpg", "_b.jpg")
                    urls.append(img_url_large)
            return urls[:max_results]
    except Exception as e:
        print(f"  [Warning] Failed to fetch Flickr feed for tag '{tag}': {e}")
        return []

def download_and_verify_image(url, save_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        # Download image bytes
        with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
            img_data = response.read()
            
        # Verify it's not empty
        if len(img_data) < 1000:
            return False
            
        # Write temporarily to verify
        with open(save_path, 'wb') as f:
            f.write(img_data)
            
        # Verify it can be opened as a valid image by PIL
        with Image.open(save_path) as img:
            img.verify()
            
        return True
    except Exception as e:
        # Remove partial file if download failed or verification failed
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass
        return False

def main():
    print("=================================================================")
    print(" AuraMed AI: Scrapes and Compiles Real Clinical/Visual Dataset")
    print("=================================================================")
    
    total_downloaded = 0
    
    for cls, tags in CLASS_TAG_MAPPING.items():
        cls_name = cls.replace(" ", "_")
        print(f"\nProcessing category: '{cls}'")
        
        # Gather unique URLs across all tags for this class
        unique_urls = []
        seen_urls = set()
        
        for tag in tags:
            print(f"  Fetching Flickr feed for tag: '{tag}'...")
            urls = get_flickr_image_urls(tag, max_results=25)
            for u in urls:
                if u not in seen_urls:
                    seen_urls.add(u)
                    unique_urls.append(u)
            time.sleep(0.5) # Avoid aggressive requests
            
        print(f"  Found {len(unique_urls)} unique image URLs.")
        random.shuffle(unique_urls)
        
        # Setup target splits
        train_cls_dir = os.path.join(DATA_DIR, "train", cls_name)
        val_cls_dir = os.path.join(DATA_DIR, "val", cls_name)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)
        
        # Download up to 40 valid images per class
        # 80% split is 32 train, 20% is 8 val
        downloaded_count = 0
        target_total = min(40, len(unique_urls))
        
        for idx, url in enumerate(unique_urls):
            if downloaded_count >= target_total:
                break
                
            # Determine split (every 5th image is validation)
            if downloaded_count % 5 == 0:
                dest_dir = val_cls_dir
                split_name = "val"
            else:
                dest_dir = train_cls_dir
                split_name = "train"
                
            save_path = os.path.join(dest_dir, f"real_{downloaded_count}.jpg")
            
            success = download_and_verify_image(url, save_path)
            if success:
                downloaded_count += 1
                total_downloaded += 1
                if downloaded_count % 5 == 0 or downloaded_count == target_total:
                    print(f"    Downloaded {downloaded_count}/{target_total} images...")
                time.sleep(0.2) # Polite delay
                
        print(f"  Successfully saved {downloaded_count} real images for '{cls}'.")
        
    print("\n=================================================================")
    print(f" Download Complete! Scraped {total_downloaded} real images across 13 classes.")
    print("=================================================================")

if __name__ == "__main__":
    main()

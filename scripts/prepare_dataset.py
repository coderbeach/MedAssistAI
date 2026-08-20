import os
import sys
import json
import urllib.request
import urllib.parse
import urllib.error
import ssl
import time
import hashlib
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# Create SSL context to bypass certification warnings
ssl_context = ssl._create_unverified_context()

PROJECT_DIR = r"c:\Users\Nisarga N\OneDrive\Documents\My Projects\Healthcare-AI-Capstone"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.json")

# 13 target classes mapped to Wikimedia Search terms
CLASS_CATEGORIES = {
    "acne": ["Category:Acne_vulgaris", "Category:Acne"],
    "eczema": ["Category:Eczema", "Category:Atopic_dermatitis"],
    "psoriasis": ["Category:Psoriasis"],
    "ringworm": ["Category:Tinea_corporis"],
    "vitiligo": ["Category:Vitiligo"],
    "chickenpox": ["Category:Chickenpox", "Category:Varicella"],
    "measles": ["Category:Measles"],
    "fungal_infection": ["Category:Dermatophytosis", "Category:Pityriasis_versicolor"],
    "dermatitis": ["Category:Dermatitis", "Category:Contact_dermatitis"],
    "suspicious_lesion": ["Category:Melanoma", "Category:Basal_cell_carcinoma", "Category:Squamous_cell_carcinoma"],
    "stye": ["Category:Hordeolum"],
    "conjunctivitis": ["Category:Conjunctivitis"],
    "normal_eye": ["Category:Human_eyes", "Category:Irises_(anatomy)"]
}

def query_mediawiki(params):
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    headers = {
        'User-Agent': 'MedicalImageClassifierResearch/1.0 (contact: nisarga.n@example.com)'
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"MediaWiki request failed: {e}")
        return None

def get_wikimedia_category_files(category_title, limit=30):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_title,
        "cmtype": "file",
        "format": "json",
        "cmlimit": str(limit)
    }
    res = query_mediawiki(params)
    if not res:
        return []
    
    members = res.get("query", {}).get("categorymembers", [])
    return [item["title"] for item in members]

def get_file_metadata(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata|mime",
        "iiurlwidth": "500",
        "format": "json"
    }
    res = query_mediawiki(params)
    if not res:
        return None
    
    pages = res.get("query", {}).get("pages", {})
    for page_id, page_data in pages.items():
        imageinfo_list = page_data.get("imageinfo", [])
        if imageinfo_list:
            info = imageinfo_list[0]
            ext_meta = info.get("extmetadata", {})
            
            # Extract fields
            license_name = ext_meta.get("LicenseShortName", {}).get("value", "Unknown/Public Domain")
            artist = ext_meta.get("Artist", {}).get("value", "Unknown Author")
            desc = ext_meta.get("ImageDescription", {}).get("value", "")
            
            # Clean HTML tags from description/artist
            artist_clean = re.sub(r'<[^>]*>', '', artist)
            desc_clean = re.sub(r'<[^>]*>', '', desc)
            
            return {
                "title": title,
                "url": info.get("url"),
                "thumburl": info.get("thumburl", info.get("url")),
                "mime": info.get("mime"),
                "width": info.get("width"),
                "height": info.get("height"),
                "license": license_name,
                "author": artist_clean,
                "description": desc_clean
            }
    return None

def main():
    print("=========================================================")
    # Phase 1: Create folders and download metadata
    print("Initializing Clinical Image Dataset Downloader...")
    print("=========================================================")
    
    # Load existing metadata if it exists
    metadata = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"Loaded existing metadata file. {len(metadata)} records present.")
        except Exception:
            pass

    for cls_idx, (cls_name, categories) in enumerate(CLASS_CATEGORIES.items()):
        cls_dir = os.path.join(RAW_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        print(f"\n[{cls_idx+1}/13] Querying Wikimedia Commons for class: '{cls_name}'")
        
        unique_titles = set()
        for cat in categories:
            print(f"  Fetching files from category: '{cat}'...")
            titles = get_wikimedia_category_files(cat, limit=40)
            unique_titles.update(titles)
            time.sleep(1.0) # Respect rate limits
            
        print(f"  Found {len(unique_titles)} candidate files.")
        
        # Download images and save metadata
        downloaded = 0
        target = 25 # Target 25 high-quality clinical images per class
        
        for title in sorted(list(unique_titles)):
            if downloaded >= target:
                break
                
            # Skip non-image file formats
            if title.lower().endswith((".pdf", ".djvu", ".ogv", ".mp4", ".webm", ".wav", ".mid", ".ogg")):
                continue
                
            # Skip if already downloaded and metadata exists
            title_hash = hashlib.md5(title.encode('utf-8')).hexdigest()
            meta_key = f"{cls_name}_{title_hash}"
            
            if meta_key in metadata:
                filename = metadata[meta_key]["filename"]
                if os.path.exists(os.path.join(cls_dir, filename)):
                    # print(f"    Skipping (already downloaded): {filename}")
                    downloaded += 1
                    continue
            
            print(f"  Fetching info for: {title}")
            info = get_file_metadata(title)
            if not info or not info.get("url"):
                continue
                
            # Filter by mime type to ensure it is an image
            if "image" not in info["mime"].lower():
                continue
                
            # Determine extension
            ext = ".jpg"
            if "png" in info["mime"].lower():
                ext = ".png"
            elif "gif" in info["mime"].lower():
                continue # Skip gifs
                
            filename = f"{cls_name}_{title_hash}{ext}"
            save_path = os.path.join(cls_dir, filename)
            
            # Download and verify
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            clean_url = info["thumburl"]
            req = urllib.request.Request(clean_url, headers=headers)
            
            img_bytes = None
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    print(f"    Downloading (attempt {attempt+1}): {clean_url}")
                    with urllib.request.urlopen(req, context=ssl_context, timeout=15) as response:
                        img_bytes = response.read()
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        print("      [429 Too Many Requests] Sleeping 15 seconds before retry...")
                        time.sleep(15.0)
                    else:
                        print(f"      HTTP Error: {e.code}")
                        break
                except Exception as e:
                    print(f"      Network Error: {e}")
                    break
                    
            if not img_bytes or len(img_bytes) < 1024:
                print("      [Warning] Failed to fetch image bytes or file too small.")
                continue
                
            try:
                with open(save_path, "wb") as out_f:
                    out_f.write(img_bytes)
                    
                # Open with PIL to verify integrity
                with Image.open(save_path) as img:
                    img.verify()
                
                # Compute md5 of the actual downloaded file
                file_md5 = hashlib.md5(img_bytes).hexdigest()
                
                # Save metadata
                metadata[meta_key] = {
                    "filename": filename,
                    "class": cls_name,
                    "title": info["title"],
                    "url": info["url"],
                    "license": info["license"],
                    "author": info["author"],
                    "description": info["description"],
                    "original_resolution": f"{info['width']}x{info['height']}",
                    "file_md5": file_md5
                }
                
                # Save metadata JSON incrementally
                with open(METADATA_PATH, "w", encoding="utf-8") as out_f:
                    json.dump(metadata, out_f, indent=4)
                    
                downloaded += 1
                time.sleep(2.0) # Rate limiting
                
            except Exception as e:
                print(f"      [Error] Failed to save or verify image: {e}")
                if os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                    except Exception:
                        pass

    print("\nDataset Discovery & Curation Complete!")
    print(f"Metadata file updated at: {METADATA_PATH}")

if __name__ == "__main__":
    import re
    main()

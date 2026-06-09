import urllib.request
import urllib.parse
import json
import ssl

ssl_context = ssl._create_unverified_context()

def get_flickr_images(tag, max_results=30):
    # Flickr public feed URL
    encoded_tag = urllib.parse.quote(tag)
    url = f"https://www.flickr.com/services/feeds/photos_public.gne?tags={encoded_tag}&format=json&nojsoncallback=1"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ssl_context) as response:
            res_str = response.read().decode("utf-8", errors="ignore")
            
            # Sometimes Flickr JSON has escaped single quotes, let's clean it up if needed
            # Load as JSON
            data = json.loads(res_str)
            items = data.get("items", [])
            
            # Extract image URLs (usually the 'm' key inside 'media')
            urls = []
            for item in items:
                media = item.get("media", {})
                img_url = media.get("m")
                if img_url:
                    # 'm' is the medium size image (_m.jpg). We can change it to large (_b.jpg) if needed.
                    img_url_large = img_url.replace("_m.jpg", "_b.jpg")
                    urls.append(img_url_large)
                    
            return urls[:max_results]
    except Exception as e:
        print(f"Failed to fetch Flickr images for {tag}: {e}")
        return []

if __name__ == "__main__":
    images = get_flickr_images("acne")
    print(f"Found {len(images)} images on Flickr.")
    for img in images[:5]:
        print(img)

import os

# Unset proxy variables if they exist
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

import requests

def test_download():
    url = "https://openml.org/datasets/0004/44282/dataset_44282.pq"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Attempting to download from {url}...")
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success! Download started.")
            # Read a bit to ensure it's not an XML error
            chunk = next(response.iter_content(chunk_size=1024))
            if b"<?xml" in chunk[:100]:
                print("Error: Received XML response (likely AccessDenied).")
                print(chunk.decode('utf-8'))
            else:
                print("Received valid binary data (Parquet header detected).")
                return True
        else:
            print("Failed to connect.")
    except Exception as e:
        print(f"Exception: {e}")
    return False

if __name__ == "__main__":
    test_download()

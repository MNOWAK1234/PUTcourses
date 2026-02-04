import os
import requests

def download_data():
    target_dir = "data"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    base_url = "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main"
    
    files_to_download = [
        "traffic.csv",
        "weather.csv",
        "electricity.csv",
        "ETTh1.csv",
        "ETTh2.csv",
        "ETTm1.csv",
        "ETTm2.csv",
        "national_illness.csv",
        "exchange_rate.csv"
    ]

    print("Starting download to 'data' folder...")

    for file_name in files_to_download:
        file_path = os.path.join(target_dir, file_name)
        
        if os.path.exists(file_path):
            print(f"[-] {file_name} already exists. Skipping.")
            continue
        
        url_name = "illness.csv" if file_name == "national_illness.csv" else file_name
        url = f"{base_url}/{url_name}"
        
        try:
            print(f"[+] Downloading {file_name}...", end=" ", flush=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print("Done.")
            
        except Exception as e:
            print(f"\n[!] Error downloading {file_name}: {e}")

if __name__ == "__main__":
    download_data()
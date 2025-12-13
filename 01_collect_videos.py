import yt_dlp
import yaml
from pathlib import Path
import time


CONFIG_PATH = "config/sources.yaml"
OUTPUT_DIR = Path("data/raw_video")
COOKIES_FILE = "cookies.txt"   # ถ้าไม่มี จะข้ามอัตโนมัติ
MAX_RETRY = 3


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_output_dir(source_name):
    out = OUTPUT_DIR / source_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_ydl_options(out_dir):
    opts = {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": "mp4",
        "merge_output_format": "mp4",
        "verbose": False,
        "ignoreerrors": True,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "retries": 10,
    }

    # ถ้ามี cookies.txt → ใช้กับ FB / TikTok
    if Path(COOKIES_FILE).exists():
        opts["cookiefile"] = COOKIES_FILE

    return opts


def download_with_retry(url, ydl_opts):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(f"\n[INFO] Downloading ({attempt}/{MAX_RETRY}): {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print("[SUCCESS] Downloaded:", url)
            return True

        except Exception as e:
            print(f"[ERROR] Attempt {attempt} failed for {url}")
            print("Reason:", e)
            time.sleep(2)

    print("[FAILED] Could not download:", url)
    return False


def detect_platform(url):
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    if "tiktok.com" in url:
        return "tiktok"
    if "facebook.com" in url or "fb.watch" in url:
        return "facebook"
    return "unknown"


def main():
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("========== NEW NARRATION DATA COLLECTOR ==========")

    for src in config["sources"]:
        src_name = src["name"]
        urls = src["playlist_or_urls"]

        print(f"\n=== Source: {src_name} ({len(urls)} items) ===")

        out_dir = create_output_dir(src_name)
        ydl_opts = get_ydl_options(out_dir)

        for url in urls:
            platform = detect_platform(url)
            print(f"\n[INFO] Platform detected: {platform.upper()} — {url}")

            # TikTok ต้องเปิด signature extraction
            if platform == "tiktok":
                ydl_opts["extract_flat"] = False
                ydl_opts["skip_download"] = False

            # Facebook ต้องใช้ cookies เท่านั้น
            if platform == "facebook" and "cookiefile" not in ydl_opts:
                print("[WARNING] Facebook requires cookies.txt but not found!")
                print("Skipping:", url)
                continue

            download_with_retry(url, ydl_opts)

    print("\n=========== COMPLETED DATA COLLECTION ===========")
    print(">> All files saved in: data/raw_video/")
    print(">> Next step: python scripts/02_extract_audio.py")


if __name__ == "__main__":
    main()

from pathlib import Path
import ffmpeg

# ใช้ ffmpeg.exe จากโฟลเดอร์โปรเจกต์
FFMPEG_BIN = "ffmpeg/ffmpeg.exe"

VIDEO_DIR = Path("data/raw_video")
AUDIO_DIR = Path("data/raw_audio")

def extract_audio(video_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / (video_path.stem + ".wav")

    (
        ffmpeg
        .input(str(video_path))
        .output(str(audio_path), ac=1, ar=16000)
        .overwrite_output()
        .run(cmd=FFMPEG_BIN)
    )

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    # รองรับทุกไฟล์ในทุกโฟลเดอร์ย่อย
    for vp in VIDEO_DIR.rglob("*.mp4"):
        print(f"[Extracting] {vp}")
        extract_audio(vp, AUDIO_DIR)

if __name__ == "__main__":
    main()

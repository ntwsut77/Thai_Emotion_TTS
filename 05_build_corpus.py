from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRANS_DIR = ROOT / "data" / "transcripts" / "merged"
OUT_DIR = ROOT / "data" / "metadata"


def main():
    rows = []
    for fp in TRANS_DIR.glob("*.json"):
        with fp.open(encoding="utf-8") as f:
            data = json.load(f)
        video_id = fp.stem
        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict) and "segments" in data:
            segments = data.get("segments", [])
        else:
            segments = []

        for seg in segments:
            rows.append({
                "video_id": video_id,
                "start": seg.get("start") if isinstance(seg, dict) else None,
                "end": seg.get("end") if isinstance(seg, dict) else None,
                "text": seg.get("text", "") if isinstance(seg, dict) else str(seg),
            })

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "corpus.parquet")


if __name__ == "__main__":
    main()

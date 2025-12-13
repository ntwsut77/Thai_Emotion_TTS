"""
Fix per-chunk Whisper JSON timestamps without re-transcribing.

Rules:
- For chunk file ..._000.json -> start=0, end=30
- For chunk file ..._001.json -> start=30, end=60
- etc.
- Last chunk uses the actual .wav duration (may be < 30s)

Outputs:
- Rewrite each chunk JSON to have a single segment with fixed start/end/text
- Write merged TXT/JSON per folder in OUT_BASE (same parent as chunk folders)
"""

import json
from pathlib import Path

import torchaudio

# ROOT = Path(__file__).resolve().parent.parent

# # === BASE PATH ===
# CHUNKS_BASE = ROOT / "data" / "chunks"
# TRANSCRIPTS_BASE = ROOT / "data" / "transcripts_chunks"
# TRANSCRIPTS_BASE.mkdir(parents=True, exist_ok=True)

# # === PROCESS ONLY THESE FOUR ===
# TARGET_FOLDERS = [
#     "OuSPfRpfm0Y_16k",
#     "qEM85cu1kMA_16k",
#     "RuP5GqfnoQ4_16k",
#     "YOZUeVej4DU_16k",
# ]
# Paths (adjust if needed)

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_BASE = ROOT / "data" / "transcripts_chunks"
CHUNKS_BASE = ROOT / "data" / "chunks"

# Seconds per chunk from original split (ffmpeg -segment_time)
CHUNK_SECONDS = 30.0


def parse_chunk_idx(path: Path) -> int | None:
    """Extract trailing integer after last underscore."""
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def load_text(json_path: Path, txt_path: Path) -> str:
    """Prefer text from JSON (first entry), fallback to TXT."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            # If list of dicts with text
            if isinstance(data[0], dict) and "text" in data[0]:
                # join all texts in case there were multiple entries
                return " ".join([str(d.get("text", "")).strip() for d in data if isinstance(d, dict)])
            # If raw string list, join
            if isinstance(data[0], str):
                return " ".join([s.strip() for s in data])
    except Exception:
        pass

    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    return ""


def fix_folder(folder: Path):
    print(f"\n=== Fixing folder: {folder.name} ===")

    wav_dir = CHUNKS_BASE / folder.name
    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        print("No JSON files, skipping…")
        return

    merged_segments = []
    merged_texts = []

    for json_file in json_files:
        chunk_idx = parse_chunk_idx(json_file)
        if chunk_idx is None:
            print(f"  [SKIP] Cannot parse chunk index from {json_file.name}")
            continue

        wav_path = wav_dir / f"{json_file.stem}.wav"
        if not wav_path.exists():
            print(f"  [SKIP] Missing wav for {json_file.name}")
            continue

        try:
            info = torchaudio.info(str(wav_path))
            duration = info.num_frames / info.sample_rate
        except Exception as exc:
            print(f"  [SKIP] Cannot read wav info for {wav_path.name}: {exc}")
            continue

        start_ts = chunk_idx * CHUNK_SECONDS
        end_ts = start_ts + duration

        txt_path = folder / f"{json_file.stem}.txt"
        text = load_text(json_file, txt_path)

        fixed_segment = [{"start": start_ts, "end": end_ts, "text": text}]
        json_file.write_text(json.dumps(fixed_segment, ensure_ascii=False, indent=2), encoding="utf-8")

        merged_segments.append(fixed_segment[0])
        merged_texts.append(text)

    if not merged_segments:
        print("No segments fixed; nothing to merge.")
        return

    merged_json = folder.parent / f"{folder.name}_merged.json"
    merged_txt = folder.parent / f"{folder.name}_merged.txt"
    merged_json.write_text(json.dumps(merged_segments, ensure_ascii=False, indent=2), encoding="utf-8")
    merged_txt.write_text("\n".join(merged_texts), encoding="utf-8")
    print(f"[OK] Rebuilt -> {merged_json.name}, {merged_txt.name}")


def main():
    if not TRANSCRIPTS_BASE.exists():
        print(f"[WARN] Transcripts base not found: {TRANSCRIPTS_BASE}")
        return
    if not CHUNKS_BASE.exists():
        print(f"[WARN] Chunks base not found: {CHUNKS_BASE}")
        return

    any_dir = False
    for folder in sorted(TRANSCRIPTS_BASE.glob("*")):
        if folder.is_dir():
            any_dir = True
            fix_folder(folder)
    if not any_dir:
        print(f"[WARN] No folders found under {TRANSCRIPTS_BASE}")


if __name__ == "__main__":
    main()

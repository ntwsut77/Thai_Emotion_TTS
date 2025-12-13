import json
import time
from pathlib import Path

import torch
import torchaudio
from transformers import pipeline

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
OUT_DIR = BASE_DIR / "data" / "transcripts_chunks"
MODEL_NAME = "biodatlab/whisper-th-medium-combined"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model
device = 0 if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Loading model {MODEL_NAME} on device={device}")
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=device,
    torch_dtype=dtype,
    return_timestamps=True,
    generate_kwargs={"language": "<|th|>", "task": "transcribe"},
)

# Process chunks
folders = sorted(CHUNKS_DIR.glob("*"))
for folder in folders:
    if not folder.is_dir():
        continue

    print(f"\n===== Processing folder: {folder.name} =====")

    wav_files = sorted(folder.glob("*.wav"))
    if not wav_files:
        print("No wav files, skipping...")
        continue

    folder_out = OUT_DIR / folder.name
    folder_out.mkdir(exist_ok=True)

    merged_segments = []
    merged_texts = []
    offset_seconds = 0.0

    for wav in wav_files:
        out_json = folder_out / f"{wav.stem}.json"
        out_txt = folder_out / f"{wav.stem}.txt"

        # Use existing outputs if present; otherwise transcribe.
        if out_json.exists() and out_txt.exists():
            print(f"    SKIP {wav.name} (use existing outputs)")
            text = out_txt.read_text(encoding="utf-8")
            try:
                chunks = json.loads(out_json.read_text(encoding="utf-8"))
            except Exception:
                chunks = []
        else:
            print(f"    Transcribing {wav.name} ...")
            start = time.time()
            result = asr(str(wav))
            text = result.get("text", "") or ""
            chunks = result.get("chunks", []) or []
            out_txt.write_text(text, encoding="utf-8")
            json.dump(chunks, out_json.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"    DONE {wav.name} in {time.time() - start:.1f}s")

        # Merge with global offsets for this folder
        for ch in chunks:
            ts = ch.get("timestamp")
            if not isinstance(ts, (list, tuple)) or len(ts) != 2:
                continue
            start_ts, end_ts = ts
            if start_ts is None or end_ts is None:
                continue
            merged_segments.append(
                {
                    "start": float(start_ts) + offset_seconds,
                    "end": float(end_ts) + offset_seconds,
                    "text": ch.get("text", "").strip(),
                }
            )

        if text:
            merged_texts.append(text)

        info = torchaudio.info(str(wav))
        duration = info.num_frames / info.sample_rate
        offset_seconds += duration

    # Write merged outputs for this folder
    if merged_segments:
        merged_txt = OUT_DIR / f"{folder.name}_merged.txt"
        merged_json = OUT_DIR / f"{folder.name}_merged.json"
        merged_txt.write_text("\n".join(merged_texts), encoding="utf-8")
        json.dump(merged_segments, merged_json.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"===== MERGED -> {merged_txt.name}, {merged_json.name}")
    else:
        print("No segments to merge.")

import subprocess
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # Optional progress bar
    tqdm = None

RAW_AUDIO = Path("data/raw_audio")
CHUNK_DIR = Path("data/chunks")
OUT_DIR = Path("data/separated_audio")

CHUNK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def split_wav(src_wav: Path) -> Path:
    base = src_wav.stem
    dst = CHUNK_DIR / base
    dst.mkdir(exist_ok=True)

    if any(dst.glob("*.wav")):
        print(f"[SKIP] Chunks already exist for {src_wav.name}")
        return dst

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_wav),
        "-f",
        "segment",
        "-segment_time",
        "30",
        "-c",
        "copy",
        str(dst / f"{base}_%03d.wav"),
    ]
    print("[Split] Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[ERROR] Split failed for {src_wav.name}: {exc}")

    return dst


def chunk_vocals_path(out_subdir: Path, chunk_stem: str, model: str) -> Path:
    model_dir = out_subdir / model
    primary = model_dir / chunk_stem / "vocals.wav"
    if primary.exists():
        return primary
    fallback = next(model_dir.glob(f"*/{chunk_stem}/vocals.wav"), None)
    if fallback:
        return fallback
    wildcard = next(model_dir.glob("*/vocals.wav"), None)
    return wildcard if wildcard else primary


def run_demucs_chunked(chunk_folder: Path, model: str = "mdx_q"):
    chunks = sorted(chunk_folder.glob("*.wav"))
    base = chunk_folder.name
    out_subdir = OUT_DIR / base
    out_subdir.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(chunks, desc=f"Chunks {base}", unit="chunk") if tqdm else chunks
    for c in iterator:
        vocals_path = chunk_vocals_path(out_subdir, c.stem, model)
        if vocals_path.exists():
            print(f"[SKIP] Vocals already exist for {c.name}")
            continue

        print(f"[Demucs GPU] Model={model} Chunk={c.name}")
        start = time.perf_counter()
        cmd = [
            "demucs",
            "--two-stems=vocals",
            "-n",
            model,
            "--device",
            "cuda",
            "--segment",
            "7",
            str(c),
            "-o",
            str(out_subdir),
        ]
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.perf_counter() - start
            print(f"[DONE] {c.name} in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"[ERROR] Demucs failed for {c.name} after {elapsed:.1f}s: {exc}")
            continue


def merge_vocals(chunk_folder: Path, model: str = "mdx_q"):
    base = chunk_folder.name
    chunks = sorted(chunk_folder.glob("*.wav"))
    out_final = OUT_DIR / f"{base}_vocals_merged.wav"
    merge_list = Path("merge_list.txt")

    vocals_in_order = []
    for c in chunks:
        idx_str = c.stem.split("_")[-1]
        try:
            chunk_idx = int(idx_str)
        except ValueError:
            chunk_idx = -1

        vf = chunk_vocals_path(OUT_DIR / base, c.stem, model)
        if vf.exists():
            vocals_in_order.append((chunk_idx, vf))
        else:
            print(f"[MISSING] vocals.wav for {c.name}")

    if len(vocals_in_order) != len(chunks):
        print(f"[SKIP MERGE] Missing {len(chunks) - len(vocals_in_order)} chunk(s) for {base}")
        return

    vocals_in_order.sort(key=lambda x: x[0])

    with merge_list.open("w") as f:
        for _, vf in vocals_in_order:
            f.write(f"file '{vf.as_posix()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(merge_list),
        "-c",
        "copy",
        str(out_final),
    ]
    print("[Merge] Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[ERROR] Merge failed for {base}: {exc}")
    finally:
        if merge_list.exists():
            merge_list.unlink()


def main():
    wavs = list(RAW_AUDIO.glob("*_16k.wav"))
    print(f"=== Found {len(wavs)} normalized WAV files ===")

    for wav in wavs:
        base = wav.stem
        out_final = OUT_DIR / f"{base}_vocals_merged.wav"
        if out_final.exists():
            print(f"[SKIP] Already merged: {out_final.name}")
            continue

        print(f"\n===== PROCESSING {wav.name} =====")
        chunk_folder = split_wav(wav)
        run_demucs_chunked(chunk_folder)
        merge_vocals(chunk_folder)
        print(f"[DONE] Finished {wav.name}")


if __name__ == "__main__":
    main()

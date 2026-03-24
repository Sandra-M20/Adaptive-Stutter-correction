import os
os.environ["PATH"] = r"C:\ffmpeg\bin" + os.pathsep + os.environ["PATH"]
import json
from pathlib import Path
import numpy as np

print("Importing Whisper...")
import whisper
print("Importing pipeline...")
from main_pipeline import AdaptiveStutterPipeline

AUDIO_DIR = Path("archive/audio")
OUTPUT_DIR = Path("results/fluency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "M_0107_07y7m_1.wav",
    "M_0030_16y4m_1.wav",
    "M_1106_25y0m_1.wav",
]


def count_repeated_words(text: str) -> int:
    words = text.lower().split()
    repeats = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            repeats += 1
    return repeats


def word_count(text: str) -> int:
    return len(text.split())


def transcribe(model, path: Path) -> str:
    result = model.transcribe(str(path), language="en")
    return result.get("text", "").strip()


def main() -> None:
    print("\nLoading Whisper small...")
    model = whisper.load_model("small")

    print("Loading pipeline...")
    pipeline = AdaptiveStutterPipeline(
        transcribe=False, use_enhancer=False, use_repetition=True
    )
    pipeline.preprocessor.noise_reduce = False
    pipeline.preprocessor.normalization = None

    results = []

    print("\n" + "=" * 70)
    print(f"{'File':<25} {'Rep(Orig)':>9} {'Rep(Corr)':>9} {'Reduction':>10} {'Words(Orig)':>11} {'Words(Corr)':>11}")
    print("-" * 70)

    for filename in FILES:
        audio_path = AUDIO_DIR / filename
        if not audio_path.exists():
            print(f"[SKIP] {filename} not found")
            continue

        corrected_path = OUTPUT_DIR / f"corrected_{filename}"

        try:
            pipeline.run(str(audio_path), output_path=str(corrected_path), optimize=False)

            tx_orig = transcribe(model, audio_path)
            tx_corr = transcribe(model, corrected_path)

            rep_orig = count_repeated_words(tx_orig)
            rep_corr = count_repeated_words(tx_corr)
            wc_orig = word_count(tx_orig)
            wc_corr = word_count(tx_corr)

            reduction = ((rep_orig - rep_corr) / rep_orig * 100) if rep_orig > 0 else 0.0

            results.append({
                "file": filename,
                "transcript_orig": tx_orig,
                "transcript_corr": tx_corr,
                "repeated_orig": rep_orig,
                "repeated_corr": rep_corr,
                "word_count_orig": wc_orig,
                "word_count_corr": wc_corr,
                "repetition_reduction_pct": reduction,
            })

            print(f"{filename:<25} {rep_orig:>9} {rep_corr:>9} {reduction:>9.1f}% {wc_orig:>11} {wc_corr:>11}")

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        avg_reduction = float(np.mean([r["repetition_reduction_pct"] for r in results]))
        print(f"\n{'Average':<25} {'':>9} {'':>9} {avg_reduction:>9.1f}%")
        print(f"\nAverage repeated-word reduction: {avg_reduction:.1f}%")

        out = OUTPUT_DIR / "fluency_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nTranscripts saved: {out}")

        r = results[0]
        print(f"\nSample - {r['file']}")
        print(f"  ORIGINAL:  {r['transcript_orig'][:200]}")
        print(f"  CORRECTED: {r['transcript_corr'][:200]}")


if __name__ == "__main__":
    main()

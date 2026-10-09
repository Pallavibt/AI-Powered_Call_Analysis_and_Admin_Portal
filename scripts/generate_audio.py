

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyttsx3
from app.data import get_all_calls

AUDIO_DIR = ROOT / "app" / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

engine = pyttsx3.init()
engine.setProperty("rate", 150)

generated_count = 0

for call in get_all_calls():
    output_file = AUDIO_DIR / f"{call['id']}.wav"

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"Skipping existing file: {output_file.name}", flush=True)
        continue

    print(f"Generating: {output_file.name}", flush=True)

    engine.save_to_file(
        call["transcript"],
        str(output_file)
    )
    engine.runAndWait()

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"Created: {output_file.name}", flush=True)
        generated_count += 1
    else:
        print(f"Failed: {output_file.name}", flush=True)

print(f"\nFinished. New recordings generated: {generated_count}")

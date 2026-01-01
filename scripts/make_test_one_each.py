# scripts/make_test_one_each.py
import os, shutil
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(BASE, "data", "genres_original")
DEST = os.path.join(BASE, "data", "test_one_each")
os.makedirs(DEST, exist_ok=True)

if not os.path.exists(SRC_ROOT):
    print("Source folder not found:", SRC_ROOT)
    raise SystemExit(1)

count = 0
for g in sorted(os.listdir(SRC_ROOT)):
    gp = os.path.join(SRC_ROOT, g)
    if not os.path.isdir(gp):
        continue
    wavs = [f for f in sorted(os.listdir(gp)) if f.lower().endswith(".wav")]
    if not wavs:
        print("No wav files in:", gp)
        continue
    src = os.path.join(gp, wavs[0])
    dest = os.path.join(DEST, f"{g}.00000.wav")
    shutil.copy2(src, dest)
    print("Copied", src, "->", dest)
    count += 1

print("Copied", count, "genre files into", DEST)

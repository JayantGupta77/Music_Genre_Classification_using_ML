# test_extractor.py
import importlib, traceback, joblib, os

print("Working dir:", os.getcwd())
print("Checking model_utils...")

try:
    m = importlib.import_module("model_utils")
except Exception as e:
    print("Failed to import model_utils:", e)
    raise SystemExit(1)

# show public function names (filtered)
fns = [n for n in dir(m) if not n.startswith("_")]
print("model_utils public names (first 80):", fns[:80])

# candidate extractor names we'll try
candidates = ("extract_features","extract","compute_features","make_features","get_features","make_feature_vector","featurize")

test_file = "uploads/temp.wav"
if not os.path.exists(test_file):
    print("Test audio not found at", test_file)
    raise SystemExit(1)

print("Test audio found:", test_file)

for fn in candidates:
    if hasattr(m, fn):
        print("Trying function:", fn)
        func = getattr(m, fn)
        try:
            # first try calling with path
            out = func(test_file)
            print("-> call with path succeeded. Returned type:", type(out))
            try:
                # try to show length/shape safely
                if hasattr(out, "shape"):
                    print("   shape:", out.shape)
                elif hasattr(out, "__len__"):
                    print("   len:", len(out))
                else:
                    print("   no len/shape; repr:", repr(out)[:200])
            except Exception as e:
                print("   couldn't get shape/len:", e)
            # print small sample (safe)
            try:
                print("   sample repr:", repr(out)[:300])
            except:
                pass
            break
        except TypeError as e:
            # maybe function expects (y, sr)
            import librosa
            print("   call with path failed (TypeError). Trying (y, sr) signature...")
            try:
                y, sr = librosa.load(test_file, duration=3, sr=None)
                out2 = func(y, sr)
                print("-> call with (y,sr) succeeded. Returned type:", type(out2))
                try:
                    if hasattr(out2, "shape"):
                        print("   shape:", out2.shape)
                    elif hasattr(out2, "__len__"):
                        print("   len:", len(out2))
                except:
                    pass
                try:
                    print("   sample repr:", repr(out2)[:300])
                except:
                    pass
                break
            except Exception as e2:
                print("   (y,sr) call failed:", type(e2), e2)
                traceback.print_exc()
        except Exception as e:
            print("   call failed:", type(e), e)
            traceback.print_exc()

else:
    print("No known extractor functions found in model_utils (or none worked).")

import requests

url = "http://127.0.0.1:5000/predict"
file_path = r"C:\path\to\some_audio.wav"   # <- change to a real .wav/.mp3 you have

with open(file_path, "rb") as f:
    files = {"file": (file_path.split("\\")[-1], f, "audio/wav")}
    try:
        r = requests.post(url, files=files, timeout=20)
        print("Status:", r.status_code)
        print("Response:", r.text)
    except Exception as e:
        print("Request failed:", e)

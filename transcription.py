import os
import json
import torch
import time
from transformers import pipeline

# === CONFIGURACIÓN PRINCIPAL ===
AUDIO_DIR = "audios/"             # carpeta con tus audios
OUTPUT_DIR = "transcripciones/"   # carpeta destino para los JSON
MODEL_NAME = "openai/whisper-large-v3"
LANG = "es"
CHUNK = 25
STRIDE = (5, 5)

# === CONFIGURACIÓN DE HARDWARE ===
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

print(f"Usando dispositivo: {device}")

# === CREAR PIPELINE UNA VEZ ===
asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    torch_dtype=dtype,
    device=device
)

# === FUNCIONES AUXILIARES ===
def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return "".join(c if c.isalnum() or c in " -_." else "_" for c in stem)

def to_segments(chunks):
    segs = []
    for ch in chunks or []:
        ts = ch.get("timestamp")
        if ts and ts[0] is not None and ts[1] is not None:
            segs.append({
                "start": float(ts[0]),
                "end": float(ts[1]),
                "text": ch.get("text", "").strip()
            })
    return segs

def transcribe_and_save(audio_path):
    """Transcribe un audio y guarda el resultado en JSON."""
    start = time.time()
    stem = safe_stem(audio_path)
    out_json = os.path.join(OUTPUT_DIR, f"{stem}.json")

    # Evita reprocesar si ya existe
    if os.path.exists(out_json):
        print(f"Ya existe: {out_json}")
        return

    print(f"\nProcesando: {audio_path}")

    try:
        result = asr(
            audio_path,
            chunk_length_s=CHUNK,
            stride_length_s=STRIDE,
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe",
                "language": LANG,
                "return_timestamps": True,
                "temperature": 0.0
            }
        )

        payload = {
            "meta": {
                "model": MODEL_NAME,
                "language": LANG,
                "chunk_length_s": CHUNK,
                "stride_length_s": list(STRIDE)
            },
            "text": result.get("text", "").strip(),
            "segments": to_segments(result.get("chunks"))
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Tiempo procesamiento: {time.time() - start:.2f} segundos")
        print(f"Guardado JSON: {out_json}")

    except Exception as e:
        print(f"Error procesando {audio_path}: {e}")

# === CREAR CARPETA DESTINO ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === RECORRER TODOS LOS AUDIOS ===
valid_ext = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(valid_ext) and f.lower().startswith('candidat')]

if not audio_files:
    print(f"No se encontraron audios en {AUDIO_DIR}")
else:
    print(f"Encontrados {len(audio_files)} audios en {AUDIO_DIR}")

for fname in audio_files:
    fpath = os.path.join(AUDIO_DIR, fname)
    transcribe_and_save(fpath)

print("\nProceso completado.")

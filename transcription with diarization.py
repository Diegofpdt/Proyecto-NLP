import os
import json
import time
import torch
from transformers import pipeline
from pyannote.audio import Pipeline as PyannotePipeline

# === RUTAS ===
AUDIO_DIR = "audios/"
OUTPUT_DIR = "transcripciones/"
MODEL_NAME = "openai/whisper-large-v3"
LANG = "es"
CHUNK = 25
STRIDE = (5, 5)

# === HARDWARE ===
# - CUDA: usa índice 0
# - MPS (Apple Silicon): usa string "mps"
# - CPU: "cpu"
if torch.cuda.is_available():
    device_for_hf = 0  # <- transformers.pipeline espera int para CUDA
    dtype = torch.float16  # Whisper acelera en fp16 en GPU
elif torch.backends.mps.is_available():
    device_for_hf = "mps"
    dtype = torch.float32  # MPS suele ir más estable en fp32
else:
    device_for_hf = "cpu"
    dtype = torch.float32

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print(f"Usando dispositivo: {device_for_hf}")

# === PIPELINE ASR (Whisper) ===
# Nota: 'torch_dtype' está deprecado -> usar 'dtype'
asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    device=device_for_hf,
    dtype=dtype,
)

# Config por defecto que no cambia entre archivos
ASR_CALL_KW = dict(
    chunk_length_s=CHUNK,
    stride_length_s=STRIDE,
    return_timestamps=True,   # devuelve 'chunks' con timestamps
    # Para Whisper en HF, language puede ir aquí directo:
    # (no necesitas ponerlo dentro de generate_kwargs)
    language=LANG,
    # Si quieres forzar "transcribe" (no translate)
    task="transcribe",
    temperature=0.0,
)

# === DIARIZACIÓN (pyannote) ===
HF_TOKEN = ""  # exporta tu token antes de correr
DIAR_MODEL = "pyannote/speaker-diarization-3.1"

# pyannote no usa MPS; CPU o CUDA
diar_device = "cuda" if torch.cuda.is_available() else "cpu"

# Crea el pipeline de pyannote con token (si es requerido por el modelo)
# y muévelo al device correspondiente
diarizer = PyannotePipeline.from_pretrained(DIAR_MODEL, use_auth_token=HF_TOKEN)
diarizer.to(torch.device(diar_device))

# === HELPERS ===
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

def _collect_speaker_segments(diarization):
    spk_segs = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        spk_segs.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker
        })
    return spk_segs

def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def _assign_speakers(whisper_segs, speaker_segs):
    if not speaker_segs:
        for seg in whisper_segs:
            seg["speaker"] = "SPEAKER_00"
        return whisper_segs

    for seg in whisper_segs:
        s0, s1 = seg["start"], seg["end"]
        best_spk, best_ov = None, 0.0
        for sp in speaker_segs:
            ov = _overlap(s0, s1, sp["start"], sp["end"])
            if ov > best_ov:
                best_ov, best_spk = ov, sp["speaker"]
        seg["speaker"] = best_spk or "SPEAKER_00"
    return whisper_segs

def transcribe_and_save(audio_path):
    start = time.time()
    stem = safe_stem(audio_path)
    out_json = os.path.join(OUTPUT_DIR, f"{stem}.json")

    if os.path.exists(out_json):
        print(f"Ya existe: {out_json}")
        return

    print(f"\nProcesando: {audio_path}")

    try:
        # --- ASR ---
        result = asr(audio_path, **ASR_CALL_KW)

        # --- Diarización ---
        try:
            dia = diarizer(audio_path)
            speaker_segments = _collect_speaker_segments(dia)
        except Exception as e:
            print(f"Advertencia diarización: {e}")
            speaker_segments = []

        segments = to_segments(result.get("chunks"))
        segments = _assign_speakers(segments, speaker_segments)

        payload = {
            "meta": {
                "model": MODEL_NAME,
                "language": LANG,
                "chunk_length_s": CHUNK,
                "stride_length_s": list(STRIDE),
                "diarization_model": DIAR_MODEL,
                "device": str(device_for_hf),
                "dtype": str(dtype).split(".")[-1],
            },
            "text": (result.get("text") or "").strip(),
            "segments": segments,
            "speakers": sorted({s.get("speaker", "SPEAKER_00") for s in segments})
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Tiempo procesamiento: {time.time() - start:.2f} s")
        print(f"Guardado JSON: {out_json}")

    except Exception as e:
        print(f"Error procesando {audio_path}: {e}")

# === LOOP DE ARCHIVOS ===
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

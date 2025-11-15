import os
import json
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from time import time

# ========= CONFIG =========
AUDIO_DIR = "audios/"
INPUT_DIR = "transcripciones/"   # aquí están los .json antiguos 
OUTPUT_DIR = "transcripciones_diarized/"   # aquí se guardarán los _diarized.json
SAMPLE_RATE = 16000               # trabajamos en 16 kHz mono
MIN_SEG_DUR = 0.1                 # seg mínimos para un embedding fiable
PAD = 0.01                        # padding (seg) a cada lado del segmento
SIM_THRESHOLD = 0.5               # umbral similitud coseno para "mismo hablante"
MAX_SPEAKERS = 10                  # máximo número de hablantes permitidos
# ==========================

def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return "".join(c if c.isalnum() or c in " -_." else "_" for c in stem)

def load_audio_16k_mono(path: str):
    wav, sr = torchaudio.load(path)             # [channels, time]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)     # a mono
    if sr != SAMPLE_RATE:
        wav = Resample(sr, SAMPLE_RATE)(wav)    # resample
    return wav.squeeze(0), SAMPLE_RATE          # [time], 16000

def slice_signal(sig: torch.Tensor, sr: int, t0: float, t1: float):
    a = max(0, int(round(sr * t0)))
    b = min(sig.numel(), int(round(sr * t1)))
    if b <= a:
        return None
    return sig[a:b]

@torch.no_grad()
def embed_segment(spk_model, sig_seg: torch.Tensor, device: str):
    if sig_seg is None or sig_seg.numel() < int(MIN_SEG_DUR * SAMPLE_RATE / 2):
        return None
    x = sig_seg.float()
    x = x / (x.abs().max() + 1e-8)   # normalización simple
    x = x.unsqueeze(0)               # [1, time]
    emb = spk_model.encode_batch(x.to(device))
    vec = emb.squeeze(0).squeeze(0).detach().cpu().numpy()
    vec = vec / (np.linalg.norm(vec) + 1e-8)    # L2 norm
    return vec

def assign_speakers_by_embeddings(segments, sig16k, sr, spk_model, spk_device, max_speakers=None):
    """
    Asigna 'speaker' a cada segmento usando embeddings (ECAPA) y clustering por similitud.
    Si max_speakers no es None, no se crean más hablantes que ese máximo.
    Devuelve (segments_con_speaker, lista_speakers_ordenados_por_duración).
    """
    speakers = []  # {"id": "SPEAKER_00", "centroid": np.array, "embs": [np.array], "dur": float}
    next_id = 0

    for seg in segments:
        s0 = max(0.0, seg["start"] - PAD)
        s1 = seg["end"] + PAD
        if (s1 - s0) < MIN_SEG_DUR:
            mid = 0.5 * (s0 + s1)
            s0 = max(0.0, mid - 0.5 * MIN_SEG_DUR)
            s1 = mid + 0.5 * MIN_SEG_DUR

        wav_seg = slice_signal(sig16k, sr, s0, s1)
        emb = embed_segment(spk_model, wav_seg, spk_device)
        if emb is None:
            if not speakers:
                spk_id = f"SPEAKER_{next_id:02d}"
                speakers.append({"id": spk_id, "centroid": None, "embs": [], "dur": 0.0})
                next_id += 1
                seg["speaker"] = spk_id
            else:
                seg["speaker"] = speakers[0]["id"]
            continue

        if not speakers:
            spk_id = f"SPEAKER_{next_id:02d}"
            speakers.append({
                "id": spk_id,
                "centroid": emb.copy(),
                "embs": [emb],
                "dur": seg["end"] - seg["start"]
            })
            seg["speaker"] = spk_id
            next_id += 1
            continue

        # comparar contra centroides
        centroids = np.vstack([sp["centroid"] for sp in speakers])
        sims = cosine_similarity([emb], centroids)[0]
        j = int(np.argmax(sims))
        best_sim = sims[j]

        # ¿podemos crear un nuevo hablante?
        can_create_new = (max_speakers is None) or (len(speakers) < max_speakers)

        if best_sim >= SIM_THRESHOLD or not can_create_new:
            # Asignar al hablante más similar
            # (si ya llegamos al máximo, forzamos a agrupar aquí aunque la similitud sea baja)
            sp = speakers[j]
            sp["embs"].append(emb)
            sp["dur"] += (seg["end"] - seg["start"])
            new_c = np.mean(np.vstack(sp["embs"]), axis=0)
            sp["centroid"] = new_c / (np.linalg.norm(new_c) + 1e-8)
            seg["speaker"] = sp["id"]
        else:
            # Crear nuevo hablante (solo si no alcanzamos el máximo y la similitud es baja)
            spk_id = f"SPEAKER_{next_id:02d}"
            speakers.append({
                "id": spk_id,
                "centroid": emb.copy(),
                "embs": [emb],
                "dur": seg["end"] - seg["start"]
            })
            seg["speaker"] = spk_id
            next_id += 1

    # Renombrar por duración total (SPK1 el que más habló)
    speakers_sorted = sorted(speakers, key=lambda d: -d["dur"])
    id_map = {sp["id"]: f"SPK{idx+1}" for idx, sp in enumerate(speakers_sorted)}
    for seg in segments:
        seg["speaker"] = id_map.get(seg["speaker"], seg["speaker"])
    spk_names = [id_map[sp["id"]] for sp in speakers_sorted]
    return segments, spk_names

def process_one(audio_path: str):
    t0 = time()
    stem = safe_stem(audio_path)
    json_in = os.path.join(INPUT_DIR, f"{stem}.json")
    json_out = os.path.join(OUTPUT_DIR, f"{stem}_diarized.json")
    
    if not os.path.exists(json_in):
        print(f"No se encontró JSON para {audio_path} → {json_in}")
        return "skip"

    # Cargar JSON
    with open(json_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print(f"JSON sin 'segments': {json_in}")
        return "skip"

    # Cargar audio 16k mono
    sig16k, sr = load_audio_16k_mono(audio_path)

    # Modelo de embeddings (CPU o CUDA; MPS no aplica)
    spk_device = "cuda" if torch.cuda.is_available() else "cpu"
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": spk_device}
    )

    # Asignar speakers con límite máximo
    segments, spk_names = assign_speakers_by_embeddings(
        segments, sig16k, sr, spk_model, spk_device, max_speakers=MAX_SPEAKERS
    )

    # Escribir salida
    payload = {
        "meta": {
            **(data.get("meta") or {}),
            "speaker_method": "ecapa_cosine",
            "sample_rate": sr,
            "min_seg_dur_s": MIN_SEG_DUR,
            "pad_s": PAD,
            "sim_threshold": SIM_THRESHOLD,
            "device_spk": spk_device,
            "max_speakers": MAX_SPEAKERS,
        },
        "text": data.get("text", ""),
        "segments": segments,
        "speakers": spk_names
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"{os.path.basename(audio_path)} → {os.path.basename(json_out)}  ({time()-t0:.1f}s)")
    return "ok"

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    valid_ext = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

    files = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.lower().endswith(valid_ext) and f.lower().endswith("ara_02-10.mp3")]
    if not files:
        print("No se encontraron audios.")
        return

    print(f"Procesando {len(files)} audios…")
    ok = skip = 0
    for ap in files:
        res = process_one(ap)
        if res == "ok":
            ok += 1
        else:
            skip += 1
    print(f"\nResumen: OK={ok}  SKIP={skip}")

if __name__ == "__main__":
    main()

# Naruto Scene Analysis — SAM + (CLIP / DINOv2) + FAISS

Pipeline end-to-end per analizzare scene multi-personaggio tratte da Naruto:
- Segmentazione con SAM → crop mascherati
- Embedding con CLIP (baseline e fine-tuned) oppure DINOv2 (alternativo)
- Vector search con cosine naive o FAISS (advanced vector DB)
- Valutazione con matching univoco (Hungarian, IoU≥0.5)
- UI Gradio per analisi singola scena e query al database  

Il training usa policy single_only: immagini con un solo personaggio noto tra Gaara, Naruto, Sakura, Tsunade.

---

## Progetto universitario
Questo repository contiene lo sviluppo del progetto d’esame per il corso di **Deep Learning** (MSc Computer Engineering – Cybersecurity and Cloud) presso il **Politecnico di Bari**, tenuto dal **Prof. Vito Walter Anelli, Ph.D.**  

**Titolo del progetto:** *Object-Level Visual Search and Scene Analysis with Foundational Models*  
**Anno accademico:** 2024/2025  

---

## Ambiente e Installazione
Testato con Python ≥ 3.10, PyTorch ≥ 2.1. Funziona su CPU, consigliata GPU.

```bash
# Dipendenze core
pip install torch torchvision
pip install open_clip_torch timm opencv-python pillow numpy pandas tqdm scipy gradio

# SAM
pip install git+https://github.com/facebookresearch/segment-anything

# FAISS (CPU)
pip install faiss-cpu

# DINOv2 (opzionale: via torch.hub alla prima esecuzione o via pip)
pip install dinov2
```

### Pesi modello
Metti il <a href="https://github.com/facebookresearch/segment-anything/blob/main/README.md">checkpoint SAM ViT-B</a> (presente nella sezione Model Checkpoints) in `weights/`, es. `weights/sam_vit_b_01ec64.pth` (Per modelli SAM diversi, usa `--sam-ckpt`).


---

## Struttura dati
```
data/
  train/
    _classes.csv
    <immagini>.jpg|png
  valid/
    _classes.csv
    <immagini>.jpg|png
  test/
    _classes.csv
    <immagini>.jpg|png
  scenes_real/
    images/
      scene_0000.jpg
      ...
    gt.json              # ground truth per scena
weights/
  sam_vit_b_01ec64.pth
manifests/              # generati
index_ft/               # indice CLIP FT (generato)
index_dino/             # indice DINOv2 (generato)
runs/
  clip_ft_v1/           # checkpoint fine-tuning
```
I file `_classes.csv` (stile Roboflow) contengono filename, Unlabeled, Gaara, Naruto, Sakura, Tsunade. Il dataset usato si trova su <a href="https://universe.roboflow.com/faisal-fida-u4qau/anime-naruto">Roboflow</a>.

---

## Preprocess -> Manifests
Genera manifest puliti con policy `single_only` (drop Unlabeled, tieni solo righe con esattamente un personaggio positivo):
```bash
# CLIP Baseline
python -m src.data_prep.build_manifests
```

---

## Buil Index (CLIP/DINOv2)
### CLIP (baseline o fine-tuned)
```bash
python -m src.indexing.build_index \
  --manifest manifests/train.csv \
  --out index_ft \                       # per clip baseline usa: index
  --finetuned runs/clip_ft_v1/best.pt   # opzionale; rimuovi per usare il pre-trained
  # --aug-index                          # opzionale: varianti gamma
  # --faiss                              # opzionale: costruisci anche indice FAISS
```
### DINOv2 (embedding alternativo)
```bash
python -m src.indexing.build_index \
  --manifest manifests/train.csv \
  --out index_dino \
  --backend dino \
  # --aug-index                          # opzionale: varianti gamma
  # --faiss                              # opzionale: costruisci anche indice FAISS
```

---

## Fine-tuning di CLIP
Fine-tuning con `--freeze-text` (rimuovi per sbloccare anche il testo):
```bash
python -m src.train.train_clip_finetune --freeze-text
```
<img width="330" height="263" alt="confusion_matrix" src="https://github.com/user-attachments/assets/9eeae324-b832-4d57-a2de-4976cb93e951" />
<img width="330" height="263" alt="curves_acc" src="https://github.com/user-attachments/assets/3fcf7cf6-869c-4d25-8890-b9a05c886adf" />
<img width="330" height="263" alt="curves_loss" src="https://github.com/user-attachments/assets/27f42826-047b-4844-aae8-31eeb00b91f9" />


## Scene Analysis
Esegue l’intera pipeline: SAM → mask/crop → embedding → cosine top-1 vs indice.
Scrive due JSON:
- `eval JSON`: predizioni per la valutazione (no soglia, no NMS)
- `overlay JSON`: predizioni per la UI (con soglia e NMS/top-K)
```bash
# Analisi di una singola immagine (con clip ft)
python -m src.scene.analyze_scene \
  --scene data/scenes_real/images/scene_0000.jpg \
  --index index_ft \                                  # index, per clip no_ft
  --out outputs_scenes_ft \
  --finetuned runs/clip_ft_v1/best.pt \                # per usare clip fine-tuned
  --search-backend auto                                # se si può usare faiss, lo fa

# Analisi di una singola immagine (con dino)
python -m src.scene.analyze_scene \
  --scene data/scenes_real/images/scene_0000.jpg \
  --index index_dino \
  --backend dino
  --out outputs_scenes_dino \
  --search-backend auto                                # se si può usare faiss, lo fa

# Analisi di un insieme di immagini
python -m src.scene.batch_analyze \
  --index index_ft \
  --out outputs_scenes_ft \
  --backend clip \                                    # per usare DINOv2: dino
  --search-backend auto \                             # se si può usare faiss, lo fa
  --finetuned runs/clip_ft_v1/best.pt                 # per usare clip fine-tuned
```
<img width="330" height="263" alt="analisi clip no_ft" src="https://github.com/user-attachments/assets/0f95e7c1-918f-4a8e-a813-f5c87575dd21" />
<img width="330" height="263" alt="analisi clip ft" src="https://github.com/user-attachments/assets/b7767326-aa93-4cb2-b65c-1e98c9d6285e" />
<img width="330" height="263" alt="analisi clip dino" src="https://github.com/user-attachments/assets/f63e6da6-a41c-4423-b728-2691364dd34d" />

---

## Valutazione
Matching GT↔Pred con Hungarian e IoU≥0.5 (1 pred per GT), poi accuracy top-1 oggetto-livello:
```bash
python -m src.eval.eval_scene \
  --gt data/scenes_real/gt.json \
  --preds-dir outputs_scenes_ft/json \  #modifica con la posizione delle analisi degli altri modelli
  --verbose
  # --use-overlay   # valuta solo gli overlay JSON (per vedere "precisione UI")
  # --iou-match 0.5
```

---

## Advanced Vector DB (FAISS)
Crea indice FAISS e confronta i tempi con la ricerca naive:
```bash
# durante la creazione indice
python -m src.prep.build_index --manifest manifests/train.csv --out index_ft --faiss

# benchmark (modifica --index per confrontare con altri modelli)
python -m src.eval.bench_vector_db --index index_ft --build-faiss --queries 500 --k 5
```

---

## UI Gradio
Switch CLIP-FT / DINOv2 da dropdown; la UI seleziona automaticamente indice, backend e (per CLIP) il checkpoint fine-tuned:
```bash
python -m src.ui.app_gradio
# apri http://127.0.0.1:7860
```
Tab:
- Singola scena — carica immagine, overlay & crops, download JSON
- Database Query — query per immagine o (solo CLIP) per testo
Controlli: SAM min area, max masks, NMS/threshold, bbox expand, RGBA→RGB bg, ecc.

---

## Risultati
Su 31 scene reali (valutazione eval JSON):
| Modello               | Object Acc   | Detect Recall  | Cls\|matched  |
| --------------------- | -----------: | --------------: | -------------: |
| **CLIP (pre-FT)**     |   **0.7258** |      **0.8929** |     **0.9000** |
| **CLIP (fine-tuned)** |   **0.6935** |      **0.8929** |     **0.8600** |
| **DINOv2 (ViT-S/14)** |   **0.6290** |      **0.8929** |     **0.7800** |


Osservazioni:
- CLIP-FT consigliato per produzione (migliori score in analisi, robusto allo stile anime, supporta query testuali).
- Il confronto CLIP no–FT vs CLIP FT va letto alla luce della diversa copertura del database: l'indice no–FT contiene 3–4x più esempi per classe (in particolare per Naruto), il che favorisce la ricerca top–1.
- DINOv2 incluso come embedding alternativo (richiesta traccia); peggiore su tratti anime (es. Sakura), discreto sugli altri.

---

## Riconoscimenti
- <a href="https://github.com/facebookresearch/segment-anything">Segment Anything (SAM)</a> — Meta AI
- <a href="https://github.com/openai/CLIP">CLIP</a>  — OpenAI / LAION / community OpenCLIP
- <a href="https://github.com/facebookresearch/dinov2">DINOv2</a> — Meta AI
- <a href="https://github.com/facebookresearch/faiss">FAISS</a> — Facebook AI Research
- E: NumPy, PyTorch, Gradio, SciPy, Timm, Pillow, OpenCV, Pandas, TQDM.

---

## Autore
Luca Crispino (MSc Computer Engineering – Cybersecurity and Cloud at Politecnico di Bari)

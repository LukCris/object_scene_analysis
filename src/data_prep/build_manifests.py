"""
build_manifest.py

Obiettivo
---------
Costruire manifest "puliti" (CSV con colonne `path,label`) a partire dai file
Roboflow-style `_classes.csv` presenti in `train/`, `valid/`, `test/`.

Policy implementata
-------------------
`single_only`: mantiene SOLO le immagini che hanno esattamente un personaggio
etichettato fra le colonne indicate (default: `Gara, Naruto, Sakura, Tsunade`).
Le immagini con 0 o >1 classi positive vengono scartate.
Le righe con colonna `Unlabeled` vengono scartate.

Output prodotti (in `--out`)
----------------------------
- `train.csv`, `valid.csv`, `test.csv`: manifest con header `path,label`. I path
  sono normalizzati in formato POSIX (`/`).
- `classes.json`: lista delle classi/colonne considerate, nell'ordine fornito.
- `stats_<split>.json`: statistiche dettagliate per split (kept/discarded, ecc.).

Note d'uso (CLI)
----------------
Esempio:
    python build_manifest.py --root data --out manifests

Prerequisiti struttura cartelle:
    <root>/train/_classes.csv
    <root>/valid/_classes.csv
    <root>/test/_classes.csv

Dove ogni `_classes.csv` contiene almeno le colonne: `filename`,
`Gara, Naruto, Sakura, Tsunade` (nel dataset usato Gaara è indicato come `Gara`).

"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import argparse
import json
import logging
import pandas as pd


# -----------------------------------------------------------------------------
# Strutture dati
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RowKeep:
    """Riga da mantenere nel manifest finale.

    Attributi
    ---------
    path: str
        Path normalizzato (POSIX) all'immagine sul disco.
    label: str
        Etichetta (nome colonna) associata all'immagine.
    """
    path: str
    label: str

# -----------------------------------------------------------------------------
# Costruttore dei manifest
# -----------------------------------------------------------------------------

class ManifestBuilder:
    """
    Costruisce manifest (path,label) a partire dai CSV Roboflow `_classes.csv`.

    Strategia adottata
    ------------------
    - policy = 'single_only':
        * Accetta solo righe con esattamente un `1` tra le colonne dei personaggi.
        * Scarta immagini con 0 o >1 classi positive.
        * Esclude righe `Unlabeled` se presente.

    Validazioni implementate
    ------------------------
    - Trim degli spazi nei nomi colonna (" Naruto" -> "Naruto").
    - Verifica esistenza colonna `filename`.
    - Verifica esistenza delle colonne `person_cols` (warning se mancanti).
    - Controllo esistenza file su disco (righe ignorate se path assente).

    Parametri
    ---------
    root_dir: str
        Cartella radice del dataset contenente `train/`, `valid/`, `test/`.
    out_dir: str
        Cartella di output per manifest e statistiche.
    policy: str
        Politica di filtraggio (solo 'single_only' supportata).
    include_unlabeled: bool
        Placeholder per future estensioni (non usato in 'single_only').
    person_cols: Iterable[str]
        Colonne (etichette) da considerare come classi valide.
    """

    def __init__(
        self,
        root_dir: str = "../../data",
        out_dir: str = "../../manifests",
        policy: str = "single_only",
        include_unlabeled: bool = False,
        person_cols: Iterable[str] = ("Gara", "Naruto", "Sakura", "Tsunade"),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.out_dir = Path(out_dir)
        self.policy = policy
        self.include_unlabeled = include_unlabeled
        self.person_cols: List[str] = list(person_cols)

        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Lettura e pulizia input CSV
    # ---------------------------
    def _read_split_csv(self, split: str) -> pd.DataFrame:
        """Legge `<root>/<split>/_classes.csv` e sanifica i nomi colonna.

        Parameters
        ----------
        split : str
            Nome dello split: 'train', 'valid' o 'test'.

        Returns
        -------
        pandas.DataFrame
            DataFrame del CSV, con `df.attrs['split'] = split` per logging.

        Raises
        ------
        FileNotFoundError
            Se il file CSV non esiste.
        ValueError
            Se manca la colonna obbligatoria `filename`.
        """
        csv_path = self.root_dir / split / "_classes.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV non trovato: {csv_path}")

        df = pd.read_csv(csv_path)
        # Trim degli spazi nei nomi colonna
        df.columns = df.columns.str.strip()
        if "filename" not in df.columns:
            raise ValueError(f"[{split}] Colonna obbligatoria 'filename' mancante in {csv_path}")

        # Conserviamo lo split per uso downstream (logging/stats)
        df.attrs["split"] = split
        return df

    # ---------------------------------------
    # Filtraggio secondo la policy single_only
    # ---------------------------------------
    def _filter_rows_single_only(self, df: pd.DataFrame) -> tuple[list[RowKeep], dict]:
        """Applica la regola "single_only" al DataFrame di uno split.

        Regola: la somma dei valori sulle colonne `person_cols` presenti deve essere
        esattamente 1. Se 0 o >1, la riga viene scartata.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame letto da `_read_split_csv` per uno specifico split.

        Returns
        -------
        (list[RowKeep], dict)
            - Lista di elementi da mantenere nel manifest.
            - Dizionario con statistiche (kept/discarded, per-label, note, ecc.).
        """
        split: str = df.attrs.get("split", "unknown")

        # 1) Verifica colonne disponibili
        available_cols = set(df.columns)
        missing_person_cols = [c for c in self.person_cols if c not in available_cols]
        notes: list[str] = []
        if missing_person_cols:
            notes.append(
                f"[{split}] Colonne etichetta mancanti: {missing_person_cols}. "
                "Verifica che i nomi coincidano con quelli del dataset Roboflow."
            )

        # 2) Rimuove `Unlabeled` se presente
        drop_cols = [c for c in ["Unlabeled"] if c in available_cols]
        df_no_unlbl = df.drop(columns=drop_cols) if drop_cols else df

        # 3) Considera solo le colonne person presenti
        person_cols_present = [c for c in self.person_cols if c in df_no_unlbl.columns]
        if not person_cols_present:
            raise ValueError(
                f"[{split}] Nessuna colonna dei personaggi trovata tra {self.person_cols}."
            )

        kept: list[RowKeep] = []
        discards: Counter[int] = Counter()
        kept_per_label: Counter[str] = Counter()
        missing_files_count = 0

        # 4) Itera e applica la regola `single_only`
        for _, row in df_no_unlbl.iterrows():
            sub = row[person_cols_present]
            # Se ci fossero NaN, li trattiamo come 0
            num_ones = int(sub.fillna(0).sum())

            if num_ones == 1:
                label = sub.idxmax()  # trova la colonna con valore massimo (=1)
                filename = str(row["filename"]).strip()
                rel_path = (self.root_dir / split / filename).as_posix()

                # Includiamo solo se il file esiste fisicamente
                if Path(rel_path).exists():
                    kept.append(RowKeep(path=rel_path, label=label))
                    kept_per_label[label] += 1
                else:
                    missing_files_count += 1
            else:
                # 0 = nessuna classe positiva; >1 = multipli
                discards[num_ones] += 1

        stats: Dict[str, object] = {
            "total_rows_kept": int(len(kept)),
            "total_rows_discarded": int(sum(discards.values())),
            # Distribuzione per numero di classi positive scartate (0,2,3,...)
            "discarded_by_num_positives": {int(k): int(v) for k, v in discards.items()},
            "missing_files": int(missing_files_count),
            "kept_per_label": {k: int(v) for k, v in kept_per_label.items()},
            "note": notes,
        }

        logging.info(f"[{split}] {stats}")
        return kept, stats

    # ---------------------------
    # Scrittura file di output
    # ---------------------------
    def _write_manifest(self, split: str, rows: Iterable[RowKeep]) -> int:
        """Scrive `<out>/<split>.csv` con righe uniche ordinate per (label, path)."""
        unique = sorted({(r.path, r.label) for r in rows}, key=lambda x: (x[1], x[0]))
        out_csv = self.out_dir / f"{split}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            f.write("path,label\n")
            for p, lab in unique:
                f.write(f"{Path(p).as_posix()},{lab}\n")
        return len(unique)

    def _write_classes_json(self) -> None:
        """Salva `classes.json` con l'elenco delle classi considerate."""
        classes_path = self.out_dir / "classes.json"
        with classes_path.open("w", encoding="utf-8") as f:
            json.dump(self.person_cols, f, ensure_ascii=False, indent=2)

    def _write_stats_json(self, split: str, stats: dict) -> None:
        """Salva `stats_<split>.json` con le statistiche dello split."""
        out_json = self.out_dir / f"stats_{split}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    # ---------------------------
    # Entry: costruzione di tutti gli split
    # ---------------------------
    def build_all(self) -> dict:
        """Costruisce manifest + stats per gli split `train`, `valid`, `test`.

        Returns
        -------
        dict
            Mappa `split -> stats` arricchita con `written_rows`.
        """
        if self.policy != "single_only":
            raise NotImplementedError("Solo 'single_only' è supportata in questa baseline.")

        summaries: dict = {}
        for split in ("train", "valid", "test"):
            df = self._read_split_csv(split)
            kept, stats = self._filter_rows_single_only(df)
            n_written = self._write_manifest(split, kept)
            stats["written_rows"] = int(n_written)
            self._write_stats_json(split, stats)
            summaries[split] = stats

        self._write_classes_json()
        return summaries


# -----------------------------------------------------------------------------
# Interfaccia a riga di comando
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Costruisci manifest (path,label) dai CSV Roboflow."
    )
    parser.add_argument(
        "--root", default="data", help="Cartella dataset con train/ valid/ test/"
    )
    parser.add_argument(
        "--out", default="manifests", help="Cartella di output per manifest e stats"
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=["Gara", "Naruto", "Sakura", "Tsunade"],
        help="Lista delle classi/colonne da considerare (ordine canonico).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    builder = ManifestBuilder(
        root_dir=args.root,
        out_dir=args.out,
        policy="single_only",
        include_unlabeled=False,
        person_cols=args.labels,
    )
    summaries = builder.build_all()

    print(f"\nManifest e statistiche salvati in: {args.out}")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

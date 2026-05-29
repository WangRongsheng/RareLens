from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _clean_orphacode(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _load_orpha_entries(ontology_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(ontology_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    out: List[Dict[str, Any]] = []
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        oc = _clean_orphacode(k)
        name = str(v.get("name") or "").strip()
        if oc is None or not name:
            continue
        out.append({"orphacode": oc, "disease_name": name})
    return out


def _load_entries(entries_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(entries_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        oc = _clean_orphacode(row.get("orphacode"))
        name = str(row.get("disease_name") or "").strip()
        if oc is None or not name:
            continue
        out.append({"orphacode": oc, "disease_name": name})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--entries",
        required=True,
        help="Path to cache entries.json, e.g. rare_diagnosis/orphacode_rag_cache/entries.json",
    )
    ap.add_argument(
        "--out-dir",
        required=False,
        default="",
        help="Output directory for vector cache; defaults to the parent directory of --entries",
    )
    ap.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
        help="SentenceTransformer model used for both cache build and runtime query encoding",
    )
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    try:
        import numpy as np  # type: ignore
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise SystemExit(f"Missing deps (need numpy, faiss, sentence-transformers): {exc}")

    entries_path = Path(args.entries).resolve()
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else entries_path.parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_entries(entries_path)
    if not entries:
        raise SystemExit("No entries loaded from entries.json")

    texts = [e["disease_name"] for e in entries]
    encoder = SentenceTransformer(str(args.embedding_model).strip())
    mat = encoder.encode(
        texts,
        batch_size=max(1, int(args.batch)),
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    mat = np.asarray(mat, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    index: Any = faiss.IndexFlatIP(int(mat.shape[1]))
    index.add(mat)

    (out_dir / "entries.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    np.save(str(out_dir / "kb_embeddings.npy"), mat)
    faiss.write_index(index, str(out_dir / "faiss_index.bin"))
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "embedding_model": str(args.embedding_model).strip(),
                "entries_path": str(entries_path),
                "entries_count": len(entries),
                "dimension": int(mat.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"OK. entries={len(entries)} dim={mat.shape[1]} out={out_dir} "
        f"embedding_model={str(args.embedding_model).strip()}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


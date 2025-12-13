
import os
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm.auto import tqdm

# -------------------------
# Helper functions (pure)
# -------------------------

def find_value(d, keys_substrings):
    """
    d: dict with lowercase keys
    keys_substrings: list of substrings to match in keys
    """
    for k, v in d.items():
        k_norm = k.lower()
        for sub in keys_substrings:
            if sub in k_norm:
                return str(v).strip()
    return ""


def extract_attrs_one(spec: dict) -> dict:
    """
    Extract core camera attributes from a single spec JSON (raw dict).
    Keys are matched by substrings, not exact names.
    """
    spec_lower = {k.lower(): v for k, v in spec.items()}

    brand = find_value(spec_lower, [
        "brand", "manufacturer", "maker", "vendor", "company"
    ])

    model = find_value(spec_lower, [
        "model", "product name", "name", "product", "title", "page title",
        "model number", "model no", "mfr part", "part number", "item model",
        "sku", "product id"
    ])

    sensor = find_value(spec_lower, [
        "sensor", "sensor type", "image sensor", "sensor size",
        "imaging sensor", "cmos", "ccd"
    ])

    lens = find_value(spec_lower, [
        "lens", "lens type", "focal length", "kit", "optical zoom",
        "zoom lens", "lens configuration", "lens model", "lens description"
    ])

    megapixels = find_value(spec_lower, [
        "megapixel", "mp", "effective megapixels", "resolution",
        "total pixels"
    ])

    mount = find_value(spec_lower, [
        "mount", "lens mount", "compatible mount", "mount type"
    ])

    video = find_value(spec_lower, [
        "video", "video resolution", "max video", "movie", "recording"
    ])

    camera_type = find_value(spec_lower, [
        "camera type", "dslr", "slr", "mirrorless", "compact camera",
        "point and shoot", "bridge camera", "system camera", "digital camera"
    ])

    body_info = find_value(spec_lower, [
        "body only", "camera body only", "kit", "camera kit",
        "with lens", "with 18-55mm lens", "with 28-70mm lens"
    ])

    return {
        "brand": brand,
        "model": model,
        "sensor": sensor,
        "lens": lens,
        "megapixels": megapixels,
        "mount": mount,
        "video": video,
        "camera_type": camera_type,
        "body_info": body_info,
    }


def camera_signature_from_attrs(attrs: dict) -> str:
    """
    Turn extracted attrs into a compact text signature.
    """
    brand       = (attrs.get("brand") or "").strip()
    model       = (attrs.get("model") or "").strip()
    cam_type    = (attrs.get("camera_type") or "").strip()
    body_info   = (attrs.get("body_info") or "").strip()
    megapixels  = (attrs.get("megapixels") or "").strip()
    sensor      = (attrs.get("sensor") or "").strip()
    mount       = (attrs.get("mount") or "").strip()
    lens        = (attrs.get("lens") or "").strip()
    video       = (attrs.get("video") or "").strip()

    name_part = " ".join(x for x in [brand, model] if x)
    if cam_type:
        name_part = (name_part + " " + cam_type).strip()
    if not name_part:
        name_part = "Digital camera"

    pieces = [name_part]

    if body_info:
        pieces.append(f"Body: {body_info}")
    if megapixels or sensor:
        spec_str = ", ".join(x for x in [megapixels, sensor] if x)
        pieces.append(f"Sensor: {spec_str}")
    if mount:
        pieces.append(f"Mount: {mount}")
    if lens:
        pieces.append(f"Lens: {lens}")
    if video:
        pieces.append(f"Video: {video}")

    return " | ".join(pieces)


FIELDS = [
    "brand", "model", "sensor", "lens",
    "megapixels", "mount", "video",
    "camera_type", "body_info",
]


def merge_attrs(spec_list):
    """
    Given a list of spec dicts belonging to the SAME entity,
    pick majority value for each attribute.
    """
    collected = {f: [] for f in FIELDS}
    for spec in spec_list:
        attrs = extract_attrs_one(spec)
        for f in FIELDS:
            val = attrs.get(f, "")
            if val:
                collected[f].append(val)

    def pick_majority(vals):
        if not vals:
            return ""
        return Counter(vals).most_common(1)[0][0]

    return {f: pick_majority(v) for f, v in collected.items()}


def get_model_core(s: str) -> str:
    """
    Extract "core" token like 70d, 5d3, etc. from model string.
    """
    if not s:
        return ""
    s = str(s).lower()
    toks = re.findall(r"[a-z0-9]+", s)
    for t in toks:
        if any(c.isdigit() for c in t) and any(c.isalpha() for c in t):
            return t
    return ""


def spec_to_record_text(spec: dict):
    """
    Convert a raw spec dict into (signature_text, attrs_dict).
    """
    attrs = extract_attrs_one(spec)
    sig = camera_signature_from_attrs(attrs)
    return sig, attrs


# -------------------------
# Main class
# -------------------------

class CameraEntityResolver:
    """
    Importable library-style class.

    On init:
      - loads GT mapping
      - loads all spec JSONs per entity
      - builds canonical_df
      - encodes canonical embeddings
      - loads SBERT and Qwen LLM

    At inference:
      - you only pass raw spec dict / JSON / file / spec_id.
    """

    def __init__(
            self,
            base_dir: str = "camera/camera_specs",
            map_csv: str = "camera/camera_ground_truths/camera_entity_resolution_gt.csv",
            emb_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            # llm_name: str = "Qwen/Qwen2.5-3B-Instruct",
            sim_min: float = 0.35,
            # llm_trust_sim: float = 0.55,
            load_gt: bool = True,
            use_cache: bool = True,
            cache_dir: str = "camera/cache",
            force_rebuild: bool = False,
        ):
            self.base_dir = base_dir
            self.map_csv = map_csv
            self.sim_min = sim_min
            # self.llm_trust_sim = llm_trust_sim
    
            self.use_cache = use_cache
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)
    
            # ground truth mapping (optional in production)
            if load_gt and os.path.exists(map_csv):
                self.gt_df = pd.read_csv(map_csv)
            else:
                self.gt_df = None
    
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
    
            # ------- cache paths -------
            df_cache = os.path.join(self.cache_dir, "canonical_df.parquet")
            emb_cache = os.path.join(self.cache_dir, "canonical_emb.npy")
    
            cache_ok = (
                use_cache
                and (not force_rebuild)
                and os.path.exists(df_cache)
                and os.path.exists(emb_cache)
            )
    
            # ------- load embedding model (used for both cache and fresh build) -------
            self.emb_model = SentenceTransformer(
                emb_model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
    
            # ------- either load from cache or build from scratch -------
            if cache_ok:
                # Fast path: just load canonical_df and embedding matrix
                self.canonical_df = pd.read_parquet(df_cache)
                if "model_core" not in self.canonical_df.columns:
                    self.canonical_df["model_core"] = self.canonical_df["model"].apply(
                        get_model_core
                    )
                self.canonical_ids = self.canonical_df["entity_id"].tolist()
                self.canonical_emb = np.load(emb_cache)
                # optional: you can skip building entity_to_specs in cached mode
                self.entity_to_specs = {}
            else:
                # Slow path (first time or force rebuild):
                # 1) build entity_to_specs (reads all JSONs)
                self.entity_to_specs = self._build_entity_to_specs()
    
                # 2) build canonical_df
                self.canonical_df = self._build_canonical_df()
    
                # 3) add model_core
                if "model_core" not in self.canonical_df.columns:
                    self.canonical_df["model_core"] = self.canonical_df["model"].apply(
                        get_model_core
                    )
    
                # 4) encode canonical embeddings
                self._encode_canonical_embeddings()
    
                # 5) save to cache for future inits
                if use_cache:
                    self.canonical_df.to_parquet(df_cache, index=False)
                    np.save(emb_cache, self.canonical_emb)
    
            # ------- LLM -------
            # self.llm_tokenizer, self.llm_model = self._load_llm(llm_name)

    # -------------------------
    # Internal builders
    # -------------------------

    def _build_entity_to_specs(self):
        """
        From GT CSV, load each spec JSON and group by entity_id.
        """
        if self.gt_df is None:
            return {}

        entity_to_specs = {}
        grouped = self.gt_df.groupby("entity_id")["spec_id"].apply(list)

        for eid, spec_list in grouped.items():
            specs = []
            for spec in spec_list:
                folder, file_id = spec.split("//")
                path = os.path.join(self.base_dir, folder, file_id + ".json")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        specs.append(json.load(f))
            entity_to_specs[eid] = specs

        return entity_to_specs

    def _build_canonical_df(self) -> pd.DataFrame:
        """
        Build canonical rows from merged attrs of specs per entity.
        """
        canonical_rows = []

        for eid, spec_list in self.entity_to_specs.items():
            merged = merge_attrs(spec_list)
            sig = camera_signature_from_attrs(merged)
            canonical_rows.append(
                {
                    "entity_id": eid,
                    "canonical_sig": sig,
                    "canonical_summary": sig,
                    **merged,
                }
            )

        if not canonical_rows:
            # empty DF if nothing (e.g., no GT loaded)
            return pd.DataFrame(
                columns=[
                    "entity_id",
                    "canonical_sig",
                    "canonical_summary",
                    *FIELDS,
                ]
            )

        return pd.DataFrame(canonical_rows)

    def _encode_canonical_embeddings(self):
        """
        Encode canonical signatures to embeddings and normalize.
        """
        texts = self.canonical_df["canonical_sig"].fillna("").tolist()
        self.canonical_ids = self.canonical_df["entity_id"].tolist()

        if len(texts) == 0:
            self.canonical_emb = np.zeros((0, 384), dtype=np.float32)
            return

        emb = self.emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.canonical_emb = emb

    def _get_candidate_entity_ids(self, record_sig: str, attrs: dict, top_k=10, pool_k=100):
        """
        Use SBERT + optional model_core boost to get candidate entity IDs.
        """
        if self.canonical_emb.shape[0] == 0:
            return []

        rec_emb = self.emb_model.encode([record_sig], convert_to_numpy=True)
        rec_emb = rec_emb / np.linalg.norm(rec_emb, axis=1, keepdims=True)

        sims_all = self.canonical_emb @ rec_emb[0]  # (n_entities,)

        # pool by embedding similarity
        idx_pool = np.argsort(-sims_all)[:pool_k]

        # optional model_core boost
        rec_model_core = get_model_core(attrs.get("model", ""))
        model_core_boost = {}

        if rec_model_core:
            mask_core = self.canonical_df["model_core"] == rec_model_core
            core_indices = np.where(mask_core.values)[0]
            if len(core_indices) > 0:
                sims_core = sims_all[core_indices]
                order_core = np.argsort(-sims_core)
                for idx_local in order_core[:min(len(order_core), pool_k)]:
                    i = core_indices[idx_local]
                    eid = self.canonical_df.iloc[i]["entity_id"]
                    model_core_boost[eid] = 0.2  # tune if needed

        # combine
        score_map = {}
        for i in idx_pool:
            eid = self.canonical_df.iloc[i]["entity_id"]
            base = float(sims_all[i])
            score_map[eid] = base

        for eid, boost in model_core_boost.items():
            base = score_map.get(eid, None)
            if base is None:
                idx = self.canonical_df.index[self.canonical_df["entity_id"] == eid][0]
                base = float(sims_all[idx])
                score_map[eid] = base + boost
            else:
                score_map[eid] = base + boost

        sorted_eids = sorted(score_map.items(), key=lambda x: -x[1])[:top_k]
        return [(eid, score_map[eid]) for eid, _ in sorted_eids]


    

    # -------------------------
    # Public API
    # -------------------------

    def load_spec_id(self, spec_id: str) -> dict:
        """
        Load a spec JSON using spec_id like 'www.ebay.com//54514'.
        """
        folder, file_id = spec_id.split("//")
        path = os.path.join(self.base_dir, folder, f"{file_id}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def resolve_from_dict(self, spec: dict) -> str:
        """
        Main method for real-time use (cosine-only).
        spec: raw JSON dict for one camera.
    
        Returns:
            entity_id (string) or "NEW_ENTITY".
        """
        record_sig, rec_attrs = spec_to_record_text(spec)
    
        candidates = self._get_candidate_entity_ids(
            record_sig,
            rec_attrs,
            top_k=10,
            pool_k=100,
        )
    
        best_raw = None
        best_raw_sim = -1.0
    
        for eid, sim in candidates:
            sim = float(sim)
            if sim > best_raw_sim:
                best_raw_sim = sim
                best_raw = eid
    
        if best_raw is not None and best_raw_sim >= self.sim_min:
            return best_raw
    
        return "NEW_ENTITY"

    
    def resolve_from_json_str(self, json_str: str) -> str:
        """
        Same as resolve_from_dict, but input is JSON string.
        """
        spec = json.loads(json_str)
        return self.resolve_from_dict(spec)

    def resolve_from_file(self, file_path: str) -> str:
        """
        Load JSON file from disk and resolve.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return self.resolve_from_dict(spec)

    def resolve_from_spec_id(self, spec_id: str) -> str:
        """
        Load spec using spec_id like 'www.ebay.com//54514' and resolve.
        """
        spec = self.load_spec_id(spec_id)
        return self.resolve_from_dict(spec)
    def evaluate_resolver(resolver, spec_ids=None, show_report=False):
        """
        Computes overall accuracy/precision/recall/F1 for entity_id classification.
    
        resolver must have:
          - resolver.gt_df with columns: ["spec_id", "entity_id"]
          - resolver.load_spec_id(spec_id)
          - resolver.resolve_from_dict(spec_dict)
        """
        gt_df = resolver.gt_df
        if gt_df is None or gt_df.empty:
            raise ValueError("resolver.gt_df is empty or not loaded.")
    
        if spec_ids is None:
            spec_ids = gt_df["spec_id"].dropna().unique().tolist()
    
        y_true, y_pred = [], []
    
        for spec_id in tqdm(spec_ids, desc="Evaluating"):
            row = gt_df[gt_df["spec_id"] == spec_id]
            if row.empty:
                continue
    
            true_eid = str(row["entity_id"].iloc[0])
            spec = resolver.load_spec_id(spec_id)
            pred_eid = str(resolver.resolve_from_dict(spec))
    
            y_true.append(true_eid)
            y_pred.append(pred_eid)
    
        # ---- overall metrics (includes NEW_ENTITY as a normal class if predicted) ----
        acc = accuracy_score(y_true, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    
        out = {
            "n": len(y_true),
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
        }
    
        if show_report:
            out["classification_report"] = classification_report(
                y_true, y_pred, zero_division=0
            )
    
        return out


    def evaluate_known_only(resolver, spec_ids=None):
        """
        Optional: evaluate only on samples where prediction is NOT NEW_ENTITY.
        This tells you performance when the resolver actually links to an existing entity.
        """
        gt_df = resolver.gt_df
        if gt_df is None or gt_df.empty:
            raise ValueError("resolver.gt_df is empty or not loaded.")
    
        if spec_ids is None:
            spec_ids = gt_df["spec_id"].dropna().unique().tolist()
    
        y_true, y_pred = [], []
    
        for spec_id in tqdm(spec_ids, desc="Evaluating (known-only)"):
            row = gt_df[gt_df["spec_id"] == spec_id]
            if row.empty:
                continue
    
            true_eid = str(row["entity_id"].iloc[0])
            spec = resolver.load_spec_id(spec_id)
            pred_eid = str(resolver.resolve_from_dict(spec))
    
            if pred_eid == "NEW_ENTITY":
                continue
    
            y_true.append(true_eid)
            y_pred.append(pred_eid)
    
        acc = accuracy_score(y_true, y_pred) if y_true else 0.0
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
    
        return {
            "n_known": len(y_true),
            "accuracy_known": acc,
            "precision_macro_known": prec,
            "recall_macro_known": rec,
            "f1_macro_known": f1,
        }

    def inspect_spec_id(self, spec_id: str):
        """
        Debug helper for offline analysis: print GT vs predicted.
        Only works if gt_df is available.
        """
        if self.gt_df is None:
            print("No ground-truth CSV loaded; cannot inspect.")
            return

        gt_rows = self.gt_df[self.gt_df["spec_id"] == spec_id]
        if gt_rows.empty:
            print(f"No GT found for spec_id={spec_id}")
            gt_entity = None
        else:
            gt_entity = gt_rows["entity_id"].iloc[0]

        spec = self.load_spec_id(spec_id)
        pred_entity = self.resolve_from_dict(spec)

        gt_canon = None
        pred_canon = None

        if gt_entity is not None and gt_entity in self.canonical_df["entity_id"].values:
            gt_canon = self.canonical_df.loc[
                self.canonical_df["entity_id"] == gt_entity, "canonical_summary"
            ].iloc[0]

        if pred_entity != "NEW_ENTITY" and pred_entity in self.canonical_df["entity_id"].values:
            pred_canon = self.canonical_df.loc[
                self.canonical_df["entity_id"] == pred_entity, "canonical_summary"
            ].iloc[0]

        print("spec_id:", spec_id)
        print("GT entity_id:      ", gt_entity)
        print("Predicted entity_id:", pred_entity)
        print("\nGT canonical summary:")
        print(gt_canon)
        print("\nPredicted canonical summary:")
        print(pred_canon)
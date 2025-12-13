import streamlit as st
import json 
import re
from typing import Dict, Any
import pandas as pd
import streamlit as st
import os
# Import your libs
from schema_matcher_lib import SchemaMatcher
from camera_entity_lib import CameraEntityResolver
from imputation_lib import ImputationConfig, DataImputationEngine


import torch



# =======================
# Helper functions
# =======================

def spec_id_to_json_path(spec_id: str, specs_root: str) -> str | None:
   
    if not isinstance(spec_id, str):
        return None

    parts = spec_id.split("//", 1)
    if len(parts) != 2:
        # unexpected format
        return None

    site_part = parts[0].strip().strip("/")      # "www.ebay.com"
    id_part   = parts[1].strip().strip("/")      # "46670"

    if not site_part or not id_part:
        return None

    json_fname = f"{id_part}.json"
    json_path = os.path.join(specs_root, site_part, json_fname)
    return json_path

def build_manual_spec_from_inputs(attrs: dict) -> dict:
    """
    Build a minimal spec dict from manually entered attributes.
    Right now it's just a shallow copy, but you can wrap it later if needed.
    """
    return dict(attrs)


def build_entity_index(gt_csv_path: str, specs_root: str) -> dict[str, list[str]]:
    df = pd.read_csv(gt_csv_path)

    entity_index: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        # normalize entity_id to string consistently
        entity_id = str(row["entity_id"]).strip()
        spec_id   = str(row["spec_id"]).strip()

        json_path = spec_id_to_json_path(spec_id, specs_root)
        if json_path is None:
            continue
        if not os.path.exists(json_path):
            # debug: see what is missing
            # print(f"Missing spec file for {spec_id}: {json_path}")
            continue

        entity_index.setdefault(entity_id, []).append(json_path)

    return entity_index


def extract_site_from_json(json_obj: Dict[str, Any]) -> str | None:
    """
    Extract site name from <page title> by taking the final word.
    Example:
        "<page title>": "Fujifilm FinePix S2950HD New Zealand Prices - PriceMe"
        → "PriceMe"
    """

    if not isinstance(json_obj, dict):
        return None

    # Get title from multiple possible keys
    title = (
        json_obj.get("<page title>")
        or json_obj.get("page title")
        or json_obj.get("title")
    )

    if not isinstance(title, str) or len(title.strip()) == 0:
        return None

    # split into tokens
    tokens = re.split(r"\s+", title.strip())
    if not tokens:
        return None

    # last word
    last_word = tokens[-1]

    # remove trailing punctuation like .,?!|:- etc.
    last_word = re.sub(r"[^\w\-\.]+$", "", last_word)

    if len(last_word) == 0:
        return None

    return last_word  



def flatten_json(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Turn nested JSON into flat key → value mapping.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def canonicalize_spec(
    flat_spec: Dict[str, Any],
    site: str,
    matcher: "SchemaMatcher",
) -> Dict[str, Any]:
    """
    Map (raw_attr, value) -> canonical_attr using SchemaMatcher.
    Supports both string and dict outputs.
    """
    canonical_spec: Dict[str, Any] = {}

    for raw_attr, value in flat_spec.items():
        value_str = "" if value is None else str(value)

        pred = matcher.predict_property(
            site=site,
            attr=raw_attr,
            value=value_str,
        )

        canonical_name = None
        confidence = 1.0

        if isinstance(pred, dict):
            canonical_name = (
                pred.get("canonical_field")
                or pred.get("canonical_name")
                or pred.get("property")
                or pred.get("label")
            )
            if "confidence" in pred:
                try:
                    confidence = float(pred["confidence"])
                except (TypeError, ValueError):
                    confidence = 1.0

        elif isinstance(pred, str):
            canonical_name = pred.strip()
            confidence = 1.0

        if not canonical_name:
            continue

        canonical_spec[canonical_name] = value

    return canonical_spec



# =======================
# Model init (once)
# =======================
if "matcher" not in st.session_state:
    st.session_state["matcher"] = SchemaMatcher(
        canonical_csv="canonical_fields.csv",
        base_model_name="Qwen/Qwen2-0.5B-Instruct",
        adapter_dir="qwen0.5-schema-matching-lora_latest",
    )

if "entity_resolver" not in st.session_state:
    st.session_state["entity_resolver"] = CameraEntityResolver()



specs_root = "camera/camera_specs"
gt_csv_path = "camera/camera_ground_truths/camera_entity_resolution_gt.csv"

entity_index = build_entity_index(gt_csv_path, specs_root)

# ---- NEW: imputation engine (once) ----
if "imputer_engine" not in st.session_state:
    entity_df = pd.read_csv("entity_clean.csv", low_memory=False)
    config = ImputationConfig(
        max_records_to_analyze=10,
        use_keyword_filtering=True,
        use_summarization=True,
        confidence_threshold=0.4,
        llm_model="qwen",
    )
    st.session_state["imputer_engine"] = DataImputationEngine(entity_df, config)

matcher = st.session_state["matcher"]
resolver = st.session_state["entity_resolver"]

assert matcher.device.type == "cuda"
assert resolver.device.type == "cuda"


## ------------------------------
# 1. Schema matching only
# ------------------------------
def run_schema_matching(
    json_obj,
    matcher: "SchemaMatcher",
):
    
    st.subheader("Schema Matching – Output")
    

    # -------------------------------
    # Auto extract site
    # -------------------------------
    site = extract_site_from_json(json_obj)
  

    # -------------------------------
    # Flatten + canonicalize
    # -------------------------------
    flat_spec = flatten_json(json_obj)
    canonical_spec = canonicalize_spec(
        site=site,
        flat_spec=flat_spec,
        matcher=matcher,

    )

    # -------------------------------
    # Prepare comparison table
    # -------------------------------
    rows = []

    for raw_attr, raw_value in flat_spec.items():
        canon_attr = None
        canon_value = None

        # If canonical mapped key matches this raw attribute
        for c_name, c_val in canonical_spec.items():
            # We don't know exact mapping → matcher only gives final canonical dict
            # So simply store canonical dict separately
            pass

        rows.append({
            "Original Attribute": raw_attr,
            "Original Value": raw_value,
        })

    # Canonical table
    canonical_rows = []
    for c_attr, c_val in canonical_spec.items():
        canonical_rows.append({
            # "site": site,
            "Canonical Attribute": c_attr,
            "Value": c_val,
        })

    # -------------------------------
    # Layout: two columns
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Attributes")
        st.dataframe(rows, use_container_width=True)

    with col2:
        st.markdown("### Canonical (Schema-Matched) Attributes")
        st.dataframe(canonical_rows, use_container_width=True)

    return canonical_spec


#single schema

def run_single_field_schema_match(
    source_name: str,
    attr_name: str,
    attr_value: str,
    matcher: "SchemaMatcher",
):
    """
    Call SchemaMatcher on a single (site, attr, value) triple
    and display the result nicely.
    """
    site = source_name.strip() or None

    pred = matcher.predict_property(
        site=site,
        attr=attr_name.strip(),
        value=attr_value.strip(),
    )

    # Handle both string and dict outputs
    if isinstance(pred, str):
        st.markdown("**Predicted canonical field:**")
        st.code(pred)
    elif isinstance(pred, dict):
        st.markdown("**Schema Matcher Output**")
        st.json(pred)
    else:
        st.warning(f"Unexpected prediction type: {type(pred)}")


# ------------------------------
# 2. Entity resolution only (RAW)
# ------------------------------
import random
import json

def run_entity_resolution(
    json_obj,
    resolver: "CameraEntityResolver",
    entity_index: dict[str, list[str]],
    view_data: bool = True,     # <--- NEW PARAM
):
    st.subheader("Entity Resolution – Output")

    # 1) predict entity
    entity_id = resolver.resolve_from_dict(json_obj)

    st.markdown("**Predicted entity ID (from raw spec)**")
    st.code(repr(entity_id))


    if isinstance(json_obj, dict):
        flat = flatten_json(json_obj)
    else:
        flat = {}

    target_record = {"entity_id": entity_id}
    for key in ["brand", "product_name", "model", "screen_size","battery_life","weight","price","storage","color"]:
        if key in flat:
            target_record[key] = flat[key]

    st.session_state["last_entity_id"] = entity_id
    st.session_state["last_target_record"] = target_record

    # If user does NOT want to view JSON data, stop here
    if not view_data:
        return entity_id

    # 2) show a random JSON example for the same entity (if available)
    if (
        entity_index is not None
        and entity_id in entity_index
        and entity_index[entity_id]
    ):
        ref_path = random.choice(entity_index[entity_id])
        abs_ref_path = os.path.abspath(ref_path)

        try:
            with open(abs_ref_path, "r", encoding="utf-8") as f:
                ref_json = json.load(f)

            st.markdown("---")
            st.markdown("### Example reference record for this entity")

            st.write("Reference JSON path:")
            st.code(abs_ref_path)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current uploaded JSON (raw)**")
                st.json(json_obj)
            with col2:
                st.markdown(f"**Random reference JSON for {entity_id}**")
                st.json(ref_json)

        except Exception as e:
            st.warning(f"Could not load reference JSON for entity {entity_id}: {e}")

    else:
        st.info("No reference JSON available for this entity in the index.")

    return entity_id

# ------------------------------
# 3. Schema + Entity (combined)
# ------------------------------
def run_schema_and_entity_resolution(
    json_obj,
    matcher: "SchemaMatcher",
    resolver: "CameraEntityResolver",
    entity_index: dict[str, list[str]] | None = None,
):
    """
    Combined view:
    - Schema matching for canonical view (display only, single table:
      Original Attribute | Canonical Attribute | Value).
    - Entity resolution on raw JSON (prints only entity_id).
    - Builds a processed record (entity_id + canonical attrs) and
      exposes download buttons (JSON + CSV).
    """

    st.subheader("Schema Matching + Entity Resolution – Output")

    # 1) Schema matching (already shows tables in the UI)
    canonical_spec = run_schema_matching(
        json_obj=json_obj,
        matcher=matcher,
    )

    # 2) Entity resolution on RAW json_obj
    entity_id = run_entity_resolution(
        json_obj=json_obj,
        resolver=resolver,
        entity_index=entity_index,
        view_data=True,   # keep JSON/ref view
    )

    st.markdown("**Canonical attributes (see table above)**")
    st.write(f"Total canonical attributes: {len(canonical_spec)}")

    # 3) Build final processed record: entity + canonical schema
    processed_record = {"entity_id": entity_id}
    processed_record.update(canonical_spec)

    st.markdown("---")
    st.markdown("### Download processed data")

    # JSON download
    processed_json_str = json.dumps(processed_record, indent=2)
    st.download_button(
        label="Download processed record (JSON)",
        data=processed_json_str,
        file_name=f"processed_{entity_id or 'record'}.json",
        mime="application/json",
    )

    # CSV download (one row, flat)
    df_out = pd.DataFrame([processed_record])
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download processed record (CSV)",
        data=csv_bytes,
        file_name=f"processed_{entity_id or 'record'}.csv",
        mime="text/csv",
    )

    return {
        "entity_id": entity_id,
        "canonical_spec": canonical_spec,
        "processed_record": processed_record,
    }




# ------------------------------
# 4. Imputation (still placeholder)
# ------------------------------
def run_imputation(field_name: str, debug: bool = False):
    """
    Run data imputation for a single attribute, using the last resolved entity.
    """
    st.subheader("Imputation – Output")

    engine = st.session_state.get("imputer_engine")
    target_record = st.session_state.get("last_target_record")

    if engine is None or target_record is None:
        st.warning("Run entity resolution first so we know the entity and have a target record.")
        return

    if debug:
        st.info(
            f"DEBUG: Imputing '{field_name}' for entity_id={target_record.get('entity_id')}"
        )

    imputed_value = engine.impute(target_record, field_name)

    st.markdown("**Target record used for imputation**")
    st.json(target_record)

    st.markdown(f"**Imputed value for '{field_name}'**")
    st.write(imputed_value)

    return imputed_value




# =======================
# Page config
# =======================
st.set_page_config(
    page_title=" Ensuring Data Quality: An LLM-Enhanced Framework for Data Diagnosis and Resolution ",
    page_icon="🧪",
    layout="wide",
)

# =======================
# Sidebar
# =======================
with st.sidebar:
    st.title("Ready to Upload a file?")
    

    # ==============
    # JSON upload
    # ==============
    st.subheader("Upload JSON")
    uploaded_json_file = st.file_uploader(
        "Upload a JSON file",
        type=["json"],
        help="Only .json files are accepted.",
    )

    if "uploaded_json" not in st.session_state:
        st.session_state.uploaded_json = None

    if uploaded_json_file is not None:
        try:
            st.session_state.uploaded_json = json.load(uploaded_json_file)
            st.success("JSON file loaded successfully.")
            if isinstance(st.session_state.uploaded_json, dict):
                st.write(
                    "Top-level keys:",
                    list(st.session_state.uploaded_json.keys())[:10],
                )
            else:
                st.write(
                    "Loaded JSON type:",
                    type(st.session_state.uploaded_json).__name__,
                )
        except Exception as e:
            st.session_state.uploaded_json = None
            st.error(f"Failed to read JSON: {e}")

    st.markdown("---")

    # ==============
    # Controls
    # ==============
    st.subheader("Controls")

    mode = st.selectbox(
        "Mode",
        ["Schema Matching", "Entity Resolution", "Schema_Entity Resolution" ,"Imputation"],
        index=0,
    )

    # debug = st.checkbox("Debug mode", value=False)
    st.markdown("---")

# ---- NEW: Imputation settings (only when mode == "Imputation") ----
    impute_field = None
    if mode == "Imputation":
        st.markdown("### Imputation Settings")
        impute_field = st.text_input(
            "Attribute name to impute",
            placeholder="e.g., mega_pixel",
            key="impute_field_main",
        )
    # ------------------------------------------------------------------


    reset = st.button("Reset session")

    if reset:
        for key in ["uploaded_json", "entered_attributes"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Session reset.")

# =======================
# Main Header
# =======================
st.set_page_config(layout="wide")

st.markdown(
    """
    ### Ensuring Data Quality: An LLM-Enhanced Framework for Data Diagnosis and Resolution

    This application provides an end-to-end workflow for exploring,
    diagnosing, and improving data quality. Use the sidebar to upload data,
    inspect attributes, apply schema alignment, and test automated
    cleaning or imputation processes.
    """
)

st.markdown("---")




st.markdown("## Run Analysis")
run_btn = st.button("Get analytics")

json_obj = st.session_state.get("uploaded_json", None)

if run_btn:
    if json_obj is None:
        st.warning("Upload a JSON file in the sidebar first.")
    else:
        if mode == "Schema Matching":
            run_schema_matching(json_obj,matcher=matcher)
        elif mode == "Entity Resolution":
            run_entity_resolution(json_obj,resolver=resolver,entity_index=entity_index)
        elif mode == "Schema_Entity Resolution":
            run_schema_and_entity_resolution(json_obj,resolver=resolver,matcher=matcher,entity_index=entity_index)
        elif mode == "Imputation":
            if not impute_field:
                st.warning("Enter the attribute name to impute in 'Imputation Settings'.")
            else:
                run_imputation(impute_field)


# =======================
# Two-column layout
# =======================
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Schema Resolver")

    st.markdown("### Key in the Details")

    source_name = st.text_input("Source / Website Name", placeholder="e.g., www.ebay.com")
    attr_name   = st.text_input("Attribute Name", placeholder="e.g., megapixels")
    attr_value  = st.text_input("Attribute Value", placeholder="e.g., 20 MP")

    run_btn = st.button("Run Schema Matcher")

    st.markdown("---")
    

    # actually run the matcher when button clicked
    if run_btn:
        if "matcher" not in st.session_state:
            st.error("SchemaMatcher is not initialized in session_state.matcher.")
        elif not attr_name or not attr_value:
            st.warning("Provide both Attribute Name and Attribute Value.")
        else:
            run_single_field_schema_match(
                source_name=source_name,
                attr_name=attr_name,
                attr_value=attr_value,
                matcher=st.session_state.matcher,
            )


# Initialize storage once
if "entered_attributes" not in st.session_state:
    st.session_state.entered_attributes = {}

with right_col:
    st.subheader("Entity Resolution")

    st.markdown("### Key in the Details")

    possible_attrs = [
        "brand",
        "model",
        "megapixels",
        "sensor_type",
        "screen_size",
        "lens_range",
        "video_resolution",
        "mount",
    ]

    selected_attr = st.selectbox(
        "Choose an attribute",
        possible_attrs,
        index=0,
    )

    # dynamic input field
    input_val = st.text_input(
        f"Enter value for '{selected_attr}'",
        placeholder=f"Type the {selected_attr} value",
        key=f"input_{selected_attr}",
    )

    # When user presses Enter, Streamlit reruns and input_val is updated
    if input_val:
        st.session_state.entered_attributes[selected_attr] = input_val

    st.markdown("### Current Inputs")

    if st.session_state.entered_attributes:
        for k, v in st.session_state.entered_attributes.items():
            st.write(f"**{k}**: {v}")
    else:
        st.write("No attributes added yet.")

    st.markdown("---")

    # button to run entity matching on the manually entered attributes
    run_entity_btn = st.button("Run Entity Matching")

    if run_entity_btn:
        if "entity_resolver" not in st.session_state:
            st.error("CameraEntityResolver is not initialized in session_state.entity_resolver.")
        elif not st.session_state.entered_attributes:
            st.warning("Add at least one attribute before running entity matching.")
        else:
            manual_spec = build_manual_spec_from_inputs(st.session_state.entered_attributes)
            run_entity_resolution(
                json_obj=manual_spec,
                resolver=st.session_state.entity_resolver,
                entity_index=entity_index
               # uses the same debug flag from sidebar
            )

    clear = st.button("Clear All")

    if clear:
        st.session_state.entered_attributes = {}
        st.rerun()


# Optional footer
st.markdown("---")
st.caption("Prototype UI scaffold – plug in your own features above.")

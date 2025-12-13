# schema_matcher.py

import json
from typing import List, Dict, Any, Iterable, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class SchemaMatcher:
    """
    Library-style wrapper around Qwen 0.5B + LoRA adapter
    for schema matching (canonical property prediction).

    Typical usage:

        matcher = SchemaMatcher(
            canonical_csv="canonical_fields.csv",
            base_model_name="Qwen/Qwen2-0.5B-Instruct",
            adapter_dir="qwen0.5-schema-matching-lora_latest"
        )

        canon = matcher.predict_property(site="amazon.com",
                                         attr="Resolution",
                                         value="1920 x 1080")

        # batch from list
        records = [
            {"site": "amazon.com", "attr": "Resolution", "value": "1920 x 1080"},
            {"site": "bestbuy.ca", "attr": "Brand", "value": "Sony"},
        ]
        preds = matcher.predict_list(records)

        # batch from JSON file
        preds = matcher.predict_from_json(
            "records.json",
            site_key="site",
            attr_key="attr",
            value_key="value",
        )
    """

    def __init__(
        self,
        canonical_csv: str = "canonical_fields.csv",
        base_model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        adapter_dir: str = "qwen0.5-schema-matching-lora_latest",
    ):
        # ---- device ----
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # ---- canonical fields ----
        df_fields = pd.read_csv(canonical_csv)
        self.canonical_fields: List[str] = df_fields["canonical_property"].tolist()

        # ---- tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )
        # ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- base model + LoRA ----
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
        )
        base_model.to(self.device)

        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
        ).to(self.device)

        self.model.eval()

    # =========================
    # Prompt building (same as training)
    # =========================

    def build_messages(self, site: str, attr: str, value: str) -> List[Dict[str, str]]:
        user_prompt = f"""You are given one attribute from a product specification record.
Your task is to identify which canonical property it belongs to.

Information:
- Source website: {site}
- attribute name: {attr}
- attribute value: {value}

Choose exactly one canonical property from the list below:
{chr(10).join(self.canonical_fields)}

Respond with only the canonical property name and nothing else."""
        messages = [
            {
                "role": "system",
                "content": "You are a schema-matching assistant. Map attributes to canonical properties.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        return messages

    def build_prompt(self, site: str, attr: str, value: str) -> str:
        messages = self.build_messages(site, attr, value)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    # =========================
    # Single prediction
    # =========================

    @torch.inference_mode()
    def predict_property(
        self,
        site: str,
        attr: str,
        value: str,
        max_new_tokens: int = 8,
    ) -> str:
        """
        Predict one canonical property for a single attribute.
        """
        prompt = self.build_prompt(site, attr, value)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # slice only generated tokens
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        canonical = text.strip().splitlines()[0].strip()
        return canonical

    # =========================
    # Batch helpers
    # =========================

    def _normalize_record(
        self,
        record: Any,
        default_site: str = "",
    ) -> Tuple[str, str, str]:
        """
        Accept different record shapes and normalize to (site, attr, value).

        Supported:
          - dict with keys: "site", "attr", "value"
          - dict with keys: "site", "attribute", "attribute_name", "field", etc. (attr)
          - dict with keys: "value", "val", "attribute_value" (value)
          - tuple/list: (site, attr, value)
        """
        if isinstance(record, (tuple, list)) and len(record) == 3:
            site, attr, value = record
            return str(site), str(attr), str(value)

        if isinstance(record, dict):
            site = str(
                record.get("site", record.get("source", default_site))
            )
            # attribute name
            attr = record.get("attr") or record.get("attribute") \
                or record.get("attribute_name") or record.get("field")
            # attribute value
            value = record.get("value") or record.get("val") \
                or record.get("attribute_value") or record.get("field_value")

            if attr is None or value is None:
                raise ValueError(
                    f"Record missing attr/value keys: {record}"
                )
            return str(site), str(attr), str(value)

        raise TypeError(f"Unsupported record type: {type(record)}")

    @torch.inference_mode()
    def predict_list(
        self,
        records: Iterable[Any],
        max_new_tokens: int = 8,
    ) -> List[str]:
        """
        Predict canonical property for a list of records.

        `records` can be:
          - list of dicts
          - list of (site, attr, value) tuples

        Returns:
          list of canonical property strings, same order as input.
        """
        preds: List[str] = []
        for rec in records:
            site, attr, value = self._normalize_record(rec)
            pred = self.predict_property(site, attr, value, max_new_tokens=max_new_tokens)
            preds.append(pred)
        return preds

    @torch.inference_mode()
    def predict_from_json(
        self,
        json_path: str,
        site_key: str = "site",
        attr_key: str = "attr",
        value_key: str = "value",
        max_new_tokens: int = 8,
    ) -> List[str]:
        """
        Load a JSON file containing a list of records and predict for each.

        JSON must be like:
            [
              {"site": "...", "attr": "...", "value": "..."},
              ...
            ]

        If your keys are different, override `site_key`, `attr_key`, `value_key`.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of records")

        records = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"Each item must be dict, got: {type(item)}")
            site = str(item.get(site_key, ""))
            attr = item.get(attr_key)
            value = item.get(value_key)
            if attr is None or value is None:
                raise ValueError(
                    f"Missing attr/value ({attr_key}/{value_key}) in item: {item}"
                )
            records.append({"site": site, "attr": attr, "value": value})

        return self.predict_list(records, max_new_tokens=max_new_tokens)

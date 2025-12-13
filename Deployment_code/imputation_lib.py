# imputation_lib.py
#
# Lightweight library-style version of your imputation code.
# - No demo / CLI
# - No prints (only logging, which you can configure in your app)
# - Public API returns ONLY the imputed value for the requested field
# - You load entity_df outside; the engine only receives target_record

import json
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)  # you configure logging in your app


# ==============================
# Config + simple result holder
# ==============================

@dataclass
class ImputationConfig:
    max_records_to_analyze: int = 20
    use_keyword_filtering: bool = True
    use_summarization: bool = True
    confidence_threshold: float = 0.7
    llm_model: str = "qwen"        # or "claude"
    api_key: Optional[str] = None  # for Claude if needed


# ==============================
# Keyword extraction
# ==============================

class KeywordExtractor:
    FIELD_KEYWORDS = {
        'mega_pixel': ['megapixel', 'mp', 'resolution', 'camera', 'photo', 'image', 'sensor'],
        'screen_size': ['screen', 'display', 'inch', 'monitor', 'lcd', 'led', 'size'],
        'battery_life': ['battery', 'mah', 'hours', 'power', 'charge', 'capacity'],
        'weight': ['weight', 'gram', 'kg', 'lb', 'ounce', 'oz', 'heavy', 'light'],
        'price': ['price', 'cost', 'usd', 'dollar', '$', 'msrp', 'retail'],
        'storage': ['storage', 'gb', 'tb', 'memory', 'capacity', 'ssd', 'hdd'],
        'ram': ['ram', 'memory', 'gb', 'ddr', 'sdram'],
        'processor': ['processor', 'cpu', 'ghz', 'core', 'intel', 'amd', 'chip'],
        'color': ['color', 'colour', 'black', 'white', 'silver', 'red', 'blue'],
        'brand': ['brand', 'manufacturer', 'make', 'company'],
    }

    @classmethod
    def extract_keywords(cls, field_name: str) -> List[str]:
        field_lower = field_name.lower().replace('_', ' ')

        if field_lower in cls.FIELD_KEYWORDS:
            return cls.FIELD_KEYWORDS[field_lower]

        for key, keywords in cls.FIELD_KEYWORDS.items():
            if key in field_lower or field_lower in key:
                return keywords

        return [field_lower] + field_lower.split()


# ==============================
# Record filtering
# ==============================

class RecordFilter:
    @staticmethod
    def score_record(record: Dict[str, Any], keywords: List[str]) -> float:
        record_text = json.dumps(record).lower()
        score = 0
        for keyword in keywords:
            if keyword in record_text:
                score += record_text.count(keyword)
        return score

    @classmethod
    def filter_relevant_records(
        cls,
        records: List[Dict[str, Any]],
        field_name: str,
        max_records: int = 20,
    ) -> List[Dict[str, Any]]:
        keywords = KeywordExtractor.extract_keywords(field_name)
        logger.info(f"Using keywords: {keywords}")

        scored_records = []
        for record in records:
            score = cls.score_record(record, keywords)
            if score > 0:
                scored_records.append({**record, "_relevance_score": score})

        scored_records.sort(key=lambda x: x["_relevance_score"], reverse=True)
        top_records = scored_records[:max_records]

        logger.info(f"Filtered {len(records)} records down to {len(top_records)} relevant records")
        return top_records


# ==============================
# Record summarization
# ==============================

class RecordSummarizer:
    @staticmethod
    def extract_field_values(records: List[Dict[str, Any]], field_name: str) -> List[Dict[str, Any]]:
        field_values = []
        field_lower = field_name.lower()

        for record in records:
            for key, value in record.items():
                if key.startswith("_"):
                    continue

                key_lower = key.lower()
                if field_lower in key_lower or any(word in key_lower for word in field_lower.split('_')):
                    if value and str(value).strip():
                        field_values.append(
                            {
                                "key": key,
                                "value": value,
                                "source": record.get("spec_id", record.get("source", "unknown")),
                            }
                        )
        return field_values

    @classmethod
    def create_summary(cls, records: List[Dict[str, Any]], field_name: str) -> Dict[str, Any]:
        summary = {
            "total_records": len(records),
            "field_name": field_name,
            "sources": list({r.get("spec_id", r.get("source", "unknown")) for r in records}),
            "field_values": cls.extract_field_values(records, field_name),
            "sample_records": records[:3] if len(records) > 3 else records,
        }

        all_keys = Counter()
        for record in records:
            all_keys.update([k for k in record.keys() if not k.startswith("_")])

        summary["common_attributes"] = [k for k, _ in all_keys.most_common(10)]
        logger.info(f"Created summary with {len(summary['field_values'])} field values from {len(records)} records")
        return summary


# ==============================
# Qwen model loader (global)
# ==============================

QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
_QWEN_MODEL = None
_QWEN_TOKENIZER = None


def get_qwen_model_and_tokenizer():
    global _QWEN_MODEL, _QWEN_TOKENIZER

    if _QWEN_MODEL is None or _QWEN_TOKENIZER is None:
        logger.info(f"Loading Qwen model ONCE: {QWEN_MODEL_NAME}")
        _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME,
            trust_remote_code=True,
        )
        _QWEN_MODEL = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )
    return _QWEN_MODEL, _QWEN_TOKENIZER


# ==============================
# LLM imputer
# ==============================

class LLMImputer:
    def __init__(self, config: ImputationConfig):
        self.config = config
        self.api_key = config.api_key
        self.MAX_PROMPT_TOKENS = 512   # reduced for speed
        self.MAX_NEW_TOKENS = 64       # reduced for speed

    def _extract_clean_values(self, summary: Dict[str, Any], field_name: str) -> Dict[str, int]:
        field_values = summary.get("field_values", [])
        value_counts: Dict[str, int] = {}

        for fv in field_values:
            value = fv.get("value")
            if value is None:
                continue
            value_str = str(value).strip()
            if not value_str or value_str.lower() in ["nan", "none", ""]:
                continue
            value_counts[value_str] = value_counts.get(value_str, 0) + 1

        return value_counts

    def _create_prompt(
        self,
        summary: Dict[str, Any],
        target_record: Dict[str, Any],
        field_name: str,
    ) -> str:
        value_counts = self._extract_clean_values(summary, field_name)

        if not value_counts:
            # No useful values; we will skip LLM later
            return ""

        value_list = []
        for value, count in sorted(value_counts.items(), key=lambda x: -x[1])[:5]:
            value_list.append(f"- {value} (seen {count} times)")
        values_text = "\n".join(value_list)

        product_name = target_record.get("product_name", "Unknown Product")
        brand = target_record.get("brand", "")
        product_type = target_record.get("product_type", "")

        # Short prompt, model should answer with JSON
        prompt = f"""Task: Predict '{field_name}' for this product.

Product: {product_name}
Brand: {brand}
Type: {product_type}

Similar products have these '{field_name}' values:
{values_text}

From the options above, pick the most likely '{field_name}'.
Return ONLY JSON in this exact format (no extra text):

{{"imputed_value": "<value or null>", "confidence": 0.0, "reasoning": "<short reason>"}}"""
        return prompt

    def _call_qwen_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not prompt:
            # No values, nothing to impute
            return None

        try:
            model, tokenizer = get_qwen_model_and_tokenizer()

            messages = [
                {"role": "system", "content": "You are a data analyst. Respond only with JSON."},
                {"role": "user", "content": prompt},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            model_inputs = tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_PROMPT_TOKENS,
            )
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    temperature=0.4,
                    top_p=0.9,
                    do_sample=False,  # deterministic for consistency
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_length = model_inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, input_length:]

            response = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            matches = list(re.finditer(json_pattern, response))

            for match in matches:
                try:
                    candidate = json.loads(match.group())
                    if "imputed_value" in candidate and "confidence" in candidate:
                        return candidate
                except json.JSONDecodeError:
                    continue

            # Fallback: very simple extraction
            value_match = re.search(r'"imputed_value"\s*:\s*"([^"]+)"', response)
            conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', response)
            if value_match:
                imputed_value = value_match.group(1)
                confidence = float(conf_match.group(1)) if conf_match else 0.5
                return {
                    "imputed_value": imputed_value,
                    "confidence": confidence,
                    "reasoning": "Extracted from partial response",
                }

            return None

        except Exception as e:
            logger.error(f"Error calling Qwen model: {e}")
            return None

    def _call_claude_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not prompt:
            return None

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text

            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text)
            if json_match:
                return json.loads(json_match.group())

            return None

        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return None

    def impute(
        self,
        summary: Dict[str, Any],
        target_record: Dict[str, Any],
        field_name: str,
    ) -> Optional[Dict[str, Any]]:
        prompt = self._create_prompt(summary, target_record, field_name)
        if not prompt:
            # Nothing to impute
            return None

        if self.config.llm_model == "qwen":
            result = self._call_qwen_api(prompt)
        elif self.config.llm_model == "claude":
            result = self._call_claude_api(prompt)
        else:
            logger.error(f"Unsupported LLM model: {self.config.llm_model}")
            return None

        if result and "imputed_value" in result and "confidence" in result:
            return result

        return None


# ==============================
# Main pipeline (library-style)
# ==============================

class DataImputationPipeline:
    def __init__(self, config: Optional[ImputationConfig] = None):
        self.config = config or ImputationConfig()
        self.imputer = LLMImputer(self.config)

    @staticmethod
    def get_similar_records(entity_df: pd.DataFrame, entity_id: str) -> List[Dict[str, Any]]:
        similar = entity_df[entity_df["entity_id"] == entity_id]
        return similar.to_dict("records")

    def impute_missing_field(
        self,
        target_record: Dict[str, Any],
        entity_df: pd.DataFrame,
        field_name: str,
    ) -> Optional[Any]:
        """
        Library API: returns ONLY the imputed value (or None).

        target_record must contain 'entity_id'.
        entity_df is your pre-loaded DataFrame (entity_clean.csv or similar).
        """
        entity_id = target_record.get("entity_id")
        if not entity_id:
            logger.error("No 'entity_id' in target_record")
            return None

        similar_records = self.get_similar_records(entity_df, entity_id)
        if not similar_records:
            return None

        total_similar = len(similar_records)

        if self.config.use_keyword_filtering:
            similar_records = RecordFilter.filter_relevant_records(
                similar_records,
                field_name,
                self.config.max_records_to_analyze,
            )

        if not similar_records:
            # no relevant records → nothing to impute
            return None

        if self.config.use_summarization:
            summary = RecordSummarizer.create_summary(similar_records, field_name)
        else:
            summary = {
                "total_records": len(similar_records),
                "field_name": field_name,
                "field_values": [],
                "records": similar_records[: self.config.max_records_to_analyze],
            }

        imputation_result = self.imputer.impute(summary, target_record, field_name)
        if not imputation_result:
            return None

        confidence = float(imputation_result.get("confidence", 0.0))
        if confidence < self.config.confidence_threshold:
            return None

        # Return ONLY the imputed value, as requested
        return imputation_result.get("imputed_value")


# ==============================
# Convenience engine
# ==============================

class DataImputationEngine:
    """
    High-level class for your app:

    engine = DataImputationEngine(entity_df, config)
    value = engine.impute(target_record, "mega_pixel")
    """

    def __init__(self, entity_df: pd.DataFrame, config: Optional[ImputationConfig] = None):
        self.entity_df = entity_df
        self.pipeline = DataImputationPipeline(config)

    def impute(self, target_record: Dict[str, Any], field_name: str) -> Optional[Any]:
        return self.pipeline.impute_missing_field(
            target_record=target_record,
            entity_df=self.entity_df,
            field_name=field_name,
        )


# ==============================
# OPTIONAL: simple functional API
# ==============================

def impute_field(
    target_record: Dict[str, Any],
    entity_df: pd.DataFrame,
    field_name: str,
    config: Optional[ImputationConfig] = None,
) -> Optional[Any]:
    """
    Functional convenience wrapper.

    Example:
        entity_df = pd.read_csv("entity_clean.csv")
        target_record = {"entity_id": "ENTITY#10", "brand": "Sony", ...}
        value = impute_field(target_record, entity_df, "mega_pixel")
    """
    pipeline = DataImputationPipeline(config)
    return pipeline.impute_missing_field(
        target_record=target_record,
        entity_df=entity_df,
        field_name=field_name,
    )

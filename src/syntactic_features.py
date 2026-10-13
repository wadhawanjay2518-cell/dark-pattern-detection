"""
src/syntactic_features.py

Extracts fixed-length linguistic and syntactic feature vectors using spaCy:
  1. Part-of-speech (POS) distributions
  2. Imperative and verb-related cues
  3. Dependency-based syntactic relations & parse tree depth
  4. Negation and modality
  5. Time expressions, quantities, comparative-price expressions, and social references
  6. Text length, sentence-level characteristics, and orthography
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import joblib
import numpy as np
import pandas as pd
import spacy
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler


FEATURE_DEFINITIONS: Dict[str, str] = {
    # Text length & sentence-level characteristics
    "length_char": "Total character count of the sanitized text string.",
    "length_token": "Total token count parsed by spaCy.",
    "length_word": "Count of alphanumeric word tokens.",
    "avg_token_len": "Average character length per token.",
    "uppercase_ratio": "Proportion of uppercase letters among alphabetic characters.",
    "exclamation_count": "Count of exclamation mark punctuation characters.",
    "question_count": "Count of question mark punctuation characters.",

    # POS distributions
    "pos_prop_VERB": "Proportion of tokens tagged as VERB (action/process verbs).",
    "pos_prop_NOUN": "Proportion of tokens tagged as NOUN (common nouns).",
    "pos_prop_ADJ": "Proportion of tokens tagged as ADJ (qualitative adjectives).",
    "pos_prop_ADV": "Proportion of tokens tagged as ADV (adverbial modifiers).",
    "pos_prop_PRON": "Proportion of tokens tagged as PRON (personal/relative pronouns).",
    "pos_prop_NUM": "Proportion of tokens tagged as NUM (cardinal numbers).",
    "pos_prop_AUX": "Proportion of tokens tagged as AUX (auxiliary and modal verbs).",
    "pos_prop_CCONJ": "Proportion of coordinating conjunctions (and, but, or).",
    "pos_prop_SCONJ": "Proportion of subordinating conjunctions (if, because, while).",
    "pos_prop_DET": "Proportion of determiners (the, a, this, that).",
    "pos_prop_PUNCT": "Proportion of punctuation tokens.",

    # Dependency relations & syntactic depth
    "dep_prop_nsubj": "Proportion of nominal subject dependency relations.",
    "dep_prop_dobj": "Proportion of direct object dependency relations.",
    "dep_prop_advmod": "Proportion of adverbial modifier dependency relations.",
    "dep_prop_amod": "Proportion of adjectival modifier dependency relations.",
    "dep_prop_prep": "Proportion of prepositional modifier dependency relations.",
    "max_tree_depth": "Maximum dependency tree depth representing syntactic embedding complexity.",

    # Imperative & verb-related cues
    "is_imperative": "Binary indicator for directive/imperative clause (base verb root without explicit subject).",
    "is_verb_initial": "Binary indicator for sentence starting with a verb or auxiliary.",

    # Negation & modality
    "modal_count": "Count of modal auxiliary verbs (can, could, will, would, must, should, may, might).",
    "modal_prop": "Proportion of modal auxiliary verbs relative to token count.",
    "negation_count": "Count of negation dependency tokens (neg) or negative particles (no, not, never).",
    "has_negation": "Binary indicator for the presence of negation.",

    # Pronoun distributions
    "pron_1p_count": "Count of 1st-person pronouns (I, me, my, we, us, our).",
    "pron_2p_count": "Count of 2nd-person directive pronouns (you, your, yours).",
    "pron_3p_count": "Count of 3rd-person pronouns (he, she, it, they, them, their).",

    # Time expressions, quantities, comparative-price, and social cues
    "time_expression_count": "Frequency of temporal interval markers (seconds, minutes, hours, days, now, soon, today).",
    "has_time_expression": "Binary indicator for presence of temporal constraint markers.",
    "quantity_scarcity_cues": "Frequency of stock/inventory and scarcity markers (left, stock, remaining, only, sold, claimed).",
    "comparative_price_cues": "Frequency of comparative pricing/discount cues (save, off, was, regular, discount, best, deal).",
    "social_reference_cues": "Frequency of crowd/social activity markers (people, viewed, bought, others, purchased, customers, demand).",
    "digit_token_ratio": "Proportion of tokens containing digits or numerical quantities.",
    "has_currency_or_percent": "Binary indicator for presence of currency symbols ($, £, €) or percentage signs (%).",
}

FEATURE_NAMES: List[str] = list(FEATURE_DEFINITIONS.keys())

FIRST_PERSON_PRONS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
SECOND_PERSON_PRONS = {"you", "your", "yours", "yourself", "yourselves"}
THIRD_PERSON_PRONS = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"}

TIME_WORDS = {
    "second", "seconds", "sec", "minute", "minutes", "min", "mins",
    "hour", "hours", "hr", "hrs", "day", "days", "today", "tonight",
    "now", "soon", "ends", "ending", "expires", "expiring", "limited", "hurry", "fast", "deadline"
}

SCARCITY_WORDS = {
    "left", "stock", "remaining", "only", "sold", "pieces", "items", "reserve",
    "reserved", "claimed", "last", "almost", "gone", "running", "low"
}

COMPARATIVE_WORDS = {
    "save", "saving", "off", "was", "now", "retail", "original", "regular",
    "discount", "deal", "best", "seller", "bestseller", "cheapest", "lowest", "free", "value"
}

SOCIAL_WORDS = {
    "people", "person", "bought", "purchased", "viewed", "viewing", "others",
    "customers", "shoppers", "ordered", "recent", "recently", "demand", "popular", "rated", "reviews"
}


def get_token_depth(token) -> int:
    """Computes the depth of a token relative to the root with cycle protection."""
    depth = 0
    current = token
    while current.head != current:
        depth += 1
        current = current.head
        if depth > 50:
            break
    return depth


class SyntacticFeatureExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            self.nlp = spacy.blank("en")
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = FEATURE_NAMES

    def extract_from_doc(self, doc, raw_text: str) -> Dict[str, float]:
        """Extracts documented linguistic and syntactic features from a spaCy Doc."""
        tokens = [t for t in doc]
        token_count = len(tokens)
        safe_token_count = max(1, token_count)

        words = [t for t in tokens if t.is_alpha]
        word_count = len(words)

        length_char = float(len(raw_text))
        length_token = float(token_count)
        length_word = float(word_count)
        avg_token_len = float(np.mean([len(t.text) for t in tokens])) if token_count > 0 else 0.0

        # POS proportions
        pos_counts: Dict[str, int] = {
            "VERB": 0, "NOUN": 0, "ADJ": 0, "ADV": 0, "PRON": 0,
            "NUM": 0, "AUX": 0, "CCONJ": 0, "SCONJ": 0, "DET": 0, "PUNCT": 0
        }
        for t in tokens:
            if t.pos_ in pos_counts:
                pos_counts[t.pos_] += 1

        pos_props = {f"pos_prop_{k}": float(v / safe_token_count) for k, v in pos_counts.items()}

        # Dependency proportions
        dep_counts: Dict[str, int] = {
            "nsubj": 0, "dobj": 0, "advmod": 0, "amod": 0, "prep": 0
        }
        for t in tokens:
            dep = t.dep_.lower()
            if dep in dep_counts:
                dep_counts[dep] += 1
            elif dep == "obj":
                dep_counts["dobj"] += 1

        dep_props = {f"dep_prop_{k}": float(v / safe_token_count) for k, v in dep_counts.items()}

        # Imperative detection: root verb in base form (VB) without explicit nominal subject
        is_imperative = 0.0
        root_tokens = [t for t in tokens if t.dep_ == "ROOT"]
        if root_tokens:
            root = root_tokens[0]
            if root.pos_ == "VERB" and root.tag_ in {"VB", "VBP"}:
                has_subj = any(child.dep_ in {"nsubj", "nsubjpass"} for child in root.children)
                if not has_subj:
                    is_imperative = 1.0

        is_verb_initial = 1.0 if (tokens and tokens[0].pos_ in {"VERB", "AUX"}) else 0.0

        # Modals & Negation
        modal_tokens = [t for t in tokens if t.tag_ == "MD"]
        modal_count = float(len(modal_tokens))
        modal_prop = float(modal_count / safe_token_count)

        neg_tokens = [
            t for t in tokens
            if t.dep_ == "neg" or t.lower_ in {"not", "no", "never", "n't", "none", "neither"}
        ]
        negation_count = float(len(neg_tokens))
        has_negation = 1.0 if negation_count > 0 else 0.0

        # Pronoun counts
        pron_1p = 0.0
        pron_2p = 0.0
        pron_3p = 0.0
        for t in tokens:
            w = t.lower_
            if w in FIRST_PERSON_PRONS:
                pron_1p += 1.0
            elif w in SECOND_PERSON_PRONS:
                pron_2p += 1.0
            elif w in THIRD_PERSON_PRONS:
                pron_3p += 1.0

        # Punctuation & Orthography
        exclamation_count = float(raw_text.count("!"))
        question_count = float(raw_text.count("?"))

        alpha_chars = [c for c in raw_text if c.isalpha()]
        uppercase_chars = [c for c in alpha_chars if c.isupper()]
        uppercase_ratio = float(len(uppercase_chars) / max(1, len(alpha_chars)))

        digit_tokens = [t for t in tokens if any(c.isdigit() for c in t.text)]
        digit_token_ratio = float(len(digit_tokens) / safe_token_count)
        has_currency_or_percent = 1.0 if any(c in raw_text for c in ["$", "%", "£", "€"]) else 0.0

        # Semantic/Linguistic cues for 4-class domain
        token_lemmas = [t.lemma_.lower() for t in tokens]
        token_lowers = [t.lower_ for t in tokens]
        all_token_words = set(token_lemmas + token_lowers)

        time_expression_count = float(len([w for w in all_token_words if w in TIME_WORDS]))
        has_time_expression = 1.0 if time_expression_count > 0 else 0.0

        quantity_scarcity_cues = float(len([w for w in all_token_words if w in SCARCITY_WORDS]))
        comparative_price_cues = float(len([w for w in all_token_words if w in COMPARATIVE_WORDS]))
        social_reference_cues = float(len([w for w in all_token_words if w in SOCIAL_WORDS]))

        # Tree Depth
        max_tree_depth = float(max([get_token_depth(t) for t in tokens])) if tokens else 0.0

        features = {
            "length_char": length_char,
            "length_token": length_token,
            "length_word": length_word,
            "avg_token_len": avg_token_len,
            "uppercase_ratio": uppercase_ratio,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            **pos_props,
            **dep_props,
            "max_tree_depth": max_tree_depth,
            "is_imperative": is_imperative,
            "is_verb_initial": is_verb_initial,
            "modal_count": modal_count,
            "modal_prop": modal_prop,
            "negation_count": negation_count,
            "has_negation": has_negation,
            "pron_1p_count": pron_1p,
            "pron_2p_count": pron_2p,
            "pron_3p_count": pron_3p,
            "time_expression_count": time_expression_count,
            "has_time_expression": has_time_expression,
            "quantity_scarcity_cues": quantity_scarcity_cues,
            "comparative_price_cues": comparative_price_cues,
            "social_reference_cues": social_reference_cues,
            "digit_token_ratio": digit_token_ratio,
            "has_currency_or_percent": has_currency_or_percent,
        }

        # Return strictly in FEATURE_NAMES order
        return {k: features.get(k, 0.0) for k in self.feature_names}

    def extract_single(self, text: str) -> Dict[str, float]:
        """Extracts features for a single input text with error fallback."""
        if not text or not str(text).strip():
            return {k: 0.0 for k in self.feature_names}
        try:
            doc = self.nlp(str(text))
            return self.extract_from_doc(doc, str(text))
        except Exception:
            return {k: 0.0 for k in self.feature_names}

    def extract_dataframe(self, texts: List[str], batch_size: int = 128) -> pd.DataFrame:
        """Extracts features across a batch of texts using nlp.pipe."""
        clean_texts = [str(t) if t is not None else "" for t in texts]
        records: List[Dict[str, float]] = []

        for doc, raw_t in zip(self.nlp.pipe(clean_texts, batch_size=batch_size), clean_texts):
            if not raw_t.strip():
                records.append({k: 0.0 for k in self.feature_names})
            else:
                try:
                    records.append(self.extract_from_doc(doc, raw_t))
                except Exception:
                    records.append({k: 0.0 for k in self.feature_names})

        return pd.DataFrame(records)[self.feature_names]

    def fit_scaler(self, df_features: pd.DataFrame) -> StandardScaler:
        """Fits StandardScaler strictly on training features."""
        self.scaler = StandardScaler()
        self.scaler.fit(df_features[self.feature_names].values)
        return self.scaler

    def transform(self, df_features: pd.DataFrame) -> np.ndarray:
        """Scales features using the fitted scaler."""
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted. Call fit_scaler on training data first.")
        vals = df_features[self.feature_names].values
        return self.scaler.transform(vals).astype(np.float32)

    def fit_transform(self, df_features: pd.DataFrame) -> np.ndarray:
        self.fit_scaler(df_features)
        return self.transform(df_features)

    def save_scaler(self, path: str) -> None:
        if self.scaler is None:
            raise RuntimeError("Cannot save un-fitted scaler.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, path)
        print(f"[SYNTAX] Scaler and feature names saved to {path}")

    def load_scaler(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        print(f"[SYNTAX] Loaded scaler with {len(self.feature_names)} features from {path}")

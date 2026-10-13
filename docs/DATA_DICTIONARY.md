# Data Dictionary: Dark Pattern Detection Research Dataset

## Overview

This dataset dictionary documents all variables and provenance fields within the e-commerce dark pattern corpus (`dark-patterns(2).csv`), comprising **1,818 raw observations** collected from real-world e-commerce platforms.

The primary research objective is four-class classification across:
1. **Social Proof**
2. **Scarcity**
3. **Urgency**
4. **Misdirection**

---

## Column Specifications

| Column Name | Data Type | Role in Pipeline | Handling & Preprocessing Rules | Research Significance |
|---|---|---|---|---|
| `Pattern String` | `String (Text)` | **Primary Input Feature** | HTML entities unescaped, whitespace normalized, stripped. **306 rows with null or empty text are safely audited and excluded** from model training. Never replaced with synthetic data. | The surface text displayed to shoppers, containing semantic and syntactic cues indicating persuasive or deceptive design patterns. |
| `Pattern Category` | `Categorical String` | **Target Classification Label** | Normalized into 4 target classes: `Social Proof` (0), `Scarcity` (1), `Urgency` (2), `Misdirection` (3). Rows labeled `Obstruction`, `Sneaking`, and `Forced Action` are safely retained in audit records but **excluded from the 4-class experiment without relabeling**. | The high-level taxonomical pattern category established by foundational HCI dark pattern literature (Mathur et al., 2019; Gray et al., 2018). |
| `Pattern Type` | `Categorical String` | Metadata & Sub-taxonomy | Preserved intact for fine-grained error analysis and subgroup performance audits. | Specific UI mechanism implementation (e.g., *Activity Notification*, *Countdown Timer*, *Confirmshaming*, *Low-stock Message*). |
| `Comment` | `String (Text)` | Provenance & Annotator Notes | Preserved intact; missing values (20 rows) left as empty string in metadata. | Qualitative notes recorded during website crawler extraction, noting UI behaviors such as *Periodic popup*, *Hardcoded in JS*, or *Cart flyout*. |
| `Where in website?` | `Categorical String` | Contextual Provenance | Preserved intact. Values include *Product Page*, *Cart Page*, *Home Page*, *Checkout Process*, *Other Page*. | Identifies the funnel stage of the shopping flow where psychological pressure or bias exploitation occurred. |
| `Deceptive?` | `Categorical String` | Ground-truth Ethical Annotation | Preserved intact. Values: `'Yes'`, `'No'`, `'Depends'`. | Distinguishes whether the observed cue was objectively verifiable or deceptive (e.g. fabricated timers vs real inventory). |
| `Website Page` | `String (URL)` | Provenance & Audit Trail | Preserved with 100% completeness (0 missing values). Domain names extracted for cross-origin audit. | Complete URL origin enabling reproducible verification of e-commerce sources. |

---

## Preprocessing & Filtering Summary

1. **Initial Dataset Size**: 1,818 rows
2. **Missing Text Exclusions**: 306 rows (Urgency: 244, Misdirection: 33, Sneaking: 14, Social Proof: 11, Forced Action: 2, Obstruction: 1, Scarcity: 1)
3. **Usable Text Records**: 1,512 rows
4. **Excluded Non-Target Categories**: 46 rows with usable text (Obstruction: 30, Sneaking: 12, Forced Action: 4)
5. **Primary 4-Class Dataset**: **1,466 records**
   - **Scarcity**: 678 samples (46.25%)
   - **Social Proof**: 314 samples (21.42%)
   - **Misdirection**: 237 samples (16.17%)
   - **Urgency**: 237 samples (16.17%)
6. **Data Partitioning**: Stratified 70% Train (1,026), 15% Validation (220), 15% Test (220) with identical partitions across all models for fair comparison.

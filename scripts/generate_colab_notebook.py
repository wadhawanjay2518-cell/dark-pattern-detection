"""
scripts/generate_colab_notebook.py

Generates notebooks/dark_pattern_fusion_colab.ipynb:
A self-contained, reproducible Google Colab notebook for the 4-class
Dark Pattern Detection research experiment.
"""

import os
import json


def generate():
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Dark Pattern Detection\n",
                    "## Semantic–Linguistic Feature Fusion for E-Commerce Text Classification\n",
                    "**Undergraduate Research Project**\n\n",
                    "### Abstract & Research Objective\n",
                    "E-commerce user interfaces frequently employ deceptive, manipulative, or coercive design patterns (*dark patterns*) to influence consumer decision-making. While contemporary NLP systems typically employ pre-trained transformer embeddings, domain-specific dark patterns rely heavily on distinctive linguistic constructs—such as imperative commands, temporal deadlines, artificial inventory scarcity, and simulated social proof.\n\n",
                    "This research notebook investigates a **Semantic–Linguistic Feature Fusion Architecture** comparing four distinct paradigms across the four core dark pattern categories:\n",
                    "1. **Social Proof** (e.g., activity notifications, crowd viewing counts)\n",
                    "2. **Scarcity** (e.g., inventory/stock counters, limited availability claims)\n",
                    "3. **Urgency** (e.g., countdown timers, expiring offers)\n",
                    "4. **Misdirection** (e.g., asymmetric framing, comparative price anchoring)\n\n",
                    "**Comparative Models Evaluated:**\n",
                    "- **Model A (Baseline):** TF-IDF Vectorization + Multinomial Logistic Regression\n",
                    "- **Model B (Transformer-Only):** Fine-tuned DistilBERT contextual text classifier\n",
                    "- **Model C (Linguistic-Only):** spaCy syntactic & linguistic feature extraction + classifier\n",
                    "- **Model D (Proposed Late Fusion):** Contextual transformer representations fused with normalized linguistic vectors\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 1: Dependencies & Accelerated Hardware Setup\n",
                    "!pip install -q torch transformers datasets spacy scikit-learn pandas matplotlib seaborn joblib\n",
                    "!python -m spacy download en_core_web_sm\n",
                    "\n",
                    "import os\n",
                    "import re\n",
                    "import html\n",
                    "import time\n",
                    "import json\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import spacy\n",
                    "\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "from torch.optim import AdamW\n",
                    "from torch.utils.data import Dataset, DataLoader\n",
                    "from transformers import AutoTokenizer, AutoModel\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.model_selection import StratifiedShuffleSplit\n",
                    "from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report\n",
                    "\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "print(f'Accelerated compute device: {device}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 2: Dataset Loading, Validation & 4-Class Filtering\n",
                    "Please upload `dark-patterns(2).csv` to the Colab files pane or current working directory."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Locate or upload dark-patterns(2).csv\n",
                    "csv_path = 'dark-patterns(2).csv'\n",
                    "if not os.path.exists(csv_path):\n",
                    "    if os.path.exists('data/dark-patterns(2).csv'):\n",
                    "        csv_path = 'data/dark-patterns(2).csv'\n",
                    "    else:\n",
                    "        from google.colab import files\n",
                    "        print('Please select dark-patterns(2).csv to upload:')\n",
                    "        uploaded = files.upload()\n",
                    "        csv_path = list(uploaded.keys())[0]\n",
                    "\n",
                    "raw_df = pd.read_csv(csv_path)\n",
                    "print(f'Raw dataset dimensions: {raw_df.shape}')\n",
                    "print('Raw columns:', raw_df.columns.tolist())\n",
                    "print('\\nRaw Category Counts:\\n', raw_df['Pattern Category'].value_counts())\n",
                    "print('\\nMissing Pattern String count:', raw_df['Pattern String'].isnull().sum())\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "TARGET_CLASSES = ['Social Proof', 'Scarcity', 'Urgency', 'Misdirection']\n",
                    "LABEL_MAP = {i: c for i, c in enumerate(TARGET_CLASSES)}\n",
                    "REVERSE_LABEL_MAP = {c: i for i, c in enumerate(TARGET_CLASSES)}\n",
                    "\n",
                    "def sanitize_text(text):\n",
                    "    if text is None:\n",
                    "        return ''\n",
                    "    t = html.unescape(str(text))\n",
                    "    t = t.replace('\\xa0', ' ').replace('\\u200b', '')\n",
                    "    t = re.sub(r'\\s+', ' ', t).strip()\n",
                    "    return t\n",
                    "\n",
                    "# Filter rows with usable text and target classes\n",
                    "clean_mask = raw_df['Pattern String'].notnull() & (raw_df['Pattern String'].astype(str).str.strip() != '')\n",
                    "df_text = raw_df[clean_mask].copy()\n",
                    "df_text['clean_text'] = df_text['Pattern String'].apply(sanitize_text)\n",
                    "df_text['clean_category'] = df_text['Pattern Category'].astype(str).str.strip()\n",
                    "\n",
                    "# Exclude non-target classes (Obstruction, Sneaking, Forced Action) without relabeling\n",
                    "df_target = df_text[df_text['clean_category'].isin(TARGET_CLASSES)].copy()\n",
                    "df_target['label'] = df_target['clean_category'].map(REVERSE_LABEL_MAP)\n",
                    "\n",
                    "print(f'Eligible 4-class samples with usable text: {len(df_target)}')\n",
                    "print(df_target['clean_category'].value_counts())\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 3: Stratified Train / Validation / Test Partitions\n",
                    "We create stratified 70% Train, 15% Validation, and 15% Test partitions to ensure identical data distributions across all models."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "SEED = 42\n",
                    "split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)\n",
                    "train_idx, val_test_idx = next(split1.split(df_target, df_target['label']))\n",
                    "train_df = df_target.iloc[train_idx].reset_index(drop=True)\n",
                    "val_test_df = df_target.iloc[val_test_idx].reset_index(drop=True)\n",
                    "\n",
                    "split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)\n",
                    "val_idx, test_idx = next(split2.split(val_test_df, val_test_df['label']))\n",
                    "val_df = val_test_df.iloc[val_idx].reset_index(drop=True)\n",
                    "test_df = val_test_df.iloc[test_idx].reset_index(drop=True)\n",
                    "\n",
                    "print(f'Partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 4: Linguistic & Syntactic Feature Extraction (spaCy)\n",
                    "Extract POS distributions, syntactic dependency depth, imperative commands, temporal deadlines, scarcity markers, comparative pricing, and crowd cues."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "nlp = spacy.load('en_core_web_sm')\n",
                    "\n",
                    "TIME_WORDS = {'second', 'seconds', 'minute', 'minutes', 'hour', 'hours', 'day', 'days', 'today', 'now', 'soon', 'ends', 'expires', 'hurry'}\n",
                    "SCARCITY_WORDS = {'left', 'stock', 'remaining', 'only', 'sold', 'pieces', 'items', 'claimed', 'last', 'running', 'low'}\n",
                    "COMPARATIVE_WORDS = {'save', 'off', 'was', 'now', 'retail', 'original', 'discount', 'deal', 'best', 'seller'}\n",
                    "SOCIAL_WORDS = {'people', 'person', 'bought', 'purchased', 'viewed', 'others', 'customers', 'demand'}\n",
                    "\n",
                    "def extract_linguistic_features(text):\n",
                    "    doc = nlp(text)\n",
                    "    tokens = [t for t in doc]\n",
                    "    n_tok = max(1, len(tokens))\n",
                    "    pos_counts = {k: 0 for k in ['VERB', 'NOUN', 'ADJ', 'ADV', 'PRON', 'NUM', 'AUX', 'PUNCT']}\n",
                    "    for t in tokens:\n",
                    "        if t.pos_ in pos_counts:\n",
                    "            pos_counts[t.pos_] += 1\n",
                    "    pos_props = {f'pos_{k}': v / n_tok for k, v in pos_counts.items()}\n",
                    "    \n",
                    "    is_imp = 0.0\n",
                    "    root = [t for t in tokens if t.dep_ == 'ROOT']\n",
                    "    if root and root[0].pos_ == 'VERB' and root[0].tag_ in {'VB', 'VBP'}:\n",
                    "        if not any(c.dep_ in {'nsubj', 'nsubjpass'} for c in root[0].children):\n",
                    "            is_imp = 1.0\n",
                    "    \n",
                    "    words = set([t.lower_ for t in tokens] + [t.lemma_.lower() for t in tokens])\n",
                    "    time_c = float(len(words & TIME_WORDS))\n",
                    "    scarcity_c = float(len(words & SCARCITY_WORDS))\n",
                    "    comp_c = float(len(words & COMPARATIVE_WORDS))\n",
                    "    social_c = float(len(words & SOCIAL_WORDS))\n",
                    "    upper_r = float(len([c for c in text if c.isupper()]) / max(1, len([c for c in text if c.isalpha()])))\n",
                    "    excl_c = float(text.count('!'))\n",
                    "    dig_r = float(len([t for t in tokens if any(c.isdigit() for c in t.text)]) / n_tok)\n",
                    "    \n",
                    "    return {\n",
                    "        **pos_props,\n",
                    "        'is_imperative': is_imp,\n",
                    "        'time_cues': time_c,\n",
                    "        'scarcity_cues': scarcity_c,\n",
                    "        'comparative_cues': comp_c,\n",
                    "        'social_cues': social_c,\n",
                    "        'uppercase_ratio': upper_r,\n",
                    "        'exclamation_count': excl_c,\n",
                    "        'digit_ratio': dig_r,\n",
                    "        'length_char': float(len(text)),\n",
                    "    }\n",
                    "\n",
                    "print('Extracting linguistic features across partitions...')\n",
                    "train_feats = pd.DataFrame([extract_linguistic_features(t) for t in train_df['clean_text']])\n",
                    "val_feats = pd.DataFrame([extract_linguistic_features(t) for t in val_df['clean_text']])\n",
                    "test_feats = pd.DataFrame([extract_linguistic_features(t) for t in test_df['clean_text']])\n",
                    "\n",
                    "scaler = StandardScaler()\n",
                    "X_train_syntax = scaler.fit_transform(train_feats)\n",
                    "X_val_syntax = scaler.transform(val_feats)\n",
                    "X_test_syntax = scaler.transform(test_feats)\n",
                    "print(f'Linguistic feature vector dimension: {X_train_syntax.shape[1]}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 5: Model A — TF-IDF + Logistic Regression Baseline"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n",
                    "X_train_tfidf = vec.fit_transform(train_df['clean_text'])\n",
                    "X_test_tfidf = vec.transform(test_df['clean_text'])\n",
                    "\n",
                    "clf_a = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)\n",
                    "clf_a.fit(X_train_tfidf, train_df['label'])\n",
                    "preds_a = clf_a.predict(X_test_tfidf)\n",
                    "acc_a = accuracy_score(test_df['label'], preds_a)\n",
                    "p_a, r_a, f1_a, _ = precision_recall_fscore_support(test_df['label'], preds_a, average='macro', zero_division=0)\n",
                    "print(f'Model A — Accuracy: {acc_a:.4f}, Macro-F1: {f1_a:.4f}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 6: Model C — Linguistic-Only Classifier"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "clf_c = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)\n",
                    "clf_c.fit(X_train_syntax, train_df['label'])\n",
                    "preds_c = clf_c.predict(X_test_syntax)\n",
                    "acc_c = accuracy_score(test_df['label'], preds_c)\n",
                    "p_c, r_c, f1_c, _ = precision_recall_fscore_support(test_df['label'], preds_c, average='macro', zero_division=0)\n",
                    "print(f'Model C — Accuracy: {acc_c:.4f}, Macro-F1: {f1_c:.4f}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 7: PyTorch Dataset & DistilBERT Tokenization"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')\n",
                    "\n",
                    "class TextDataset(Dataset):\n",
                    "    def __init__(self, texts, syntax_feats, labels, max_len=64):\n",
                    "        self.encodings = tokenizer(list(texts), padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')\n",
                    "        self.syntax = torch.tensor(syntax_feats, dtype=torch.float32)\n",
                    "        self.labels = torch.tensor(labels, dtype=torch.long)\n",
                    "    def __len__(self):\n",
                    "        return len(self.labels)\n",
                    "    def __getitem__(self, idx):\n",
                    "        return {\n",
                    "            'input_ids': self.encodings['input_ids'][idx],\n",
                    "            'attention_mask': self.encodings['attention_mask'][idx],\n",
                    "            'syntax': self.syntax[idx],\n",
                    "            'label': self.labels[idx]\n",
                    "        }\n",
                    "\n",
                    "train_ds = TextDataset(train_df['clean_text'], X_train_syntax, train_df['label'].values)\n",
                    "val_ds = TextDataset(val_df['clean_text'], X_val_syntax, val_df['label'].values)\n",
                    "test_ds = TextDataset(test_df['clean_text'], X_test_syntax, test_df['label'].values)\n",
                    "\n",
                    "train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)\n",
                    "val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)\n",
                    "test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 8: Model B (DistilBERT-Only) & Model D (Semantic-Linguistic Fusion Architecture)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "class DarkPatternFusionModel(nn.Module):\n",
                    "    def __init__(self, syntax_dim, num_classes=4, use_syntax=True):\n",
                    "        super().__init__()\n",
                    "        self.use_syntax = use_syntax\n",
                    "        self.transformer = AutoModel.from_pretrained('distilbert-base-uncased')\n",
                    "        hidden_dim = self.transformer.config.hidden_size\n",
                    "        \n",
                    "        if self.use_syntax:\n",
                    "            self.syntax_proj = nn.Sequential(\n",
                    "                nn.Linear(syntax_dim, 64),\n",
                    "                nn.LayerNorm(64),\n",
                    "                nn.GELU(),\n",
                    "                nn.Dropout(0.2)\n",
                    "            )\n",
                    "            fused_dim = hidden_dim + 64\n",
                    "        else:\n",
                    "            self.syntax_proj = None\n",
                    "            fused_dim = hidden_dim\n",
                    "            \n",
                    "        self.classifier = nn.Sequential(\n",
                    "            nn.Linear(fused_dim, 128),\n",
                    "            nn.LayerNorm(128),\n",
                    "            nn.GELU(),\n",
                    "            nn.Dropout(0.2),\n",
                    "            nn.Linear(128, num_classes)\n",
                    "        )\n",
                    "        \n",
                    "    def forward(self, input_ids, attention_mask, syntax_feats=None):\n",
                    "        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)\n",
                    "        cls_vec = out.last_hidden_state[:, 0, :]\n",
                    "        if self.use_syntax and syntax_feats is not None:\n",
                    "            syn_proj = self.syntax_proj(syntax_feats)\n",
                    "            rep = torch.cat([cls_vec, syn_proj], dim=-1)\n",
                    "        else:\n",
                    "            rep = cls_vec\n",
                    "        return self.classifier(rep)\n",
                    "\n",
                    "def train_epoch(model, loader, opt, crit):\n",
                    "    model.train()\n",
                    "    total_loss = 0.0\n",
                    "    for b in loader:\n",
                    "        opt.zero_grad()\n",
                    "        ids, mask, syn, y = b['input_ids'].to(device), b['attention_mask'].to(device), b['syntax'].to(device), b['label'].to(device)\n",
                    "        logits = model(ids, mask, syn)\n",
                    "        loss = crit(logits, y)\n",
                    "        loss.backward()\n",
                    "        opt.step()\n",
                    "        total_loss += loss.item()\n",
                    "    return total_loss / len(loader)\n",
                    "\n",
                    "def evaluate_model(model, loader):\n",
                    "    model.eval()\n",
                    "    preds, trues = [], []\n",
                    "    with torch.no_grad():\n",
                    "        for b in loader:\n",
                    "            ids, mask, syn, y = b['input_ids'].to(device), b['attention_mask'].to(device), b['syntax'].to(device), b['label'].to(device)\n",
                    "            logits = model(ids, mask, syn)\n",
                    "            p = torch.argmax(logits, dim=-1)\n",
                    "            preds.extend(p.cpu().numpy().tolist())\n",
                    "            trues.extend(y.cpu().numpy().tolist())\n",
                    "    acc = accuracy_score(trues, preds)\n",
                    "    _, _, f1, _ = precision_recall_fscore_support(trues, preds, average='macro', zero_division=0)\n",
                    "    return acc, f1, np.array(preds), np.array(trues)\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train Model B: DistilBERT-Only\n",
                    "print('--- Training Model B (DistilBERT-Only) ---')\n",
                    "model_b = DarkPatternFusionModel(syntax_dim=X_train_syntax.shape[1], use_syntax=False).to(device)\n",
                    "opt_b = AdamW(model_b.parameters(), lr=2e-5, weight_decay=0.01)\n",
                    "crit = nn.CrossEntropyLoss()\n",
                    "\n",
                    "for ep in range(3):\n",
                    "    loss = train_epoch(model_b, train_loader, opt_b, crit)\n",
                    "    v_acc, v_f1, _, _ = evaluate_model(model_b, val_loader)\n",
                    "    print(f'Epoch {ep+1}/3 - Loss: {loss:.4f} | Val Acc: {v_acc:.4f} | Val Macro-F1: {v_f1:.4f}')\n",
                    "\n",
                    "acc_b, f1_b, preds_b, y_test = evaluate_model(model_b, test_loader)\n",
                    "print(f'Model B Test — Accuracy: {acc_b:.4f}, Macro-F1: {f1_b:.4f}')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train Model D: Late Semantic-Linguistic Fusion\n",
                    "print('--- Training Model D (Semantic-Linguistic Fusion) ---')\n",
                    "model_d = DarkPatternFusionModel(syntax_dim=X_train_syntax.shape[1], use_syntax=True).to(device)\n",
                    "opt_d = AdamW([\n",
                    "    {'params': model_d.transformer.parameters(), 'lr': 2e-5},\n",
                    "    {'params': list(model_d.syntax_proj.parameters()) + list(model_d.classifier.parameters()), 'lr': 1e-3}\n",
                    "], weight_decay=0.01)\n",
                    "\n",
                    "for ep in range(3):\n",
                    "    loss = train_epoch(model_d, train_loader, opt_d, crit)\n",
                    "    v_acc, v_f1, _, _ = evaluate_model(model_d, val_loader)\n",
                    "    print(f'Epoch {ep+1}/3 - Loss: {loss:.4f} | Val Acc: {v_acc:.4f} | Val Macro-F1: {v_f1:.4f}')\n",
                    "\n",
                    "acc_d, f1_d, preds_d, _ = evaluate_model(model_d, test_loader)\n",
                    "print(f'Model D Test — Accuracy: {acc_d:.4f}, Macro-F1: {f1_d:.4f}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Step 9: Comparative Research Results & Visualization"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "comparison_df = pd.DataFrame([\n",
                    "    {'Model': 'Model A: TF-IDF + LogReg', 'Accuracy': acc_a, 'Macro-F1': f1_a},\n",
                    "    {'Model': 'Model C: Linguistic-Only', 'Accuracy': acc_c, 'Macro-F1': f1_c},\n",
                    "    {'Model': 'Model B: DistilBERT-Only', 'Accuracy': acc_b, 'Macro-F1': f1_b},\n",
                    "    {'Model': 'Model D: Semantic-Linguistic Fusion', 'Accuracy': acc_d, 'Macro-F1': f1_d},\n",
                    "])\n",
                    "print(comparison_df.to_string(index=False))\n",
                    "\n",
                    "plt.figure(figsize=(9, 5))\n",
                    "x = np.arange(len(comparison_df))\n",
                    "width = 0.35\n",
                    "plt.bar(x - width/2, comparison_df['Macro-F1'], width, label='Macro F1', color='#6366F1')\n",
                    "plt.bar(x + width/2, comparison_df['Accuracy'], width, label='Accuracy', color='#A5B4FC', edgecolor='#4F46E5')\n",
                    "plt.xticks(x, comparison_df['Model'], rotation=15, ha='right')\n",
                    "plt.ylabel('Score (0.0 - 1.0)')\n",
                    "plt.title('4-Class Model Comparison: Linguistic vs Semantic vs Fusion')\n",
                    "plt.ylim([0, 1.05])\n",
                    "plt.legend()\n",
                    "plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "cm_d = confusion_matrix(y_test, preds_d)\n",
                    "plt.figure(figsize=(6, 5))\n",
                    "sns.heatmap(cm_d, annot=True, fmt='d', cmap='Purples', xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)\n",
                    "plt.title('Model D (Proposed Fusion) Test Confusion Matrix')\n",
                    "plt.ylabel('True Category')\n",
                    "plt.xlabel('Predicted Category')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print('\\nClassification Report (Model D):\\n')\n",
                    "print(classification_report(y_test, preds_d, target_names=TARGET_CLASSES, digits=4))\n"
                ]
            }
        ]
    }

    os.makedirs("notebooks", exist_ok=True)
    out_path = "notebooks/dark_pattern_fusion_colab.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Colab notebook successfully written to {out_path}")


if __name__ == "__main__":
    generate()

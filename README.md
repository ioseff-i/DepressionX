# DEPRESSIONX: Knowledge-Infused Residual Attention for Explainable Depression Severity Assessment

**Accepted for**: THE 9th INTERNATIONAL WORKSHOP ON ​HEALTH INTELLIGENCE (W3PHIAI-25), March 04, 2025, Philadelphia, PA, USA, in conjunction with AAAI 2025.

---

## Overview

DEPRESSIONX is a domain knowledge-infused residual attention model designed for explainable depression severity detection using social media data. Unlike traditional deep learning models, DEPRESSIONX focuses on interpretability and achieves state-of-the-art performance in depression severity classification while providing insights into its decision-making process.

---

### Features

1. **Explainable AI**: Combines textual and knowledge graph-based features to deliver interpretable predictions.
2. **Multi-Level Embeddings**: Combines word, sentence, and post-level embeddings for contextual analysis.
3. **Ordinal Regression**: Supports classification of depression severity into clinical categories (minimal, mild, moderate, severe).
4. **Knowledge Graph Integration**: Leverages domain knowledge to contextualize and enhance predictions.

---

## Datasets

The model has been evaluated on:
- **D1**: A slightly imbalanced dataset derived from the Dreaddit dataset.
- **D2**: A balanced dataset from Reddit subreddits focused on mental health.

Both datasets are categorized into four depression severity levels:
- **Minimal**
- **Mild**
- **Moderate**
- **Severe**

---

## Architecture

### DEPRESSIONX Model Components:
1. **Textual Representation**: 
   - Encodes social media posts at word, sentence, and post levels using **FastText** and **SentenceTransformer**.
   - Residual multi-head attention enhances contextual understanding.

2. **Knowledge Graph Representation**:
   - Constructs a domain-specific depression knowledge graph using Wikipedia and REBEL.
   - Applies **Graph Isomorphism Network (GIN)** and **Graph Attention Network (GAT)** layers for relational reasoning.

3. **Explainability**:
   - Generates attention-based explanations for textual inputs.
   - Extracts a subgraph highlighting influential knowledge graph relations.

![Model Diagram](DepressionX.pdf)

---

## Results

Performance metrics on benchmark datasets:

| Dataset | Precision | Recall | F1 Score | 
|---------|-----------|--------|----------|
| **D1**  | 97.4%    | 71.8%  | 82.5%    | 
| **D2**  | 91.1%    | 90.9%  | 90.9%    |

---

## Citation

If you use this work, please cite our paper (Just draft, not yet ready!):

```
@inproceedings{ibrahimov2025depressionx,
               title={DEPRESSIONX: Knowledge Infused Residual Attention for Explainable Depression Severity Assessment},
               author={Ibrahimov, Yusif and Anwar, Tarique and Yuan, Tommy},
               booktitle={Proceedings of W3PHIAI-25}, year={2025}
}
```
---

## Contact

For questions or collaborations, please contact:
- Yusif Ibrahimov (yusif.ibrahimov@york.ac.uk)
- Tarique Anwar (tarique.anwar@york.ac.uk)
- Tommy Yuan (tommy.yuan@york.ac.uk)


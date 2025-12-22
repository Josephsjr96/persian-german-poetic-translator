# Persian ↔ German Poetic Translator - Research Platform

A **research-grade desktop application** for bidirectional translation and in-depth analysis of Persian and German literary/poetic text using **NLLB-200** (Meta 2022) and **M2M100** (Facebook 2020).

This platform enables systematic evaluation of machine translation quality on poetic text through automatic metrics, poetic feature extraction, structured human evaluation, persistent data storage, research collaboration tools, and an analytics dashboard.

Developed by **Yousef Sanjari** as a major portfolio project for the **M.Sc. in Natural Language Processing** at Universität Trier.

![Main Interface](screenshots/main_interface.png)  
*Translation & Analysis tab with side-by-side outputs and metrics pop-up.*

## Key Features

- **Bidirectional translation** (Persian ↔ German)
- **Side-by-side model comparison** with real-time output
- **Automatic evaluation metrics**:
  - BLEU score
  - Semantic similarity
  - Word count and compression analysis
- **Poetic feature extraction** (rhyme, meter, metaphor detection)
- **Structured human evaluation**:
  - Ratings for fluency, poetic preservation, cultural accuracy
  - Model preference and detailed comments
- **Multi-tab research interface**:
  - Translation & Analysis
  - Human Evaluation
  - Research Collaboration (with project management and discussion)
  - Analytics & Export (model performance, human ratings, data export)
- **Persistent SQLite database** for translations, metrics, and human judgments
- **Data export** (JSON, CSV) and research report generation
- **Modern dark UI** with ttkbootstrap

## Research Value

Poetry translation remains one of the most challenging tasks in NLP due to metaphor, rhythm, cultural nuance, and stylistic fidelity. This platform addresses core research questions:
- How well do SOTA models preserve poetic structure and meaning?
- Do automatic metrics correlate with human judgment of poetic quality?
- What are the limitations of older vs newer multilingual models on literary text?

The collected data (human judgments, poetic features) forms the foundation for future benchmarks, fine-tuning, or publication — especially valuable for low-resource languages like Persian.



<img width="1602" height="1039" alt="Screenshot 2025-12-19 225620" src="https://github.com/user-attachments/assets/0d0cc0a7-5c3e-46b6-a430-876813f577ea" />
<img width="1602" height="1039" alt="Screenshot 2025-12-19 225640" src="https://github.com/user-attachments/assets/580db854-5197-40d1-9c2e-8313a1f5d80c" />
<img width="1602" height="1039" alt="Screenshot 2025-12-19 225656" src="https://github.com/user-attachments/assets/f04746c9-cad8-4fe5-aba5-006e89b3bd13" />
<img width="1602" height="1039" alt="Screenshot 2025-12-19 225604" src="https://github.com/user-attachments/assets/e0c9bca3-c5c6-4b70-a010-d3b417d8cf40" />

## Requirements

```bash
pip install torch transformers sentence-transformers ttkbootstrap nltk syllables

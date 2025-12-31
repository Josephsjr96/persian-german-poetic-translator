# Persian ↔ German Poetic Translator - Research Platform

A **research-grade desktop application** for bidirectional translation and in-depth analysis of Persian and German literary/poetic text using **NLLB-200** and **M2M100**.

This platform enables systematic evaluation of machine translation quality on poetic text through automatic metrics, poetic feature extraction, structured human evaluation, persistent data storage, research collaboration tools, and an analytics dashboard.
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
- How well do state-of-the-art models preserve poetic structure and meaning?
- Do automatic metrics correlate with human judgment of poetic quality?
- What are the limitations of different multilingual architectures on literary text?

The collected data (human judgments, poetic features) forms the foundation for future benchmarks and fine-tuning. The project serves as a practical framework for investigating the intersection of computational linguistics, literary studies, and machine learning, especially for low-resource and stylistically complex languages like Persian.

## Future Vision & Research Goals

This project is more than a tool; it's a foundation for a broader research program at the intersection of NLP and the humanities.

### Short-Term Objectives (Next 6-12 Months)
- **Model Expansion**: Integrate the latest large language models (LLMs) to compare architectural approaches.
- **Specialized Corpora**: Build and release the first open-source, parallel corpus of Persian-German poetry for benchmarking literary translation.
- **Novel Metrics**: Develop evaluation metrics (e.g., "PoeticBLEU") that better correlate with human judgments of literary quality.
- **Multilingual Extension**: Expand the framework to other linguistically diverse language pairs.

### Medium-Term Aspirations (1-3 Years)
- **Universal Framework**: Develop the platform into a domain-adaptive system that can be customized for any language pair's poetic traditions.
- **Knowledge-Augmented Translation**: Implement structured knowledge graphs to model poetic devices (metaphor, allusion, cultural references) that current models often miss.
- **Interactive Translation Studio**: Evolve the tool for professional translators, offering AI-suggested variants that preserve authorial voice and stylistic intent.
- **Computational Poetics Research**: Use the platform to conduct large-scale, data-driven studies on how poetic forms transform across languages and cultures.

### Long-Term Impact (3-5+ Years)
- **Democratizing Literary Access**: Help make world poetry more accessible across language barriers while preserving its artistic essence, not just its literal meaning.
- **Preserving Cultural Heritage**: Apply the framework to document and translate the oral and written poetic traditions of minority and endangered languages.
- **Advancing Explainable AI**: Contribute to making complex AI models more interpretable by forcing them to articulate their "reasoning" about subjective qualities like beauty and emotional resonance.

## What This Project Represents

This work tackles a fundamental question: **Can computational systems learn to understand and preserve what makes poetry human?** By quantifying subtle artistic features, we aim to build bridges not just between languages, but between computational precision and humanistic subtlety. The ultimate goal is not to replace human translators but to create **augmented intelligence** tools that deepen cross-cultural understanding and empower human creativity.


<img width="1602" height="1039" alt="Screenshot 2025-12-19 225604" src="https://github.com/user-attachments/assets/e0c9bca3-c5c6-4b70-a010-d3b417d8cf40" />
<img width="1602" height="1039" alt="Screenshot 2025-12-19 225604" src="https://github.com/user-attachments/assets/e0c9bca3-c5c6-4b70-a010-d3b417d8cf40" />
<img width="1602" height="1039" alt="Screenshot 2025-12-19 225656" src="https://github.com/user-attachments/assets/f04746c9-cad8-4fe5-aba5-006e89b3bd13" />
<img width="1602" height="1039" alt="Screenshot 2025-12-19 225620" src="https://github.com/user-attachments/assets/0d0cc0a7-5c3e-46b6-a430-876813f577ea" />
<img width="1602" height="1039" alt="Screenshot 2025-12-19 225640" src="https://github.com/user-attachments/assets/580db854-5197-40d1-9c2e-8313a1f5d80c" />

## Requirements

```bash
pip install torch transformers sentence-transformers ttkbootstrap nltk syllables


torch – PyTorch (for model inference; CPU version is sufficient)
transformers – Hugging Face Transformers library (NLLB-200 and M2M100)
sentence-transformers – For semantic similarity evaluation
ttkbootstrap – Modern themed GUI (dark "superhero" theme)
nltk – For BLEU score calculation (downloads 'punkt' and 'punkt_tab' automatically)
syllables – For poetic meter and syllable analysis

Hardware Requirements

RAM – 16 GB recommended (8 GB minimum – large models load into memory)
Disk Space – Approximately 6 GB free on first run
(Models are downloaded and cached: NLLB-200 ~2.5 GB, M2M100 ~2 GB, plus safetensors conversion, embeddings, and SQLite database)
CPU – Modern multi-core processor (e.g., Ryzen 5 or Intel i5/i7) – no GPU required
(First run loading may take 5–15 minutes on CPU; subsequent runs: seconds)

Internet Connection (First Run Only)

Required for downloading models (~5–6 GB total) and small NLTK data
After first run, the application works completely offline

Additional Notes

On first launch, the app downloads and converts models — be patient.
Models are cached locally (in ~/.cache/huggingface/ or C:\Users\<YourName>\.cache\huggingface\ on Windows).
No API keys, registration, or external services required.
Database (poetic_translator_research.db) is created automatically in the project folder.

With these requirements met, the application will run smoothly and provide high-quality translations, poetic analysis, human evaluation, and research analytics.
Enjoy exploring Persian and German poetry with cutting-edge NLP tools! 🎓

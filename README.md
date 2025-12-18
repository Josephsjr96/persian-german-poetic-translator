Persian ↔ German Poetic Translator
A desktop application for translating Persian (Farsi) poetry to German (and vice versa) using two state-of-the-art multilingual models: NLLB-200 (Meta 2022) and M2M100 (Facebook 2020). The app provides side-by-side translations, automatic evaluation with BLEU score and semantic similarity, a bidirectional switch, and results in an elegant pop-up window.
This project was developed by Yousef Sanjari as part of a portfolio for the M.Sc. in Natural Language Processing at Universität Trier, demonstrating hands-on skills in multilingual machine translation, model comparison, evaluation metrics, and user interface design.









<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ec082473-a72b-4d0a-a712-77c54191e517" />

Side-by-side translation of a Persian poem with comparison results in a pop-up window.
Features

Bidirectional translation (Persian → German or German → Persian)
Side-by-side comparison of NLLB-200 (recommended) and M2M100
Automatic evaluation:
BLEU score (surface-level wording overlap)
Semantic similarity (meaning and poetic feel)

Beautiful pop-up window for comparison results
Modern dark theme with ttkbootstrap ("superhero")
Fully offline after first run (models cached locally)
No GPU required (runs on CPU)

Requirements

Python 3.10+
Libraries:Bashpip install torch transformers sentence-transformers ttkbootstrap nltk
Disk space: ~5-6 GB on first run (large models downloaded and cached)
RAM: 16 GB recommended

How to Run

Clone the repository:Bashgit clone https://github.com/Josephsjr96/persian-german-poetic-translator.git
cd persian-german-poetic-translator
Install dependencies:Bashpip install torch transformers sentence-transformers ttkbootstrap nltk
Run the app:Bashpython persian_german_translator.py
On first run: Models download and convert (~5 GB, may take time).
Subsequent runs: Fast startup.
Paste a poem, choose direction if desired, and click "Translate & Compare Models".
Results appear in output boxes; comparison details in a pop-up window.

Tip: Try verses from Hafez, Saadi, or Rumi for the best demonstration of poetic handling.
Technology Stack

Python
Hugging Face Transformers (NLLB-200, M2M100)
Sentence-Transformers (semantic similarity)
Tkinter + ttkbootstrap (GUI)
NLTK (BLEU score)

Future Potential
This application has strong potential to evolve into a valuable tool for researchers, students, and enthusiasts in Natural Language Processing and literary studies:

Human evaluation mode: Add rating sliders to collect judgments on fluency, fidelity, and poetic quality → build a Persian-German literary MT benchmark.
Post-editing interface: Allow users to correct translations → crowdsource high-quality parallel data for fine-tuning.
Style control: Experiment with prompts or parameters for "more formal" or "more poetic" outputs.
Educational tool: Add explanations of tokenization, beam search, and metric limitations.
Expansion: Support Arabic, Urdu, or Turkish → German (poetically rich, low-resource pairs).

With community contributions, it could become a reference platform for studying machine translation of poetry — an area where current systems still face significant challenges.

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import tkinter as tk
from tkinter import scrolledtext, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import threading
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class PersianGermanPoetTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Persian ↔ German Poetic Translator")
        self.root.geometry("1500x950")

        self.nllb_tokenizer = None
        self.nllb_model = None
        self.m2m_tokenizer = None
        self.m2m_model = None
        self.similarity_model = None

        self.direction_var = tk.BooleanVar(value=False)

        self.setup_ui()
        self.load_models_async()

    def setup_ui(self):
        # Header
        header = ttk.Frame(self.root, padding=30)
        header.pack(fill=X)

        title = ttk.Label(header, text="Persian ↔ German Poetic Translator", font=("Helvetica", 30, "bold"), foreground="#00ffff")
        title.pack()

        subtitle = ttk.Label(header, text="Compare NLLB-200 and M2M100 on literary translation quality", font=("Helvetica", 14), foreground="#cccccc")
        subtitle.pack(pady=(8, 15))

        # Direction toggle
        dir_frame = ttk.Frame(header)
        dir_frame.pack(pady=10)

        ttk.Label(dir_frame, text="Direction:", font=("Helvetica", 13, "bold")).pack(side=LEFT, padx=(0, 15))
        toggle = ttk.Checkbutton(dir_frame, text="German → Persian", variable=self.direction_var, bootstyle="success-round-toggle", command=self.update_input_label)
        toggle.pack(side=LEFT)

        # Status
        self.status_label = ttk.Label(self.root, text="Loading models...", font=("Helvetica", 12), foreground="#ffcc00")
        self.status_label.pack(pady=15)

        # Input
        input_container = ttk.Frame(self.root, padding=20)
        input_container.pack(fill=BOTH, expand=True, pady=(0, 20))

        self.input_frame = ttk.Labelframe(input_container, text=" Input: Persian Poem (فارسی) ", bootstyle="primary", padding=20)
        self.input_frame.pack(fill=BOTH, expand=True)

        self.input_text = scrolledtext.ScrolledText(self.input_frame, height=9, font=("Segoe UI", 13), wrap=tk.WORD)
        self.input_text.pack(fill=BOTH, expand=True)

        # Translate button
        btn_container = ttk.Frame(self.root)
        btn_container.pack(pady=20)
        self.translate_btn = ttk.Button(btn_container, text="Translate & Compare Models", bootstyle="success", width=35, command=self.start_translation, state=DISABLED)
        self.translate_btn.pack()

        # Results
        results_container = ttk.Frame(self.root, padding=20)
        results_container.pack(fill=BOTH, expand=True)

        results_grid = ttk.Frame(results_container)
        results_grid.pack(fill=BOTH, expand=True)

        # NLLB
        left = ttk.Labelframe(results_grid, text=" NLLB-200 (Meta 2022 – Recommended) ", bootstyle="info", padding=20)
        left.grid(row=0, column=0, padx=(0, 15), pady=10, sticky="nsew")
        self.nllb_output = scrolledtext.ScrolledText(left, height=12, font=("Segoe UI", 12), wrap=tk.WORD)
        self.nllb_output.pack(fill=BOTH, expand=True)

        # M2M100
        right = ttk.Labelframe(results_grid, text=" M2M100 (Facebook 2020) ", bootstyle="warning", padding=20)
        right.grid(row=0, column=1, padx=(15, 0), pady=10, sticky="nsew")
        self.m2m_output = scrolledtext.ScrolledText(right, height=12, font=("Segoe UI", 12), wrap=tk.WORD)
        self.m2m_output.pack(fill=BOTH, expand=True)

        results_grid.grid_columnconfigure((0, 1), weight=1)

    def update_input_label(self):
        if self.direction_var.get():
            self.input_frame.config(text=" Input: German Text ")
        else:
            self.input_frame.config(text=" Input: Persian Poem (فارسی) ")

    def load_models_async(self):
        def load():
            try:
                self.status_label.config(text="Loading NLLB-200...")
                self.root.update()
                self.nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
                self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

                self.status_label.config(text="Loading M2M100...")
                self.root.update()
                self.m2m_tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
                self.m2m_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")

                self.status_label.config(text="Loading similarity model...")
                self.root.update()
                self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

                self.status_label.config(text="All models loaded! Ready to translate.", foreground="#00ff00")
                self.translate_btn.config(state=NORMAL)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Model loading failed: {str(exc)}"))

        threading.Thread(target=load, daemon=True).start()

    def translate_nllb(self, text, reverse):
        if reverse:
            self.nllb_tokenizer.src_lang = "deu_Latn"
            forced_id = self.nllb_tokenizer.convert_tokens_to_ids("pes_Arab")
        else:
            self.nllb_tokenizer.src_lang = "pes_Arab"
            forced_id = self.nllb_tokenizer.convert_tokens_to_ids("deu_Latn")

        inputs = self.nllb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        generated = self.nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_id,
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        return self.nllb_tokenizer.decode(generated[0], skip_special_tokens=True)

    def translate_m2m(self, text, reverse):
        if reverse:
            self.m2m_tokenizer.src_lang = "de"
            forced_id = self.m2m_tokenizer.get_lang_id("fa")
        else:
            self.m2m_tokenizer.src_lang = "fa"
            forced_id = self.m2m_tokenizer.get_lang_id("de")

        encoded = self.m2m_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        generated = self.m2m_model.generate(
            **encoded,
            forced_bos_token_id=forced_id,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        return self.m2m_tokenizer.decode(generated[0], skip_special_tokens=True)

    def show_comparison_popup(self, direction, semantic, bleu):
        popup = tk.Toplevel(self.root)
        popup.title("Model Comparison Results")
        popup.geometry("700x450")
        popup.configure(bg="#121212")
        popup.transient(self.root)
        popup.grab_set()

        # Center on main window
        popup.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (popup.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")

        # Green success frame
        frame = ttk.Frame(popup, bootstyle="success", padding=30)
        frame.pack(fill=BOTH, expand=True)

        title = ttk.Label(frame, text=f"MODEL COMPARISON RESULTS ({direction})", font=("Helvetica", 16, "bold"), foreground="white")
        title.pack(pady=(0, 20))

        text = (
            f"Semantic Similarity: {semantic if isinstance(semantic, str) else f'{semantic:.1f}%'}\n\n"
            f"BLEU Score (M2M100 vs NLLB-200): {bleu:.1f}\n\n"
            "• NLLB-200 is generally more fluent and contextually accurate.\n"
            "• High semantic similarity + low BLEU = different wording but preserved poetic spirit.\n"
            "• This demonstrates key challenges in literary machine translation evaluation."
        )
        result_label = ttk.Label(frame, text=text, font=("Helvetica", 13), foreground="white", justify=LEFT, wraplength=600)
        result_label.pack(pady=20)

        close_btn = ttk.Button(frame, text="Close", bootstyle="light", command=popup.destroy)
        close_btn.pack(pady=10)

    def start_translation(self):
        poem = self.input_text.get("1.0", tk.END).strip()
        if len(poem) < 10:
            messagebox.showwarning("Input needed", "Please enter at least 10 characters.")
            return

        reverse = self.direction_var.get()
        self.translate_btn.config(state=DISABLED)
        self.status_label.config(text="Translating...", foreground="#ffaa00")
        self.nllb_output.delete("1.0", tk.END)
        self.m2m_output.delete("1.0", tk.END)

        def run():
            try:
                nllb_result = self.translate_nllb(poem, reverse)
                m2m_result = self.translate_m2m(poem, reverse)

                self.root.after(0, lambda: self.nllb_output.insert(tk.END, nllb_result))
                self.root.after(0, lambda: self.m2m_output.insert(tk.END, m2m_result))

                ref_tokens = [nltk.word_tokenize(nllb_result.lower())]
                cand_tokens = nltk.word_tokenize(m2m_result.lower())
                bleu = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=SmoothingFunction().method1) * 100

                semantic = "N/A (embedding model not ready)"
                if self.similarity_model:
                    try:
                        emb1 = self.similarity_model.encode(nllb_result)
                        emb2 = self.similarity_model.encode(m2m_result)
                        semantic = util.cos_sim(emb1, emb2).item() * 100
                    except:
                        semantic = "N/A (error)"

                direction = "German → Persian" if reverse else "Persian → German"
                self.root.after(0, lambda: self.show_comparison_popup(direction, semantic, bleu))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Translation failed: {str(exc)}"))
            finally:
                self.root.after(0, lambda: self.translate_btn.config(state=NORMAL))
                self.root.after(0, lambda: self.status_label.config(text="Translation complete.", foreground="#00ff00"))

        threading.Thread(target=run, daemon=True).start()

    # translate_nllb and translate_m2m remain the same as previous

    def translate_nllb(self, text, reverse):
        if reverse:
            self.nllb_tokenizer.src_lang = "deu_Latn"
            forced_id = self.nllb_tokenizer.convert_tokens_to_ids("pes_Arab")
        else:
            self.nllb_tokenizer.src_lang = "pes_Arab"
            forced_id = self.nllb_tokenizer.convert_tokens_to_ids("deu_Latn")

        inputs = self.nllb_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        generated = self.nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_id,
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        return self.nllb_tokenizer.decode(generated[0], skip_special_tokens=True)

    def translate_m2m(self, text, reverse):
        if reverse:
            self.m2m_tokenizer.src_lang = "de"
            forced_id = self.m2m_tokenizer.get_lang_id("fa")
        else:
            self.m2m_tokenizer.src_lang = "fa"
            forced_id = self.m2m_tokenizer.get_lang_id("de")

        encoded = self.m2m_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        generated = self.m2m_model.generate(
            **encoded,
            forced_bos_token_id=forced_id,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        return self.m2m_tokenizer.decode(generated[0], skip_special_tokens=True)

if __name__ == "__main__":
    app = ttk.Window(themename="superhero")
    translator = PersianGermanPoetTranslator(app)
    app.mainloop()
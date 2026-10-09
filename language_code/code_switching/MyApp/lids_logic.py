import os
import re
import gc
import json
import torch
import requests
from collections import Counter
from openai import OpenAI
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

# --- Configuration ---
CACHE_DIR = r'E:\huggingface_cache'

# User Defined Constants
GEMINI_API_KEY = ""
GROQ_API_KEY = ""

AVAILABLE_MODELS = [
    {"value": "llama-3.3-70b-versatile", "label": "🦙 Llama 3.3 70B ( Most reliable - works)"},
    {"value": "gpt-oss-120b", "label": "GPT OSS 120B (Try it - may need approval)"},
    {"value": "gpt-oss-20b", "label": " GPT OSS 20B (Faster, may need approval)"},
    {"value": "llama-4-scout-17b-16e-instruct", "label": "🦙 Llama 4 Scout (Latest - may need Dev plan)"},
]

class MaerifaTranslator:
    def __init__(self, cache_dir, device_id=-1):
        self.device = "cuda" if (device_id >= 0 and torch.cuda.is_available()) else "cpu"
        
        # NLLB for standard MSA/FR/EN
        self.nllb_name = "facebook/nllb-200-distilled-600M"
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(self.nllb_name, cache_dir=cache_dir)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(self.nllb_name, cache_dir=cache_dir).to(self.device)

        # Gemma-2-2b-it for code-switching and Darija
        self.gemma_id = "google/gemma-2-2b-it"
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(self.gemma_id)
        self.gemma_model = AutoModelForCausalLM.from_pretrained(
            self.gemma_id,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=cache_dir
        )

        # Groq Client for Llama models
        self.groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY
        )

        # Optimized Algerian Mapping
        self.darija_map = {
            "rani": "I am", "rak": "you are", "rahi": "she is",
            "bzaf": "very", "wach": "what", "dir": "do", "chwiya": "a little",
            "bash": "in order to", "lazem": "must", "waqt": "time"
        }

    def _preprocess_darija(self, text):
        """Normalizes Arabizi and applies local dialect mapping."""
        text = text.lower()
        # Arabizi normalization
        text = text.replace("3", "a").replace("7", "h").replace("9", "q").replace("5", "kh")
        return " ".join([self.darija_map.get(w, w) for w in text.split()])

    def translate(self, text, target_lang_code="ara_Arab", model_choice="llama-3.3-70b-versatile"):
        target_map = {
            "eng_Latn": "Natural English",
            "fra_Latn": "French (Français)",
            "ara_Arab": "Modern Standard Arabic",
            "ary_Arab": "Algerian Darija (Constantinois/Algérois)"
        }
        target_lang = target_map.get(target_lang_code, "English")
        
        # Strictly anchored system prompt for Algerian Dialect
        system_prompt = f"""
        You are Maerifa Engine V9.6. Convert this mixed-language text into {target_lang}.
        STRICT RULES:
        1. For Algerian Darija: Use NATURAL phrasing (e.g., 'khouya', 'mlih'). No Egyptian words.
        2. Keep technical terms (Machine Learning, Python) in their original form.
        3. Output ONLY the translated text.
        """
        
        try:
            # Routing logic based on model choice
            response = self.groq_client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with {model_choice}: {str(e)}"

    def translate_with_gemma(self, text, target_lang):
        """Forces Algerian Darija anchoring and prevents Egyptian contamination."""
        processed_text = self._preprocess_darija(text)
        
        system_prompt = f"""
        You are an expert Algerian translator. Convert this mixed-language sentence into {target_lang}.
        RULES:
        1. If the target is Algerian Darija, NEVER use Egyptian words like 'عايز' or 'باشا'.
        2. Maintain French/English technical terms as used in Algerian professional speech.
        3. Output ONLY the translated text.
        """
        
        messages = [{"role": "user", "content": f"{system_prompt}\nSentence: {processed_text}"}]
        prompt = self.gemma_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.gemma_tokenizer(prompt, return_tensors="pt").to(self.gemma_model.device)
        
        outputs = self.gemma_model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=0.1,
            do_sample=False 
        )
        
        result = self.gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.split("model")[-1].strip()

class MaerifaPrecision:
    def __init__(self, device=-1):
        self.ner_1 = pipeline("ner", model="marefa-nlp/marefa-ner", aggregation_strategy="simple", device=device)
        self.ner_2 = pipeline("ner", model="hatmimoha/arabic-ner", aggregation_strategy="simple", device=device)
        self.tech_lexicon = {"bouramoul", "sdia", "ntic", "uc2", "ifa", "dziribert", "aws", "python", "nlp", "pytorch"}
        self.fr_lexicon = {"je", "tu", "il", "elle", "nous", "vous", "est", "suis", "avec", "pour", "dans", "mais"}
        self.en_lexicon = {"the", "is", "to", "we", "are", "it", "dataset", "accuracy", "model", "processing"}

    def get_entities(self, text):
        try:
            res = self.ner_1(text[:1000]) + self.ner_2(text[:1000])
            return {r["word"].lower().strip() for r in res if r["entity_group"] in ["PER", "LOC", "ORG"]}
        except: return set()

class MaerifaHybridEngine:
    def __init__(self, path_ar, path_eu, openai_key=None):
        self.gpu_id = 0 if torch.cuda.is_available() else -1
        self.ar_model = pipeline("token-classification", model=path_ar, device=self.gpu_id, aggregation_strategy="simple")
        self.eu_model = pipeline("token-classification", model=path_eu, device=-1, aggregation_strategy="simple")
        self.maerifa = MaerifaPrecision(device=self.gpu_id)
        self.translator = MaerifaTranslator(cache_dir=CACHE_DIR, device_id=self.gpu_id)
        self.client = OpenAI(api_key=openai_key) if openai_key else None
        self.confidence_threshold = 0.85

    def get_reformulations(self, text):
        mapping = {
            "darija": "ary_Arab",
            "arabic": "ara_Arab",
            "french": "fra_Latn",
            "english": "eng_Latn"
        }
        results = {}
        for label, code in mapping.items():
            results[label] = self.translator.translate(text, target_lang_code=code)
        return results

    def process(self, text, dictionary=None):
        words = re.findall(r"[\u0600-\u06FF]+|[a-zA-Z0-9]+(?:[-'][a-zA-Z]+)*|[^\w\s]", text)
        if not words: return [], 0
        
        entities = self.maerifa.get_entities(text)
        label_map = {"DZA_AR": "DZA_AR", "DZA_LAT": "DZA_LAT", "MSA": "MSA", "FR": "FR", "EN": "EN", "ENTITY": "ENTITY"}
        
        with torch.no_grad():
            preds_eu = self.eu_model(text[:1500])
            preds_ar = self.ar_model(text[:1500])
        
        scores = []
        offset = 0
        
        for w in words:
            low_w = w.lower().strip()
            dist = Counter()
            detected_source = "xlmr" 
            
            # 1. Vérification Dictionnaire (Heuristique)
            if dictionary:
                found_dict = False
                for category, terms in dictionary.items():
                    if low_w in [t.lower() for t in terms]:
                        dist["ENTITY"] += 100
                        detected_source = "rules"
                        found_dict = True
                        break
                if found_dict:
                    scores.append({"dist": dist, "source": detected_source})
                    continue

            # 2. Vérification Lexique Technique
            if low_w in self.maerifa.tech_lexicon or low_w in entities:
                dist["ENTITY"] += 60
                detected_source = "rules"
            
            # 3. Détection via Modèles Fine-tunés
            else:
                w_start = text.find(w, offset)
                w_end = w_start + len(w)
                offset = w_end
                
                found_model = False
                for ent in (preds_eu + preds_ar):
                    if not (w_end <= ent["start"] or w_start >= ent["end"]) and ent["score"] >= self.confidence_threshold:
                        lbl = ent["entity_group"].replace("LABEL_", "").replace("LID_", "").upper()
                        dist[lbl] += 40
                        detected_source = "dziribert" if lbl in ["DZA_AR", "MSA"] else "xlmr"
                        found_model = True
                        break
            
            scores.append({"dist": dist, "source": detected_source})

        # Sequence building
        labels = [label_map.get(max(s["dist"], key=lambda k: s["dist"][k]) if s["dist"] else "ENTITY", "ENTITY") for s in scores]
        sources = [s["source"] for s in scores]
        
        segments = []
        curr_txt, curr_lbl, curr_src = words[0], labels[0], sources[0]
        
        for i in range(1, len(words)):
            if labels[i] == curr_lbl:
                curr_txt += (" " + words[i]) if re.search(r'\w$', curr_txt) else words[i]
            else:
                segments.append({"text": curr_txt.strip(), "label": curr_lbl, "source": curr_src})
                curr_txt, curr_lbl, curr_src = words[i], labels[i], sources[i]
        
        segments.append({"text": curr_txt.strip(), "label": curr_lbl, "source": curr_src})
        
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return segments, 100.0

def load_my_local_models():
    # OpenAI key for general purposes
    api_key = ""
    
    return MaerifaHybridEngine(
        "models/dziribert_final", 
        "models/xlmr_final", 
        openai_key=api_key
    )

import json
import os
import whisper
import tempfile
import traceback
from collections import Counter
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .lids_logic import load_my_local_models

# --- GLOBAL RESOURCE CACHE ---
_LID_ENGINE = None
_WHISPER_MODEL = None

# --- HYBRID DATA LAYER (Server-Side Mirror of Frontend DICT) ---
LOCAL_DICT = {
    "Family": ["baba", "yemma", "khou", "okht", "بابا", "يمة"],
    "Food": ["khoubz", "ma", "l'hem", "qahwa", "خبز", "ماء"],
    "Verbs": ["yakoul", "yishrob", "yrouh", "yorgod", "ياكل", "يشرب"]
}

def wanlp(request):
    """Renders the main Maerifa Precision Engine interface."""
    return render(request, 'wanlp.html')

def get_resources():
    """Lazy loads heavy ML models to conserve memory."""
    global _LID_ENGINE, _WHISPER_MODEL
    if _LID_ENGINE is None:
        _LID_ENGINE = load_my_local_models()
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("base")
    return _LID_ENGINE, _WHISPER_MODEL

@csrf_exempt
def analyze_text(request):
    """Hybrid Analysis: Dictionary Lookup + Neural LID."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    try:
        lid_engine, stt_engine = get_resources() 
        final_text = ""

        # 1. INPUT HANDLING (Audio Transcription vs. Raw Text)
        if request.FILES.get('audio_data'):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                for chunk in request.FILES['audio_data'].chunks():
                    tmp.write(chunk)
                temp_path = tmp.name
            res = stt_engine.transcribe(temp_path)
            final_text = res["text"].strip()
            os.remove(temp_path)
        else:
            body = request.body.decode('utf-8') if request.body else "{}"
            payload = json.loads(body)
            final_text = payload.get("text", "")
            # Capture model choice from AVAILABLE_MODELS frontend dropdown
            selected_model = payload.get("model", "mixtral-8x7b-groq")

        if not final_text:
            return JsonResponse({"error": "Texte vide"}, status=400)

        # 2. HYBRID PROCESSING: Lexicon + Neural
        # We pass LOCAL_DICT to give the engine 'Ground Truth' hints
        segments, cmi = lid_engine.process(final_text, dictionary=LOCAL_DICT)
        
        # Reformulation using context (DZA/FR/MSA)
        reformulations = lid_engine.get_reformulations(final_text)

        # 3. RESPONSE
        return JsonResponse({
            "status": "success",
            "transcript": final_text,
            "segments": segments, # Now includes 'is_dict_match' and 'category'
            "cmi": cmi,
            "reformulations": {
                "darija": reformulations.get("darija", ""),
                "french": reformulations.get("french", ""),
                "arabic": reformulations.get("arabic", ""),
                "english": reformulations.get("english", "")
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def ai_expert(request):
    """Semantic Analysis & Architectural Explanation with Layer Attribution."""
    if request.method != "POST": 
        return JsonResponse({"error": "POST required"})
    
    try:
        data = json.loads(request.body)
        query = data.get("query", "").lower()
        context = data.get("context", {})
        segments = context.get('segments', []) # Expecting: [{"text": "...", "label": "...", "source": "rules/dziribert/xlmr/llm"}]
        
        # 1. Extract sources to identify which layer did the heavy lifting
        sources = [s.get('source', 'UNKNOWN') for s in segments]
        labels = [s.get('label', 'UNKNOWN') for s in segments]
        
        source_stats = Counter(sources)
        label_stats = Counter(labels)
        
        # 2. Check for Dictionary-driven categories
        categories = [s.get('category') for s in segments if s.get('category')]
        cat_stats = Counter(categories)

        # Logic: Domain Awareness
        if any(k in query for k in ["domaine", "catégorie", "sujet"]):
            if categories:
                top_cat = cat_stats.most_common(1)[0][0]
                answer = f"Le moteur a détecté une dominance du domaine **{top_cat}** via le dictionnaire local (Couche Heuristique)[cite: 1]."
            else:
                answer = "Aucun domaine spécifique n'a été identifié par le lexique local."

        # Logic: Detailed Architectural Explanation
        elif any(k in query for k in ["model", "couche", "expert", "comment", "détection"]):
            answer = "L'architecture Maerifa a traité votre requête via ces couches spécifiques[cite: 1, 2]:\n\n"
            
            # Identify if Rules (Heuristic) triggered
            if source_stats.get('rules'):
                answer += f"1. **Couche Heuristique (Règles)**: {source_stats['rules']} segment(s) détecté(s) par `LOCAL_DICT`[cite: 1].\n"
            
            # Identify if DziriBERT (Fine-tuned) triggered
            if any(l == "DZA_AR" for l in labels) or source_stats.get('dziribert'):
                count = source_stats.get('dziribert', 0)
                answer += f"2. **Couche Arabe (DziriBERT)**: Ce modèle a identifié {count} segment(s) en script arabe.\n"
            
            # Identify if XLM-RoBERTa (Fine-tuned) triggered
            if any(l in ["DZA_LAT", "FR", "EN"] for l in labels) or source_stats.get('xlmr'):
                count = source_stats.get('xlmr', 0)
                answer += f"3. **Couche Multilingue (XLM-RoBERTa)**: Utilisée pour {count} segment(s) en code-switching latin[cite: 2].\n"
            
            # Identify if LLM (Gemini/Groq) was used
            if source_stats.get('llm'):
                answer += f"4. **Couche LLM (Inférence)**: L'intelligence générative (Llama-3.3/Gemini) a été sollicitée pour la reformulation finale[cite: 2].\n"
            
            details = [f"{k}: {v}" for k, v in label_stats.items()]
            answer += f"\n**Distribution des étiquettes :** {', '.join(details)}."
        
        else:
            # Default response showing the most active detection source
            top_source = source_stats.most_common(1)[0][0] if source_stats else "N/A"
            answer = f"L'analyse est terminée. La détection a principalement été assurée par la couche **{top_source}**. Voulez-vous voir le détail des modèles utilisés ?"

        return JsonResponse({"response": answer})
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
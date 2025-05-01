from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

def evaluate_embeddings(questions: Dict[str, List[str]],
                        data: List[Dict[str, Any]],
                        orig_model: SentenceTransformer,
                        fine_model: Optional[SentenceTransformer]):
    """
    Porównuje jakość embeddingów przed i po dostrojeniu na podstawie
    średniego podobieństwa kosinusowego między pytaniami a abstraktami.
    """
    print("\n🧪 Rozpoczynanie ewaluacji jakości embeddingów...")

    if not data or not questions:
        print("⚠️ Nie można przeprowadzić ewaluacji: Brak danych PubMed lub pytań.")
        return
    if fine_model is None:
        print("⚠️ Nie można przeprowadzić pełnej ewaluacji: Brak dostrojonego modelu (`fine_model`).")
        # Można opcjonalnie przeprowadzić ewaluację tylko dla modelu oryginalnego
        return

    # Przygotuj dane do ewaluacji: wybierz próbkę ID, które mają abstrakt i pytania
    valid_ids = [
        r['id'] for r in data
        if r.get('abstrakt') and r['id'] in questions and questions[r['id']]
    ]

    if not valid_ids:
        print("⚠️ Nie znaleziono wpisów z abstraktami i pytaniami do ewaluacji.")
        return

    # Wybierz próbkę (np. 100 wpisów lub mniej, jeśli nie ma tylu)
    sample_size = min(100, len(valid_ids))
    sample_ids = random.sample(valid_ids, sample_size)
    sample_data_map = {r['id']: r for r in data if r['id'] in sample_ids} # Mapa dla szybkiego dostępu

    print(f"Ewaluacja na próbce {sample_size} artykułów...")

    orig_similarities = []
    fine_similarities = []
    evaluation_pairs_count = 0

    for pid in sample_ids:
        record = sample_data_map[pid]
        abstract = record.get("abstrakt")
        qs = questions.get(pid, [])

        if not abstract or not qs: continue # Powinno być już odfiltrowane, ale dla pewności

        # Oblicz embedding abstraktu raz dla każdego modelu
        try:
            o_a_emb = orig_model.encode(abstract).reshape(1, -1)
            f_a_emb = fine_model.encode(abstract).reshape(1, -1)
        except Exception as e:
             print(f"⚠️ Błąd podczas kodowania abstraktu dla ID {pid}: {e}. Pomijanie tego artykułu.")
             continue

        # Porównaj każde pytanie z abstraktem
        for q in qs:
            if not q: continue
            try:
                # Oblicz embedding pytania
                o_q_emb = orig_model.encode(q).reshape(1, -1)
                f_q_emb = fine_model.encode(q).reshape(1, -1)

                # Oblicz podobieństwo kosinusowe
                orig_sim = cosine_similarity(o_q_emb, o_a_emb)[0][0]
                fine_sim = cosine_similarity(f_q_emb, f_a_emb)[0][0]

                orig_similarities.append(orig_sim)
                fine_similarities.append(fine_sim)
                evaluation_pairs_count += 1
            except Exception as e:
                print(f"⚠️ Błąd podczas kodowania pytania lub obliczania podobieństwa dla ID {pid}, Pytanie: '{q[:50]}...': {e}")

    if not orig_similarities or not fine_similarities:
        print("⚠️ Nie udało się obliczyć żadnych wyników podobieństwa do ewaluacji.")
        return

    # Oblicz średnie wyniki
    mean_orig = np.mean(orig_similarities)
    mean_fine = np.mean(fine_similarities)
    delta = mean_fine - mean_orig
    improvement_percent = (delta / abs(mean_orig) * 100) if mean_orig != 0 else float('inf')

    print("-" * 30)
    print(f"📊 Wyniki Ewaluacji (na podstawie {evaluation_pairs_count} par pytanie-abstrakt):")
    print(f"  - Średnie podobieństwo (model bazowy): {mean_orig:.4f}")
    print(f"  - Średnie podobieństwo (model dostrojony): {mean_fine:.4f}")
    print(f"  - Różnica (Delta): {delta:+.4f}")
    print(f"  - Poprawa: {improvement_percent:+.2f}%")
    print("-" * 30)

print("Funkcja evaluate_embeddings zdefiniowana.")
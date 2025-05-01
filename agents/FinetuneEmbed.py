import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer
from Tools.State_dict import StateDict
import pandas as pd
def agent_embedding(state: StateDict) -> StateDict:
    # Sprawdź, czy model został już załadowany/wytrenowany w tej sesji LUB załadowany z dysku na początku
    if state.get("model") is not None:
        print("ℹ️ [FineTuneEmbed] Model już istnieje w stanie (załadowany lub wcześniej dostrojony). Pomijanie ponownego dostrajania.")
        # Upewnijmy się, że globalne zmienne są zsynchronizowane ze stanem
        global embed_model, embed_fn
        if embed_model is not state["model"]: # Jeśli stan ma inny model niż globalny
            print("🔄 Aktualizowanie globalnego modelu i funkcji embeddingów zgodnie ze stanem.")
            embed_model = state["model"]
            embed_fn = EmbeddingFunction(embed_model)
            if collection:
                collection.embedding_function = embed_fn # Zaktualizuj też w kolekcji ChromaDB
        return state

    # Sprawdź, czy istnieje zapisany model na dysku (z poprzedniego uruchomienia),
    # jeśli nie został załadowany na początku lub w stanie.
    # Ten kod jest trochę redundantny z komórką 2, ale zapewnia spójność.
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
          print(f"ℹ️ [FineTuneEmbed] Znaleziono istniejący model w {MODEL_DIR}. Ładowanie...")
          try:
              loaded_model = SentenceTransformer(MODEL_DIR)
              state["model"] = loaded_model # Zapisz załadowany model w stanie
              # Zaktualizuj globalne zmienne
              embed_model = loaded_model
              embed_fn = EmbeddingFunction(embed_model)
              if collection:
                  collection.embedding_function = embed_fn
              print("✅ [FineTuneEmbed] Pomyślnie załadowano istniejący model. Pomijanie dostrajania.")
              return state
          except Exception as e:
               print(f"⚠️ Nie udało się załadować modelu z {MODEL_DIR}: {e}. Próba dostrojenia od nowa.")
               # Nie ustawiaj state["model"], przejdź do dostrajania

    # --- Kontynuuj tylko, jeśli nie załadowano modelu ---

    print("🚀 [FineTuneEmbed] Rozpoczynanie procesu dostrajania modelu embeddingów.")
    if not state.get("pubmed_data") or not state.get("questions"):
        print("⚠️ [FineTuneEmbed] Brak danych PubMed lub pytań w stanie. Nie można dostroić modelu.")
        state["model"] = embed_model # Ustaw model bazowy w stanie, jeśli nic innego nie ma
        return state

    print("📝 [FineTuneEmbed] Przygotowywanie przykładów treningowych (pozytywnych i negatywnych)...")
    examples: List[InputExample] = []
    all_ids_with_questions = list(state["questions"].keys()) # ID, dla których mamy pytania

    positive_pairs = 0
    negative_pairs = 0

    # Użyj tylko danych, które mają zarówno abstrakt, jak i pytania
    valid_data = [item for item in state["pubmed_data"] if item['id'] in state["questions"] and state["questions"][item['id']] and item.get("abstrakt")]

    if not valid_data:
         print("⚠️ [FineTuneEmbed] Brak danych z abstraktami i pytaniami do utworzenia przykładów. Pomijanie dostrajania.")
         state["model"] = embed_model
         return state

    for rec in valid_data:
        rec_id = rec["id"]
        abstract = rec.get("abstrakt") # Wiemy, że istnieje i nie jest pusty z filtra powyżej
        qs = state["questions"].get(rec_id, []) # Wiemy, że lista nie jest pusta

        # 1. Przykłady pozytywne: (pytanie z tego artykułu, abstrakt tego artykułu)
        for q in qs:
            if q and abstract: # Dodatkowe sprawdzenie poprawności
                examples.append(InputExample(texts=[q, abstract], label=1.0))
                positive_pairs += 1

        # 2. Przykłady negatywne: (pytanie z INNEGO artykułu, abstrakt TEGO artykułu)
        # Wybierz kilka losowych ID innych niż bieżący, które mają pytania
        possible_neg_ids = [pid for pid in all_ids_with_questions if pid != rec_id]
        num_neg_samples = min(len(possible_neg_ids), 3) # Ile negatywnych przykładów na pozytywny

        if num_neg_samples > 0:
            neg_ids_sample = random.sample(possible_neg_ids, num_neg_samples)
            for neg_id in neg_ids_sample:
                neg_qs = state["questions"].get(neg_id, [])
                if neg_qs: # Jeśli dla negatywnego ID są pytania
                    # Wybierz losowe pytanie z negatywnego przykładu
                    neg_q = random.choice(neg_qs)
                    if neg_q and abstract: # Upewnij się, że mamy oba teksty
                        examples.append(InputExample(texts=[neg_q, abstract], label=0.0))
                        negative_pairs += 1

    if not examples:
        print("⚠️ [FineTuneEmbed] Nie udało się utworzyć żadnych przykładów treningowych. Pomijanie dostrajania.")
        state["model"] = embed_model # Ustaw model bazowy
        return state

    print(f"📊 Przygotowano {len(examples)} przykładów treningowych ({positive_pairs} pozytywnych, {negative_pairs} negatywnych).")

    # --- Rozpocznij trening ---
    # Użyj bieżącego globalnego `embed_model` (może być bazowy lub załadowany z dysku)
    current_embed_model = embed_model
    print(f"💪 Rozpoczynanie dostrajania modelu: {current_embed_model.__class__.__name__} (na bazie {MODEL_NAME} lub z {MODEL_DIR})")

    # Przygotuj DataLoader i funkcję straty
    # Zwiększ batch_size, jeśli pamięć GPU pozwala
    train_batch_size = 128
    loader = DataLoader(examples, shuffle=True, batch_size=train_batch_size)
    loss_fn = losses.CosineSimilarityLoss(model=current_embed_model)

    # Definicja callback'a (opcjonalnie, dla śledzenia postępów)
    epochs_done = 0
    def simple_callback(score, epoch, steps):
        nonlocal epochs_done
        if epoch > epochs_done:
            print(f"Epoch {epoch+1} zakończony. Średnia strata: {score:.4f}")
            epochs_done = epoch

    num_epochs = 1 # Trenuj przez jedną epokę (można zwiększyć dla lepszych wyników)
    warmup_steps = int(len(loader) * num_epochs * 0.1) # 10% kroków na rozgrzewkę

    try:
        current_embed_model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            # callback=simple_callback, # Włącz, jeśli chcesz widzieć postęp epok
            output_path=MODEL_DIR,      # Zapisuj model (checkpointy) tutaj
            show_progress_bar=True,     # Pokaż pasek postępu
            save_best_model=False       # Zapisz model po zakończeniu ostatniej epoki
                                        # Ustaw True, jeśli chcesz zapisywać najlepszy model na podstawie dev set (wymaga dev set)
        )
        print("✅ [FineTuneEmbed] Dostrajanie zakończone pomyślnie.")
        state["model"] = current_embed_model # Zapisz dostrojony model w stanie

        # Zaktualizuj globalny model i funkcję embeddingów
        embed_model = current_embed_model
        embed_fn = EmbeddingFunction(embed_model)
        if collection:
            collection.embedding_function = embed_fn # Zaktualizuj funkcję w ChromaDB
            print("⚙️ Zaktualizowano funkcję embeddingów w kolekcji ChromaDB.")

        # Zapisz finalny model (fit też zapisuje, ale dla pewności)
        os.makedirs(MODEL_DIR, exist_ok=True)
        current_embed_model.save(MODEL_DIR)
        print(f"💾 Dostrojony model zapisany w {MODEL_DIR}")

    except Exception as e:
        print(f"❌ Błąd podczas procesu dostrajania modelu: {e}")
        # W razie błędu, zachowaj model sprzed próby dostrojenia w stanie
        # (zakładamy, że `embed_model` nie został zmieniony przez nieudany `fit`)
        state["model"] = embed_model
        print("⚠️ Używany będzie model sprzed nieudanej próby dostrojenia.")

    return state

print("Agent FineTuneEmbed zdefiniowany.")
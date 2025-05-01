import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer
from Tools.State_dict import StateDict
import pandas as pd
def agent_update_embeddings(state: StateDict) -> StateDict:
    # Pobierz model do użycia ze stanu (bazowy lub dostrojony)
    model_to_use = state.get("model")

    if model_to_use is None:
        print("⚠️ [UpdateEmbeddings] Brak modelu w stanie (ani bazowego, ani dostrojonego). Pomijanie aktualizacji embeddingów.")
        # Można rozważyć próbę użycia globalnego `embed_model`, ale lepiej polegać na stanie
        return state

    if not state.get("pubmed_data"):
        print("⚠️ [UpdateEmbeddings] Brak danych PubMed w stanie. Pomijanie aktualizacji embeddingów.")
        return state

    print(f"📡 [UpdateEmbeddings] Rozpoczynanie aktualizacji embeddingów w ChromaDB przy użyciu modelu: {'Dostrojony' if model_to_use != SentenceTransformer(MODEL_NAME) else 'Bazowy'}...")

    updated_count = 0
    total_to_update = len(state["pubmed_data"])
    print(f"Łączna liczba wpisów do przetworzenia: {total_to_update}")

    # Użyj EMB_BATCH z konfiguracji
    for idx, batch_data in enumerate(chunked(state["pubmed_data"], EMB_BATCH), start=1):
        ids_batch = [r["id"] for r in batch_data]
        # Upewnij się, że 'abstrakt' istnieje i jest stringiem
        texts_batch = [str(r["abstrakt"]) for r in batch_data if r.get("abstrakt")]
        # Odfiltruj ID dla tych rekordów, które mają poprawny abstrakt
        valid_ids_batch = [r["id"] for r in batch_data if r.get("abstrakt")]

        if not valid_ids_batch:
            print(f"⚠️ Batch {idx} nie zawiera wpisów z abstraktami. Pomijanie.")
            continue

        print(f"Przetwarzanie batcha {idx} ({len(valid_ids_batch)} wpisów)...", end='\r')

        try:
            # Wygeneruj embeddingi tylko dla poprawnych tekstów
            vecs = model_to_use.encode(texts_batch, batch_size=len(texts_batch), show_progress_bar=False).tolist()

            # Zaktualizuj ChromaDB używając pasujących ID
            collection.update(ids=valid_ids_batch, embeddings=vecs)
            updated_count += len(valid_ids_batch)

            # Rzadsze logowanie postępu
            # if idx % 10 == 0 or len(valid_ids_batch) < EMB_BATCH :
            #    print(f"✅ Przetworzono batch {idx}. Łącznie zaktualizowano: {updated_count}/{total_to_update} embeddingów.")

        except Exception as e:
            print(f"\n❌ Błąd podczas aktualizacji embeddingów dla batcha {idx} (pierwsze ID: {valid_ids_batch[0] if valid_ids_batch else 'brak'}): {e}")
            # Można zdecydować, czy kontynuować, czy przerwać przy błędzie

    print(f"\n✅ [UpdateEmbeddings] Zakończono aktualizację embeddingów dla {updated_count}/{total_to_update} wpisów.")
    return state

print("Agent UpdateEmbeddings zdefiniowany.")
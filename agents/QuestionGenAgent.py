import os, sys, torch, numpy as np
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer
from Tools.chunked import chunked
from Tools.State_dict import StateDict
from config import collection

load_dotenv()

# Configure question generation model
qg_model_name = "google/long-t5-local-base"
QG_BATCH = 32
try:
    tokenizer_qg = AutoTokenizer.from_pretrained(qg_model_name)
    qg = pipeline("text2text-generation", model=qg_model_name, tokenizer=tokenizer_qg, device=0 if torch.cuda.is_available() else -1)
except Exception:
    tokenizer_qg = AutoTokenizer.from_pretrained("t5-small")
    qg = pipeline("text2text-generation", model="t5-small", tokenizer=tokenizer_qg, device=0 if torch.cuda.is_available() else -1)

# Definicja agenta generującego pytania
def agent_generate_questions(state: StateDict) -> StateDict:
    if not qg:
        print("❌ [QuestionGen] Model generowania pytań (qg) nie jest dostępny. Pomijanie kroku.")
        return state
    if not state.get("pubmed_data"):
        print("⚠️ [QuestionGen] Brak danych PubMed ('pubmed_data') w stanie. Pomijanie generowania pytań.")
        return state

    # Inicjalizuj słownik pytań, jeśli nie istnieje
    if "questions" not in state or state["questions"] is None:
        state["questions"] = {}

    # Sprawdź, ile wpisów już ma pytania (wczytane ze stanu/ChromaDB)
    ids_with_questions = set(state["questions"].keys())
    all_ids_in_data = {item['id'] for item in state["pubmed_data"]}
    ids_needing_questions = list(all_ids_in_data - ids_with_questions)

    total_entries = len(all_ids_in_data)
    processed_count = total_entries - len(ids_needing_questions)

    if not ids_needing_questions:
        print(f"ℹ️ [QuestionGen] Wszystkie {total_entries} wpisy mają już pytania. Pomijanie generowania.")
        return state
    else:
        print(f"ℹ️ [QuestionGen] Stan początkowy: {processed_count}/{total_entries} wpisów ma pytania.")
        print(f"Rozpoczynanie generowania pytań dla pozostałych {len(ids_needing_questions)} wpisów...")

    # Stwórz mapę ID -> rekord dla szybkiego dostępu
    data_map = {item['id']: item for item in state["pubmed_data"]}
    processed_in_run = 0
    num_return_sequences = 3 # Ile pytań generować na abstrakt

    # Przetwarzaj tylko te ID, które potrzebują pytań
    for id_batch in chunked(ids_needing_questions, QG_BATCH):
        batch_records = [data_map[pid] for pid in id_batch if pid in data_map]
        if not batch_records: continue # Pomiń pusty batch

        # Przygotuj prompty (tylko dla rekordów z abstraktem)
        prompts = []
        records_in_batch = []
        for rec in batch_records:
             abstract = rec.get("abstrakt")
             if abstract and abstract.strip(): # Upewnij się, że abstrakt istnieje i nie jest pusty
                 # Ogranicz długość abstraktu, aby zmieścić się w limicie tokenów modelu T5
                 max_input_length = tokenizer_qg.model_max_length - 64 # Zostaw margines na prompt i tokeny specjalne
                 truncated_abstract = abstract[:max_input_length] if len(abstract) > max_input_length else abstract
                 prompts.append(f"generate questions based on this text: {truncated_abstract}")
                 records_in_batch.append(rec)
             else:
                  print(f"⚠️ Pomijanie ID {rec.get('id')} - brak abstraktu.")

        if not prompts: # Jeśli żaden rekord w batchu nie miał abstraktu
             print(f"⚠️ Batch pominięty - brak abstraktów do przetworzenia.")
             continue

        print(f"Generowanie pytań dla batcha {len(prompts)} abstraktów (pierwszy ID: {records_in_batch[0].get('id')})...")

        try:
            # Użyj dynamicznego batch_size dla pipeline
            outs = qg(prompts,
                      max_length=64,            # Maksymalna długość generowanego pytania
                      num_beams=num_return_sequences, # Użyj beam search dla lepszej jakości
                      num_return_sequences=num_return_sequences,
                      batch_size=len(prompts),   # Dopasuj batch size do liczby promptów
                      do_sample=False)           # Wyłącz sampling dla większej spójności przy beam search
        except Exception as e:
            print(f"❌ Błąd podczas generowania pytań dla batcha (pierwszy ID: {records_in_batch[0].get('id')}): {e}")
            # Rozważ logowanie ID batcha, który się nie powiódł
            continue # Pomiń resztę przetwarzania dla tego batcha

        # Przetwarzanie wyników - qg zwraca listę list lub płaską listę
        grouped_outs = {}
        if isinstance(outs[0], list): # Jeśli wynik to lista list [[q1, q2, q3], [q1, q2, q3], ...]
            if len(outs) == len(records_in_batch):
                for i, rec_outputs in enumerate(outs):
                    grouped_outs[records_in_batch[i]['id']] = rec_outputs
            else:
                 print(f"⚠️ Niezgodna liczba wyników ({len(outs)}) z liczbą rekordów ({len(records_in_batch)}) w batchu. Pomijanie przypisania wyników.")
                 continue # Przejdź do następnego batcha
        else: # Jeśli wynik to płaska lista [q1a, q1b, q1c, q2a, q2b, q2c, ...]
            if len(outs) == len(records_in_batch) * num_return_sequences:
                for i, rec in enumerate(records_in_batch):
                    start_idx = i * num_return_sequences
                    end_idx = (i + 1) * num_return_sequences
                    grouped_outs[rec['id']] = outs[start_idx:end_idx]
            else:
                print(f"⚠️ Niezgodna liczba wyników ({len(outs)}) z oczekiwaną ({len(records_in_batch) * num_return_sequences}) w batchu. Pomijanie przypisania wyników.")
                continue # Przejdź do następnego batcha


        # Aktualizacja stanu i ChromaDB
        ids_to_update_chroma = []
        metadatas_to_update_chroma = []

        for rec_id, generated_outputs in grouped_outs.items():
            # Wyciągnij tekst pytania i odfiltruj puste lub nieudane generacje
            pytania = []
            for o in generated_outputs:
                if isinstance(o, dict):
                    text = o.get("generated_text", "")
                else:
                    text = str(o)
                text = text.strip()
                if text and not text.lower().startswith("generate questions"):
                    pytania.append(text)

            if not pytania:
                print(f"⚠️ Nie wygenerowano poprawnych pytań dla ID {rec_id}")
                # Można rozważyć dodanie pustej listy do stanu, aby zaznaczyć, że próbowano
                # state["questions"][rec_id] = []
                continue # Nie aktualizuj, jeśli nie ma pytań

            # Dodaj pytania do stanu
            state["questions"][rec_id] = pytania
            processed_in_run += 1

            # Przygotuj metadane do aktualizacji w ChromaDB
            # Pobierz istniejące metadane, aby ich nie nadpisać
            try:
                 existing_meta = collection.get(ids=[rec_id], include=["metadatas"])["metadatas"]
                 if existing_meta and existing_meta[0] is not None:
                     current_meta = existing_meta[0]
                 else:
                     # Jeśli nie ma metadanych, pobierz z `data_map` (mniej preferowane)
                     record_data = data_map.get(rec_id, {})
                     current_meta = {k: v for k, v in record_data.items() if k not in ["id", "abstrakt"]}
            except Exception as e:
                 print(f"⚠️ Błąd pobierania istniejących metadanych dla {rec_id}: {e}. Używanie pustych metadanych.")
                 current_meta = {}

            # Dodaj/zaktualizuj pole 'pytania'
            current_meta["pytania"] = " || ".join(pytania) # Zapisz jako string oddzielony " || "

            ids_to_update_chroma.append(rec_id)
            metadatas_to_update_chroma.append(current_meta)

        # Zaktualizuj ChromaDB dla przetworzonego batcha
        if ids_to_update_chroma:
            try:
                collection.update(ids=ids_to_update_chroma, metadatas=metadatas_to_update_chroma)
                # print(f"✅ Zaktualizowano metadane (pytania) w ChromaDB dla {len(ids_to_update_chroma)} wpisów.")
            except Exception as e:
                print(f"❌ Błąd podczas aktualizacji metadanych w ChromaDB dla batcha (pierwszy ID: {ids_to_update_chroma[0]}): {e}")

        print(f"✅ Batch zakończony. Wygenerowano pytania dla {len(grouped_outs)} wpisów.")

    final_processed_count = total_entries - len(ids_needing_questions) + processed_in_run
    print(f"✅ [QuestionGen] Zakończono generowanie pytań. Łącznie przetworzono w tej sesji: {processed_in_run} nowych wpisów.")
    print(f"Aktualny stan: {len(state['questions'])}/{total_entries} wpisów ma wygenerowane pytania.")
    return state

print("Agent QuestionGen zdefiniowany.")
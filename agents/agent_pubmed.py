import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer
from Tools.State_dict import StateDict
import pandas as pd
import os
import shutil
import pandas as pd
from langgraph.graph import StateGraph
from typing import TypedDict, Dict, List, Optional, Any
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import pipeline, AutoTokenizer
from Bio import Entrez
from itertools import islice
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import time
from http.client import IncompleteRead
import torch
from Tools.EmbeddingFunction import EmbeddingFunction
from Tools.State_dict import StateDict
from agents.QuestionGenAgent import agent_generate_questions
from agents.FinetuneEmbed import agent_embedding
from agents.UpdateEmbedding import agent_update_embeddings
from Tools.evaluate_embeddings import evaluate_embeddings
from itertools import islice
from config import collection
import os, json

WORK_PERSIST  = 'Outputs/Part2_outputs/chroma-persist'
WORK_MODEL    = 'Outputs/Part2_outputs/Sentence transformer'

# Entrez API from .env file
ENTREZ_EMAIL = os.getenv('ENTREZ_EMAIL')

MODEL_NAME      = "all-MiniLM-L6-v2" 
COLLECTION_NAME = "Baza1"            
PERSIST_DIR     = WORK_PERSIST
MODEL_DIR       = WORK_MODEL  

# loading model from dir - we previosly saved it so its faster to load
print(f"Ładowanie bazowego modelu embeddingów: {MODEL_NAME}...")
if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
     print(f"Znaleziono istniejący model w {MODEL_DIR}. Ładowanie...")
     try:
         embed_model = SentenceTransformer(MODEL_DIR, device="cpu")
         print("Załadowano model z dysku.")
     except Exception as e:
         print(f"Błąd ładowania modelu z {MODEL_DIR}: {e}. Ładowanie bazowego modelu {MODEL_NAME}.")
         embed_model = SentenceTransformer(MODEL_NAME, device="cpu")
else:
     print(f"Brak modelu w {MODEL_DIR}. Ładowanie bazowego modelu {MODEL_NAME}.")
     embed_model = SentenceTransformer(MODEL_NAME, device="cpu")

try:
    embed_fn = EmbeddingFunction(embed_model)
    print("Embedded model loaded.")
except Exception as e:
    print('Error loading embedding function:', e)
    embed_fn = None

PUBMED_BATCH = 64

def agent_pubmed(state: StateDict) -> StateDict:
    if state.get('pubmed_data'):
        print('[PubMedFetch] Data already fetched. Skipping...')
        current_ids_in_state = {item['id'] for item in state['pubmed_data']}
        existing_in_chroma = set(collection.get(ids=list(current_ids_in_state))['ids'])
        ids_to_potentially_add = current_ids_in_state - existing_in_chroma

        if ids_to_potentially_add:
            print(f"ℹ️ [PubMedFetch] Dodajemy {len(ids_to_potentialnie_add)} brakujących danych do ChromaDB...")
            docs_to_add, metas_to_add = [], []
            for item in state["pubmed_data"]:
                if item['id'] in ids_to_potentially_add:
                    docs_to_add.append(item.get("abstrakt", ""))
                    meta = {k: v for k, v in item.items() if k not in ["id", "abstrakt"]}
                    metas_to_add.append(meta)
            try:
                collection.add(ids=ids_to_potentially_add, documents=docs_to_add, metadatas=metas_to_add)
                print(f"✅ [PubMedFetch] Dodano {len(ids_to_potentially_add)} wpisów do ChromaDB.")
            except Exception as e:
                print(f"❌ [PubMedFetch] Błąd przy dodawaniu do ChromaDB: {e}")
        else:
            print("ℹ️ [PubMedFetch] Wszystkie dane ze stanu są już w ChromaDB.")
        return state

    print("📡 [PubMedFetch] Rozpoczynam pobieranie abstraktów z PubMed...")

    try:
        df = pd.read_csv(state["csv_path"], dtype={"pmid": str}, low_memory=False)
    except Exception as e:
        print(f"❌ [PubMedFetch] Błąd przy ładowaniu CSV: {e}")
        state["pubmed_data"] = []
        return state

    rheu_df = df[df["label"] == "rheu"]
    if rheu_df.empty:
        print("⚠️ [PubMedFetch] Brak artykułów z etykietą 'rheu'.")
        state["pubmed_data"] = []
        return state

    meta_map = {
        str(r.pmid): {"kategoria": r.label, "tytuł": r.title, "czasopismo": r.journal}
        for r in rheu_df.itertuples() if r.pmid
    }
    all_pmids = list(meta_map.keys())
    print(f"Znaleziono {len(all_pmids)} unikalnych PMID.")

    try:
        existing_ids_in_chroma = set(collection.get(ids=all_pmids)['ids'])
    except Exception as e:
        print(f"⚠️ [PubMedFetch] Błąd przy sprawdzaniu ChromaDB: {e}")
        existing_ids_in_chroma = set()

    pmids_to_fetch = [pid for pid in all_pmids if pid not in existing_ids_in_chroma]
    print(f"Liczba nowych PMID do pobrania: {len(pmids_to_fetch)}")

    if not pmids_to_fetch:
        print("✅ [PubMedFetch] Brak nowych artykułów do pobrania.")
        return state

    fetched_data_list = []

    for pmid_batch in chunked(pmids_to_fetch, PUBMED_BATCH):
        print(f"Pobieranie batcha ({len(pmid_batch)} ID), pierwszy: {pmid_batch[0]}...")
        success = False
        attempts = 0
        max_attempts = 3

        while not success and attempts < max_attempts:
            attempts += 1
            try:
                handle = Entrez.efetch(db="pubmed", id=",".join(pmid_batch), retmode="xml")
                rec = Entrez.read(handle)  # ← bez czytania .read() i bez StringIO
                success = True
                handle.close()
            except Exception as e:
                print(f"⚠️ Błąd podczas pobierania Entrez (próba {attempts}/{max_attempts}): {e}")
                time.sleep(2 ** attempts)
                if handle:
                    handle.close()

        if not success:
            print(f"❌ Nie udało się pobrać batcha zaczynającego się od {pmid_batch[0]} po {max_attempts} próbach. Pomijanie.")
            continue

        articles = rec.get("PubmedArticle", [])
        ids_batch, docs_batch, metas_batch = [], [], []

        for art in articles:
            medline_citation = art.get("MedlineCitation")
            if not medline_citation: continue
            pmid_obj = medline_citation.get("PMID")
            if not pmid_obj: continue
            pid = str(pmid_obj)

            article_info = medline_citation.get("Article")
            if not article_info: continue

            abstract_info = article_info.get("Abstract", {})
            abs_list = abstract_info.get("AbstractText", [])

            if not abs_list:
                continue

            abstract = " ".join(str(part) for part in abs_list if part).strip()
            if not abstract:
                continue

            info = meta_map.get(pid, {})
            if info:
                fetched_data_list.append({"id": pid, "abstrakt": abstract, **info})
                ids_batch.append(pid)
                docs_batch.append(abstract)
                metas_batch.append(info)

        if ids_batch:
            try:
                collection.add(ids=ids_batch, documents=docs_batch, metadatas=metas_batch)
                print(f"✅ Dodano {len(ids_batch)} nowych wpisów do ChromaDB.")
            except Exception as e:
                print(f"❌ Błąd podczas dodawania batcha do ChromaDB: {e}")
        else:
            print("ℹ️ Brak poprawnych artykułów z abstraktami w batchu.")

    all_ids_in_chroma = existing_ids_in_chroma.union({item['id'] for item in fetched_data_list})
    if all_ids_in_chroma:
        try:
            entries = collection.get(ids=list(all_ids_in_chroma), include=["documents", "metadatas"])
            # W agent_pubmed, po załadowaniu 'entries' z ChromaDB
            loaded_data = []
            loaded_questions = {} # Nowy słownik na pytania
            for pid, doc, m in zip(entries["ids"], entries["documents"], entries["metadatas"]):
                if m is None: m = {}
                item = {"id": pid, "abstrakt": doc}
                # Zachowaj inne metadane
                item.update({k: v for k, v in m.items() if k != "pytania"})
                loaded_data.append(item)
                # Jeśli są pytania w metadanych, załaduj je
                if "pytania" in m and m["pytania"]:
                    # Zakładając, że pytania są stringiem rozdzielonym " || "
                    loaded_questions[pid] = m["pytania"].split(" || ")

                state["pubmed_data"] = loaded_data
                state["questions"] = loaded_questions # <--- Przekaż załadowane pytania do stanu
                print(f"✅ [PubMedFetch] Gotowe. Stan zawiera {len(state['pubmed_data'])} artykułów.")
                print(f"ℹ️ [PubMedFetch] Załadowano pytania dla {len(loaded_questions)} artykułów z ChromaDB.")
                print(f"✅ [PubMedFetch] Gotowe. Stan zawiera {len(state['pubmed_data'])} artykułów.")
        except Exception as e:
            print(f"❌ Błąd podczas finalnego ładowania danych: {e}")
            state["pubmed_data"] = fetched_data_list
    else:
        state["pubmed_data"] = []

    return state

from typing import Dict, Any, List
from Bio import Entrez
import os 
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

def search_pubmed(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    #Here we are searching PubMed for the query, parse the results, 
    #and look for the IdList - if does not exist, return empty list
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
    except RuntimeError as e:
        print(f"⚠️ [search_pubmed] PubMed search failed for query '{query[:50]}...': {e}")
        return []
    except Exception as e:
        print(f"❌ [search_pubmed] Unexpected error for query '{query[:50]}...': {e}")
        return []
    pmids = record.get("IdList", [])
    if not pmids: return []

    #Here we are using the IDList created above to fetch full records from PubMed
    try:
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype='xml', retmode="xml")
        records = Entrez.read(handle)
        handle.close()
    except RuntimeError as e:
        print(f"⚠️ [search_pubmed] PubMed efetch failed for IDs {pmids}: {e}")
        return []
    except Exception as e:
        print(f"❌ [search_pubmed] Unexpected efetch error for IDs {pmids}: {e}")
        return []

    articles = []
    
    print(records)

    # Check if 'PubmedArticleSet' is in the records
    # If not, return an empty list
    if 'PubmedArticle' not in records:
        print("No PubmedArticle found in the records.")
        return []

    #Here we are iterating through the found articles and extracting needed information
    for art in records.get('PubmedArticle', []):
        cit      = art["MedlineCitation"]
        info     = cit.get('Article', {})
        title    = info.get('ArticleTitle', '')
        abstract = ''.join(info.get('Abstract',{}).get('AbstractText',['']))
        pmid     = cit.get('PMID', '')
        articles.append({
            'title': title,
            'abstract': abstract,
            'pmid': pmid        
            })
    return articles
    
example_query = "bone resorption"

Entrez.email = os.getenv('PubMed_email')
Entrez.api_key = os.getenv('PubMed_API_KEY')

articles = search_pubmed(example_query, max_results=5)
print(articles)
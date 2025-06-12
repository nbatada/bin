#!/usr/bin/env python3

import re
import requests
from bs4 import BeautifulSoup
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# TCGA Cancer Type Abbreviations with common synonyms
cancer_types = {
    "LAML": ["acute myeloid leukemia"],
    "ACC": ["adrenocortical carcinoma"],
    "BLCA": ["bladder urothelial carcinoma","bladder cancer", "bladder"],
    "BCC": ["basal cell carcinoma"], # Skin
    "LGG": ["brain lower grade glioma"],
    "BRCA": ["breast invasive carcinoma", 'breast cancer', 'breast tumor'],
    "CESC": ["cervical squamous cell carcinoma", "endocervical adenocarcinoma"],
    "CHOL": ["cholangiocarcinoma"],
    "LCML": ["chronic myelogenous leukemia"],
    "COAD": ["colon adenocarcinoma", "colorectal cancer", "colorectal carcinoma", 'colon cancer',"colorectal"],
    "ESCA": ["esophageal"],
    "FPPP": ["ffpe pilot phase ii"],
    "GBM": ["glioblastoma"],
    "HNSC": ["head and neck squamous cell carcinoma", "squamous cell carcinoma","head and neck","nasopharyngeal"],
    "KICH": ["kidney chromophobe"],
    "KIRC": ["kidney renal clear cell carcinoma", "clear cell renal cell carcinoma", "renal cell carcinoma"],
    "KIRP": ["kidney renal papillary cell carcinoma"],
    "LIHC": ["liver hepatocellular carcinoma", "hepatocellular carcinoma", 'liver cancer'],
    "LUAD": ["lung adenocarcinoma", 'lung cancer'],
    "LUSC": ["lung squamous cell carcinoma"],
    "DLBC": ["lymphoid neoplasm diffuse large b-cell lymphoma"],
    "MESO": ["mesothelioma"],
    "OV": ["ovarian serous cystadenocarcinoma", "ovarian"],
    "PAAD": ["pancreatic adenocarcinoma", "pancreatic"],
    "PCPG": ["pheochromocytoma", "paraganglioma"],
    "PRAD": ["prostate adenocarcinoma", "prostate"],
    "READ": ["rectum adenocarcinoma"],
    "SARC": ["sarcoma"],
    "SKCM": ["skin cutaneous melanoma", "skin cancer", "melanoma"],
    "STAD": ["stomach adenocarcinoma", "gastric adenocarcinoma", "gastric tumor", "gastric"],
    "TGCT": ["testicular germ cell tumors"],
    "THYM": ["thymoma"],
    "THCA": ["thyroid carcinoma", "thyroid cancer"],
    "UCS": ["uterine carcinosarcoma"],
    "UCEC": ["uterine corpus endometrial carcinoma"],
    "UVM": ["uveal melanoma"],
    "PANCANCER":["cancer types"],
}

def get_cancer_type(text):
    text = text.lower()
    if "pancancer" in text or "pan-cancer" in text or re.search(r"\b\d+\s+human cancer[s]?\s*(types)?\b", text):
        return "PANCANCER"
    for acronym, names in cancer_types.items():
        for name in names:
            if name in text:
                return acronym
    return "UNKNOWN"

def classify_unknown(text):
    text = text.lower()
    if "pancancer" in text:
        return "PANCANCER"
    if re.search(r"\b\d+\s+human cancer[s]?\s*(types)?\b", text):
        return "PANCANCER"
    if any(term in text for term in ["inflammatory bowel disease", "rheumatoid arthritis"]):
        return "INFLAM"
    return "UNKNOWN"

def parse_pubmed(pubmed_id):
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract relevant fields
    title = soup.find('h1', class_='heading-title').get_text(strip=True)
    citation_text = soup.find('span', class_='cit').get_text()
    year = re.search(r'\b\d{4}\b', citation_text).group(0)
    abstract = soup.find('div', class_='abstract').get_text(strip=True)
    first_author_name = soup.find('a', class_='full-name').get_text(strip=True)
    first_author_lastname = first_author_name.split(' ')[-1].upper().replace('-','')
    #journal_name, volume_issue = citation_text.split('.', 1)
        # Extract Journal Name
    journal_element = soup.find('button', class_='journal-actions-trigger')
    journal_name = journal_element.get_text(strip=True) if journal_element else "Journal not available"

    # Extract Volume and Issue
    volume_issue = citation_text.split('.', 1)[1].strip() if '.' in citation_text else "Volume/Issue not available"

    # Extract DOI
    doi_element = soup.find('span', class_='citation-doi')
    doi = doi_element.get_text(strip=True) if doi_element else "DOI not available"

    # Check both title and abstract for cancer type
    cancer_type_title = get_cancer_type(title)
    cancer_type_abstract = get_cancer_type(abstract)

    # Prioritize specific matches from the abstract
    cancer_type = cancer_type_abstract if cancer_type_abstract != "UNKNOWN" else cancer_type_title

    # If still unknown, classify as unknown
    if cancer_type == "UNKNOWN":
        cancer_type = classify_unknown(abstract)

    # Format the result
    data_handle = f"{cancer_type}_{year}_{pubmed_id}_{first_author_lastname}"
    full_reference = (
        f"{first_author_lastname.capitalize()} et al. {title}. {year}. {journal_name.strip()}. PMID: {pubmed_id}. {doi}, {url}"
    )
    
    #full_reference = (
    #    f"{first_author_lastname.capitalize()} et al. {title}. {year}. {journal_name.strip()}. {volume_issue.strip()}"
    #)

    return data_handle, full_reference

def fetch_geo_id(geo_id):
    geo_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={geo_id}"
    response = requests.get(geo_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    citations = soup.find_all('a', href=re.compile(r'/pubmed/\d+'))
    if citations:
        pubmed_id = citations[0].get_text(strip=True)
        return parse_pubmed(pubmed_id)
    else:
        return "No PMID found for GEO ID"

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <pubmed_id_or_geo_id>")
        sys.exit(1)

    input_id = sys.argv[1]
    if input_id.startswith("GSE"):
        result = fetch_geo_id(input_id)
    else:
        data_handle, full_reference = parse_pubmed(input_id)
        print(data_handle)
        print(full_reference)

import requests
from bs4 import BeautifulSoup

def get_synonyms(word):
    url = f"https://www.merriam-webster.com/thesaurus/{word}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    synonyms = []
    for a in soup.find_all("a", class_="thes-list-content"):
        if a.text != word:
            synonyms.append(a.text)
    
    return synonyms


print (get_synonyms('car'))
print (get_synonyms('hair'))
print (get_synonyms('like'))
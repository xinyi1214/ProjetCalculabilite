import spacy
import re
from spacy.pipeline import EntityRuler
from spacy.language import Language

nlp = spacy.load("fr_core_news_sm")

ustensiles = []
#récupère une liste d'ustensiles
with open("Data/ustensilesCuisineSurInternet.txt", "r") as file2:
    for us in file2:
        us = us.split("\n")[0]
        ustensiles.append(us)

#le patterns qu'on va chercher
patterns1 = ["NOUN"]
patterns2 = ["NOUN NOUN"]
quantity = ["g", "ml", "kg", "cl", "cm", "l", "dl", "batons","grams", "kilograms", "lb", "lbs", "pounds", "c.à.s." ,"c.à.c." "grammes", "cs", "cc", "verre", "branche", "branches", "verres", "cuillères", "poignée", "poignées", "gousse", "gousses", "cuillère", "sachet", "sachets", "rouleau", "rouleaux", "pincée", "pincées", "litre", "litres", "filaments", "filament", "cubes", "cube", "paquet", "paquets", "boîte", "boîtes", "brique", "briques", "pot","pots", "feuilles", "feuille", "morceau", "morceaux", "zestes", "zeste", "boules", "boule", "tête", "têtes", "bol", "bols", "bouquet", "bouquets", "portion", "portions"]
re_patterns1 = [" ".join(["(\w+)_!"+pos for pos in p.split()]) for p in patterns1]
re_patterns2 = [" ".join(["(\w+)_!"+pos for pos in p.split()]) for p in patterns2]
matches1 = []
matches2 = []
doc2 = []
with open("Data/ingredient.txt", "r") as file:
    for ingre in file:
        ingre = ingre.split("(")[0]
        file = nlp(ingre)
        text_pos = " ".join([token.text+"_!"+token.pos_ for token in file if token.text not in quantity])
        for i, pattern in enumerate(re_patterns1):
            for result in re.findall(pattern, text_pos):       
                matches1.append("".join(result))
        for i, pattern in enumerate(re_patterns2):
            for result in re.findall(pattern, text_pos):
                matches2.append(" ".join(result))

adj = []
for item in matches2:
    adj.append(item.split(" ")[1])

ingredients1 = []
for ingre in matches1:
    if ingre not in adj and ingre not in quantity and ingre not in ustensiles:
        ingredients1.append(ingre)
ingredients1 = list(set(ingredients1))
with open ("IngredientParSpacy.txt", "w") as file3:
    for item in ingredients1:
        file3.write(item)
        file3.write("\n")


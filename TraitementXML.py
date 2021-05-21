"""
consignes:
entrée: le dossier qui contient les fichiers xml de recettes culinaires
résultat : deux fichiers txt, un fichier txt de tous les ingrédients, un fichier txt de tous les étapes de recettes
"""

import xml.dom.minidom as xmldom
import glob
import re

recettes = []
recettes2 = []
ingredients = []
for file in glob.glob('2014/*.xml'): #parcouris tous les fichier xml
	xml_file = xmldom.parse(file) #parser xml 
	eles = xml_file.documentElement #cherche tous les éléments dans le fichier
	ingre = eles.getElementsByTagName("p")
	for i in range(len(ingre)): #un boucle pour parcourir tous les "p"
		ingredient = ingre[i].firstChild.data #récupère le contenu d'ingrédients
		ingredients.append(ingredient) #une liste de tous les ingrédients
	text = eles.getElementsByTagName("preparation")[0].firstChild.wholeText #récupère le contenu des préparations
	recettes.append(text) #une liste de tous les préparations de tous les recettes


for texte in recettes:
	texte = re.sub("\n"," ", texte) #supprime les lignes sautées dans une recette, du coup chaque ligne représente une recette
	recettes2.append(texte) 

#écrire le résultat de préparation dans un fichier txt
with open ("corpus_recettes.txt", "w") as file:
	for texte in recettes2:
		file.write(texte)
		file.write("\n")
#écrire le résultat d'ingrédeint dans un fichier txt
with open ("corpus_ingredient.txt", "w") as file2:
	for texte in ingredients:
		file2.write(texte)
		file2.write("\n")



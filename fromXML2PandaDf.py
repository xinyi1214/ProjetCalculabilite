import os
import pandas as pd
from xml.etree import ElementTree
import glob
import json


#Lecture d'un fichier XML
# renvoie un dictionnaire contenant:
# le titre, les ingredients et la preparation
def convertXMLToList(full_file):
    
    dom = ElementTree.parse(full_file)
    recetteXML = dom.getroot()
    
    #attribut
    attrib = recetteXML.attrib
    attrib = attrib['id']
    
    
    #titre 
    #♥title = recetteXML.find('titre').text
    titre = recetteXML.findall('titre')
    titre = titre[0].text
    #cout
    cout = recetteXML.findall('cout')
    cout = cout[0].text
    #niveau
    niveau = recetteXML.findall('niveau')
    niveau = niveau[0].text
    
    #preparation
    preparationXML = recetteXML.findall('preparation')
    preparation = preparationXML[0].text.strip()
    
    
    # ingredients    
    ingredientP = dom.findall('ingredients/p')
    ingredients = []
    for ing in ingredientP:
        #print(ing.text)
        ingredients.append(ing.text)
        
    dicRecette = {"titre": titre, "preparation" :preparation, "ingredients" : ingredients }
    
        
    return dicRecette 




# renvoi une liste de tous les fichiers XML présent 
# dans le repertoire dont le chemin est : path_to_target
def getListOfXMLFiles(path_to_target):
    path_to_file_list = glob.glob(path_to_target + '*xml' )    
    
    file_list = [i.split('/')[-1] for i in path_to_file_list]
    
    
    fileNameList = [i.split('\\')[-1] for i in file_list]
    
    return file_list


def writeJsonFile(data, outputFileName):
    with open(outputFileName, "a",encoding="utf-8") as f:
        json.dump(data, f, indent=3 ,ensure_ascii=False)
        
def readJsonFile(fileName):
    with open(fileName, "r") as jsonFile:
        data = json.load(jsonFile)   
        return data

# lemmatized lowercase tokens    
def preprocess_token(token):
    # Reduce token to its lowercase lemma form
    return token.lemma_.strip().lower()





if __name__ == "__main__":
   
   
    # récupérer la liste des noms des fichiers XML avec le chemin
    path_to_target = './dataRaw/'    
    fileNameList = getListOfXMLFiles(path_to_target)
    
    # parcourir tous les fichiers et les mettre sous la forme d'un dictionnaire
    # boucler sur les recettes
    # ajouter le dict de recette à cette liste : dicRecette
    dicRecette = []
    for file in fileNameList:
          dicRecette.append(convertXMLToList(file))
    
    # enregistrer  l ensembre des recettes dans un json    
    writeJsonFile(dicRecette, "jsonCorpusRecettes.json")
  
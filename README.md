# ProjetCalculabilite

**Codes**
|  Codes  |  Input  |  Output  |  Commentaires  |
|  ----  | ----  |  ----  |  ----  |
|  TraitementXML.py  |  2014/.*xml  |  ingredient.txt, recettes.txt  |  Code qui lit l’ensemble des fichiers XML de notre corpus et extrait l’éléments ingredient et l’élément preparation  |
|  ExtractionIngredient.py |  recettes.txt, ustensilesCuisineSurInternet.txt |  IngredientParSpacy.txt  |  Code qui lit l’ensemble de la partie préparation et extrait les ingrédients  |
|  AjoutEntiteNommee+CalculComplexite.ipynb  |  listIngredientToSave.txt ou IngredientParSpacy.txt, listVerbeToSave.txt ou verbesCusineSurInternet.txt, ustensilesCuisine.txt, 2014/.*xml  |  Un dictionnaire de la corrélation entre le niveau de la recette et la complexité en tempsUn dictionnaire de la corrélation entre le niveau de la recette et la complexité en temps  |  Ce Notebook contient deux partie: la première partie est de l’ajout de trois entités nommées : INGREDIENT, VERBE et Ustensile dans le pipeline. La deuxième partie est du calcul de la complexité en temps et la corrélation entre le niveau de la recette et la complexité en temps (ATTENTION: pour AjoutEntiteNommee, il existe deux versions d’input: soit listIngredientToSave.txt, listVerbeToSave.txt, ustensilesCusine.txt, soit IngredientParSpacy.txt, verbesCusineInternet.txt, ustensilesCusine.txt)  |

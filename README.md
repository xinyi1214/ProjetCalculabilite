# ProjetCalculabilite

## Membres
Shererazade NIBEB, Xinyi SHEN

## Résume du projet

Nous présentons ici un modèle de calcul de complexité en temps des programmes informatiques appliqué aux recettes de cuisine en français. Dans un  premier temps, afin de bien identifier les différents éléments des recettes utiles au calcul – et sans lesquels la complexité ne peut être calculée – à savoir les ingrédients, les ustensiles de cuisine utilisés et les opérations culinaires effectuées, nous avons imbriqué deux méthodes : une consistant à extraire du vocabulaire relatif au domaine à travers internet et quelques fonctionnalités de spaCy pour constituer un premier lexique et une seconde méthode qui tente d’identifier ces éléments de la façon la plus automatisée possible grâce à un plongement des mots et à un algorithme non supervisé de type clustering. Ces étapes permettent la création de nouvelles entités nommées françaises par rapport aux recettes : VerbeCulinaie, Ingredient et Ustensile et donc l’apprentissage d’un modèle. A partir de la reconnaissance des entités nommées, nous avons calculé pour chaque recette sa complexité en temps et avons fait la corrélation entre la complexité en temps et le niveau difficulté de chaque recette.

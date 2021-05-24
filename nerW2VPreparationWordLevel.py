# -*- coding: utf-8 -*-
"""

        "Machine learning with word embeddings at word level"
        
Ce fichier étudie la partie "preparation"

Nous avons cherché à mettre en place une méthode semi-automatique de détection
des ingrédients/ustenciles/process afin de simplifier, rendre l'annotation optimale (dans le sens où il est impossible d'annoter 
un nombre significatif de recettes: cette méthode rendra alors l'annotation optimale) tout en la diminuant au maximum.

L'idée ici était d'expérimenter si l'embedding et la clusterisation pouvait permettre de regrouper les tokens représentants 
les ingrédients, les ustenciles et les process dans des clusters bien séparés.


Pour mettre en place ces clusters nous avons opté pour:
    - un prétraitement du texte: normalisation,  extraction des ponctuations, stemmatization (équivalent à lemmatisation pour enlever les doublons
dus au pluriel ou à la conjugaison), élimination des stopswords, de qq ajdjectifs et adverbes, des quantités..
    - un embedding (de type bow) à l'echelle des mots sur le texte prétraité,   
    - l'application ensuite d'un apprentissage non-supervisé de type KMeans,
    - une estimation des résultats via un score, un silhouette_score
    - Une visualisation des résultats via une PCA en 2 dimensions et T-SNE (N.B.: nous n'avons pas utilisé la PCA pour accélérer l'apprentissage)
    
Nous avons alors analysé les clusters pour dégager les tendances.

La conclusion: bien que l'embedding rapproche parfois les verbes ou ustenciles, il reste cependant beaucoup de mélange.
L'embedding à ce stade à l'échelle des mots n'est pas concluant, il faudrait repartir sur une méthode qui considère la syntaxe
des phrases en plus.

Nous ne sommes pas arrivées à l'étape d'annotation ou de création du dataset d'entrainement avec cette étude. 

"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
import csv
from nltk.tokenize import word_tokenize


#### Load Data
df =  pd.read_csv('./Data/recettesParPhrases2.txt')
sentences = df[df["Instructions"].notna()]
sentences =  sentences.values.tolist()

sentencesList=[]
for i in range(len(sentences)):
    sentencesList.append(word_tokenize(sentences[i][0]))
    
    
    
#### Analyse du corpus #####


# On énumère les tokens présents et on regarde leur fréquence
list_Mot_Corpus=[]
for i,sent in enumerate(sentencesList):
    #print(sent)
    for j in range(len(sent)):        
        #print(sent[j])
        list_Mot_Corpus.append(sent[j])
        
print("Nombre de mots dans le corpus avec répétition: ", len(list_Mot_Corpus),"\n")        
# Le nombre de mots adns le corpus avec répétition est : 1 334 969 
        
from collections import Counter

# statistical analysis on the text
word_freq = Counter(list_Mot_Corpus)
print("Nombre de mots dans le corpus sans répétition: ", len(word_freq),"\n")  

## 100 commonly occurring words with their frequencies
common_words = word_freq.most_common(100)
print("Les 100 mots les plus fréquents dans le corpus : \n\n", common_words,"\n")  

# Unique words
unique_words = [word for (word, freq) in word_freq.items() if freq <= 5]
print("Le nombre de mots et les mots présents moins de 5 fois dans le corpus : ",len(unique_words), "\n\n", unique_words,"\n")  


# on fait le choix de faire passer un word2vec avec  min_count=10 pour diminuer les bruits

##### Fin Analyse du corpus ######    
    
#######################################################################
#  Entrainements: Embedding de type skipgram & KMeans  & PCA & TSNE   #
#######################################################################      


# Input list of sentences: word level

model = Word2Vec(sentencesList,sg=0,hs=1, min_count=10)#,epochs=10 )


# Nous allons tester le KMEans de la bibliothèque NLTK puis celui de Sklearn
# Nous fournissons le word embeddings au clustering algorithm 
# k Means est un des algorithmes non supervisés le plus populaire 
# L'idée est de trouver des sous-ensembles intéressants. 

X = model.wv[model.wv.key_to_index]
# model.wv[model.wv.key_to_index] equivaut à model.wv.vectors
print("Le format de X est : ",X.shape, "\n\n")  

# quelques tests
# => Suite à divereses exprériences nous avons enlevé les stopwords, les ponctuations, qq adjectifs, adverbes..
 
print("Quelques exmples : \n\n") 
print("Les tokens les plus proches de oignon : \n\n",model.wv.most_similar('oignon'), "\n\n") 
print("Les tokens les plus proches de aubergine : \n\n",model.wv.most_similar('aubergin'), "\n\n") 
print("Les tokens les plus proches de coupelle : \n\n",model.wv.most_similar('coupel'), "\n\n") 
print("Les tokens les plus proches de découper : \n\n",model.wv.most_similar('découp'), "\n\n") 
print("Les tokens les plus proches de laver : \n\n",model.wv.most_similar('lav'), "\n\n") 
print("Les tokens les plus proches de assiette : \n\n",model.wv.most_similar('assiet'), "\n\n") 
print("Les tokens les plus proches de dresser : \n\n",model.wv.most_similar('dress'), "\n\n")   
print("Les tokens les plus proches de eau : \n\n",model.wv.most_similar('eau'), "\n\n")      



# On peut maintenant pluggué nos données X à l'algorithme de clustering
# D'abord la méthode nltk

from nltk.cluster import KMeansClusterer
import nltk
NUM_CLUSTERS=3
# we use cosine distance to cluster our data.
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=5)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print (assigned_clusters)


# After we got cluster results we can associate each word with the cluster that it got assigned to:
# pour atteindre le vocab ( nouvelle méthode : key_to_index )
# https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4

words = list(model.wv.key_to_index)
for i, word in enumerate(words):
    if i > 150:
        break
    print (word + ":" + str(assigned_clusters[i]))
    


# autre KMeans de Sklearn

from sklearn.cluster import KMeans
from sklearn import metrics

model_sk = KMeans(n_clusters=3)
model_sk.fit(X)
model_sk.predict(X)

labels = model_sk.labels_
centroids = model_sk.cluster_centers_


print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
  
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (model_sk.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
print ("Silhouette_score: ")
print (silhouette_score) 

plt.scatter(X[:,0], X[:,1], c=model_sk.predict(X))
plt.scatter(model_sk.cluster_centers_[:,0], model_sk.cluster_centers_[:,1], c='r')


# Pour la visu on peut utiliser t-sne ou PCA

##################   T-SNE   ##################

import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
 
#model_tsne = TSNE(n_components=3, random_state=0)
model_tsne = TSNE(perplexity=2, n_components=2, init='pca')#, random_state=42)
np.set_printoptions(suppress=True)

Y=model_tsne.fit_transform(X)
 
 
plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters)
plt.title('TSNE [n_components=2]')

##################   FIN  T-SNE    ##################

##################   PCA   ##################

## PCA 
# réduction de la dimension du dataset tout en préservant au maximum 
# la variance de nos données pour obtenir la projection qui soit
# la plus fidèle possible aux données

# on s'intéresse à la visu et non à l'accélération de l'apprentissage
# donc n_components = 2 pour visu 2D



from sklearn.decomposition import PCA

X.shape 

# => projection de 100 variables dans un espace 2D
# on crée donc un model PCa dont le nb de composante = 2

model_pca = PCA(n_components=2)

# chaque composante comtient 100 valeurs
#model.components_.shape

#On entraine le modèle
X_reduced = model_pca.fit_transform(X) # shape: (7638,2)
#model.fit_transform(X).shape
# visualisation des clusters
# organisation dans un espace 2D des échantillons de données des recettes
# assigned_clusters = étiquettes de nos échantillons 0,1,2
plt.scatter(X_reduced[:,0], X_reduced[:,1],c=assigned_clusters)# c nos label
#plt.text(X_reduced[:,0], X_reduced[:,1], words[:])
plt.colorbar()
plt.title('PCA [n_components=2]')
# pas sur que çà soit bon... au vu des résultats des variances des composantes
variances = model_pca.explained_variance_ratio_

##################   FIN PCA   ##################

###########################################################################
#  Entrainements: FIN Embedding de type skipgram & KMeans  & PCA & TSNE   #
###########################################################################  






###########################################################################
############################        PLOT          #########################
###########################################################################

#pip install adjustText
# pour ajuster le texte 
from adjustText import adjust_text

# Choose a colormap
import matplotlib.colors as mcolors
import matplotlib.cm as cm
colormap = cm.viridis

# Which parameter should we represent with color?
colorparams = assigned_clusters
normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

# plt.figure()
# plt.xlim(-2, 8)
# plt.ylim(-4, 6)
# for i in range(100):
#     plt.text(X_reduced[i,0], X_reduced[i,1], str(words[i]))
    
    
    
plt.figure()
plt.xlim(-4, 8)
plt.ylim(-6, 8)
   
adjust=[]
for i in range(50):
    plt.scatter(X_reduced[i,0], X_reduced[i,1],c=colormap(normalize(assigned_clusters[i])))
    adjust.append(plt.text(X_reduced[i,0], X_reduced[i,1], str(words[i])))
adjust_text(adjust)
plt.title('Echantillon de 50 données')
plt.show()    


    
vectors = X
val0 = [x for idx, x in enumerate(vectors) if assigned_clusters[idx]==0]
ind0 = [idx for idx, x in enumerate(vectors) if assigned_clusters[idx]==0]
word0 = [words[idx] for idx, x in enumerate(vectors) if assigned_clusters[idx]==0]
x0 = np.array([x[0] for idx, x in enumerate(vectors) if assigned_clusters[idx]==0])
y0 = np.array([x[1] for idx, x in enumerate(vectors) if assigned_clusters[idx]==0])
label0 = [0 for idx, x in enumerate(vectors) if assigned_clusters[idx]==0]



word1 = [words[idx] for idx, x in enumerate(vectors) if assigned_clusters[idx]==1]
val1 = [x for idx, x in enumerate(vectors) if assigned_clusters[idx]==1]
ind1 = [idx for idx, x in enumerate(vectors) if assigned_clusters[idx]==1]
x1 = np.array([x[0] for idx, x in enumerate(vectors) if assigned_clusters[idx]==1])
y1 = np.array([x[1] for idx, x in enumerate(vectors) if assigned_clusters[idx]==1])
label1 = [1 for idx, x in enumerate(vectors) if assigned_clusters[idx]==1]


word2 = [words[idx] for idx, x in enumerate(vectors) if assigned_clusters[idx]==2]
val2 = [x for idx, x in enumerate(vectors) if assigned_clusters[idx]==2]
ind2 = [idx for idx, x in enumerate(vectors) if assigned_clusters[idx]==2]
x2 = np.array([x[0] for idx, x in enumerate(vectors) if assigned_clusters[idx]==2])
y2 = np.array([x[1] for idx, x in enumerate(vectors) if assigned_clusters[idx]==2])
label2 = [2 for idx, x in enumerate(vectors) if assigned_clusters[idx]==2]



plt.figure()
plt.xlim(-4, 8)
plt.ylim(-6, 8)
   
adjust=[]
for i in range(50):
    plt.scatter(x0[i], y0[i],c=colormap(normalize(label0[i])))
    adjust.append(plt.text(x0[i], y0[i], str(word0[i])))
adjust_text(adjust)
plt.title('Echantillon de 50 données')
plt.show()    

adjust=[]
for i in range(50):
    plt.scatter(x1[i], y1[i],c=colormap(normalize(label1[i])))
    adjust.append(plt.text(x1[i], y1[i], str(word1[i])))
adjust_text(adjust)
plt.title('Echantillon de 50 données')
plt.show()

adjust=[]
for i in range(50):
    plt.scatter(x0[i], y0[i],c=colormap(normalize(label0[i])))
    adjust.append(plt.text(x0[i], y0[i], str(word0[i])))
adjust_text(adjust)
plt.title('Echantillon de 50 données')
plt.show()



###########################################################################
###################             FIN PLOT                  #################
###########################################################################
    
############ Analyse des clusters  ############

# On énumère les tokens présents et on regarde leur fréquence

cluster0 = [ len(listElem) for listElem in word0]
print("Nombre d'éléments du cluster 0: ", len(cluster0), "\n\n" )


cluster1 = [ len(listElem) for listElem in word1]
print("Nombre d'éléments  du cluster 1: ", len(cluster1), "\n\n" )


cluster2 = [ len(listElem) for listElem in word2]
print("Nombre d'éléments  du cluster 2: ", len(cluster2), "\n\n" )


# -*- coding: utf-8 -*-
"""

        "Machine learning with word embeddings at sentence level"
        
Ce fichier étudie la partie "ingrédients"

Nous avons cherché à mettre en place une méthode semi-automatique de détection
des ingrédients afin de simplifier, rendre l'annotation optimale (dans le sens où il est impossible d'annoter 
un nombre significatif de recettes: cette méthode rendra alors l'annotation optimale) tout en la diminuant au maximum.

L'idée ici est que les phrases qui décrivent/expriment les quantités, les ingrédients '
répondent à un pattern d'une façon générale.

Pour mettre en évidence ces patterns/clusters nous avons opté pour:
    - un prétraitement du texte: normalisation,  extraction des ponctuations, stemmatization (équivalent à lemmatisation pour enlever les doublons
dus au pluriel),
    - un embedding (de type skipgram) à l'echelle des phrases sur le texte prétraité,
    - l'utilisation de la méthode Elbow afin d'identifier/sélectionner le nombre de cluster qui répondrait le mieux à notre corpus,
    - l'application ensuite d'un apprentissage non-supervisé de type KMeans,
    - une estimation des résultats via un score, un silhouette_score
    - Une visualisation des résultats via une PCA en 2 dimensions (N.B.: nous n'avons pas utilisé la PCA pour accélérer l'apprentissage)
    
Nous avons alors analysé (du moins commencé car c'est un véritable travail de foumi) les clusters pour dégager les tendances.
                          
L'idée étant de sélectionner un pourcentage de phrases répondant à certains patterns de chaque cluster sur lesquelles une annotation manuelle
sera alors ensuite menée. Finalement nous n'avons pas annotée manuellement mais avons utilisée la liste créee grâce à ces clusters et 
l'avons injecté dans une EntityRuler ce qui a rendu la création de donner pour le train et le test simple et sur tout le corpus.

"""



import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
import csv
#pip uninstall nltk
import nltk 

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
#!pip install FrenchLefffLemmatizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from collections import Counter


#### Load Data

# un code Python a crée le fichier "ingredient.txt"
# on y retrouve tous les ingrédients de toutes les recettes du corpus
# on charge le fichier pour récupérer les différents ingrédients
f = open("./ingredient.txt",encoding="utf-8")
lines = f.readlines()
tmp = lines
#len(tmp) # nombre de lignes


#######################################################################
#                       PRETAITEMENT                                  #
#######################################################################

listIngredients=[]
for i in range(len(tmp)):  
    if tmp[i] != '':
        listIngredients.append(tmp[i].replace('.',' . ').replace('-',' - ').replace('=',' = ').replace('/',' / ').replace('+',' + ').replace('!',' ! ').replace('?',' ? ').replace('\'','  ').replace('\n','').replace('(',' ( ').replace(')',' ) ').replace('"',' ').lower())


# Normalisation, tokenisation
def initialNormalizeAndTokenize(phrase_test):

    word_text =[]
    word_text = nltk.word_tokenize(phrase_test)

    # on peut enlever les ponctuations ce n est pas une analyse de sentiments.. 
    ponc ="!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~»«…\ '’"
    table = str.maketrans("","", ponc)

    # convert to lower case
    stripped = [w.translate(table).lower() for w in word_text]   
        
    # remove remaining tokens that are not alphabetic
    #stripped = [word for word in stripped if word.isalpha()]
    
    assembled =" ".join(stripped)
    assembled = assembled.rstrip()
    word_text = nltk.word_tokenize(assembled)    
    return word_text


# Lemmatiseur en français [stemmatization ici]
def initialLemmatizer(word_text):

    # lemmatizer = FrenchLefffLemmatizer()
    # lemWords = [lemmatizer.lemmatize(word) for word in word_text]
    stemmer = FrenchStemmer()
    lemWords = [stemmer.stem(word) for word in word_text]

    return lemWords 

# Methode qui appelle qles autres méthodes de prétraitement
def preprocessingUsingLem(text):
    
    textLem=[]
    for i,val in enumerate(text):
        phrase_test=val 
    
        # tokenization, suppression ponctuation et normalisation
        word_text = initialNormalizeAndTokenize(phrase_test)
    
        # II) Lemmatisation
        lemWords = initialLemmatizer(word_text)
     
        assembled =" ".join(lemWords)
        assembled = assembled.rstrip()
        textLem.append(assembled)
    
    return textLem


IngrNorm = preprocessingUsingLem(listIngredients)


sentencesList=[]
for i in range(len(IngrNorm)):
    tmp = word_tokenize(IngrNorm[i])
    if tmp != "":
        sentencesList.append(tmp)
    
#######################################################################
#                       FIN PRETAITEMENT                              #
#######################################################################    
    

#######################################################################
#                       PREMIERE ANALYSE DU CORPUS                    #
#######################################################################  

# On énumère les tokens présents et on regarde leur fréquence
list_Mot_Corpus=[]
for i,sent in enumerate(sentencesList):
    #print(sent)
    for j in range(len(sent)):        
        #print(sent[j])
        list_Mot_Corpus.append(sent[j])
        
print("Nombre de mots dans le corpus avec répétition: ", len(list_Mot_Corpus),"\n")        


# statistical analysis on the text
word_freq = Counter(list_Mot_Corpus)
print("Nombre de mots dans le corpus sans répétition: ", len(word_freq),"\n")  

## 100 commonly occurring words with their frequencies
common_words = word_freq.most_common(100)
print("Les 100 mots les plus fréquents dans le corpus : \n\n", common_words,"\n")  

# Unique words
unique_words = [word for (word, freq) in word_freq.items() if freq <= 5] 
print("Le nombre de mots et les mots présents moins de 5 fois dans le corpus : ",len(unique_words), "\n\n", unique_words,"\n")  


#######################################################################
#                   FIN PREMIERE ANALYSE DU CORPUS                    #
#######################################################################    
    
    
#######################################################################
#  Entrainements: Embedding de type skipgram & Elbow Method & KMeans  #
#######################################################################  

# Word Embredding (word level)
# Skip-Gram Version

model = Word2Vec(sentencesList,sg=1,hs=1, min_count=1)


print("Quelques exmples de similarité du vocabulaire : \n\n") 
print("Les tokens les plus proches de oignon : \n\n",model.wv.most_similar('oignon'), "\n\n") 
print("Les tokens les plus proches de aubergine : \n\n",model.wv.most_similar('farin'), "\n\n") 
print("Les tokens les plus proches de coupelle : \n\n",model.wv.most_similar('pain'), "\n\n") 



######### Clusterisation à l'echelle de la phrase: on fait une moyenne de tous les embedding word de la phrase  #######
######### Somme sur tous les mots puis divisé par el nombre de mots dans la phrase                              #######
######### Methodes pour filtrer le corpus de tout vocabulaire non présent dans le dictionnaire word2vec    ############
######### Et calcul de la moyenne sur le word2vec (car nous n'utilisons pas doc2vec)                        ############


def sent_vectorizer(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in list(word2vec_model.wv.key_to_index)] 
    return np.mean(word2vec_model.wv[doc], axis=0)

def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in list(word2vec_model.wv.key_to_index) for word in doc)

def filter_docs(corpus, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return corpus


sentencesList = filter_docs(sentencesList,  lambda doc: has_vector_representation(model, doc))

X =[]
#look up each doc in model
for doc in sentencesList: 
    X.append(sent_vectorizer(model, doc))
    
X = np.array(X) #list to array    

########################################################
#########  FIN  embedding à l'échelle de la phrase  ####
########################################################


# On va pouvoir maintenant plugger nos données X à un algorithme de clustering: K-Means.

    
# Elbow Method

from sklearn.cluster import KMeans

inertia = []
K_range = range(1,20)# 80)
for k in K_range:
    model_Elbow = KMeans(n_clusters=k)
    Y=model_Elbow.fit(X)
    inertia.append(Y.inertia_)

plt.plot(K_range, inertia)
plt.xlabel('nombre de clusters')
plt.ylabel('Cout du modele (Inertia)')
    
# Fin Elbow Method

# KMeans de Sklearn



from sklearn import metrics


model_sk = KMeans(n_clusters=7)
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

# plt.scatter(X[:,0], X[:,1], c=(model_sk.predict(X.astype('float'))))
# plt.scatter(model_sk.cluster_centers_[:,0], model_sk.cluster_centers_[:,1], c='r')


##########################################################################
#  FIN Entrainements: Embedding de type skipgram & Elbow Method & KMeans #
##########################################################################  



##################   PCA   ##################
 
# réduction de la dimension du dataset tout en préservant au maximum 
# la variance de nos données pour obtenir la projection qui soit
# la plus fidèle possible aux données
# on s'intéresse à la visu et non à l'accélération de l'apprentissage
# donc n_components = 2 pour visu 2D

from sklearn.decomposition import PCA


# => projection de 100 variables dans un espace 2D
# on crée donc un model PCa dont le nb de composante = 2
model_pca = PCA(n_components=2)

#On entraine le modèle
X_reduced = model_pca.fit_transform(X) 

# visualisation des clusters
# organisation dans un espace 2D des échantillons de données des recettes
# labels = étiquettes de nos échantillons 0,1,2,...
plt.scatter(X_reduced[:,0], X_reduced[:,1],c=labels)
plt.colorbar()

# Résultats des variances des composantes
variances = model_pca.explained_variance_ratio_

################## Fin PCA  ##################


##########################################################################   
###########              Analyse des clusters        #####################
##########################################################################    
    
vectors = X
val0 = [x for idx, x in enumerate(vectors) if labels[idx]==0]
ind0 = [idx for idx, x in enumerate(vectors) if labels[idx]==0]
word0 = [sentencesList[idx] for idx, x in enumerate(vectors) if labels[idx]==0]
x0 = np.array([x[0] for idx, x in enumerate(vectors) if labels[idx]==0])
y0 = np.array([x[1] for idx, x in enumerate(vectors) if labels[idx]==0])
label0 = [0 for idx, x in enumerate(vectors) if labels[idx]==0]



word1 = [sentencesList[idx] for idx, x in enumerate(vectors) if labels[idx]==1]
val1 = [x for idx, x in enumerate(vectors) if labels[idx]==1]
ind1 = [idx for idx, x in enumerate(vectors) if labels[idx]==1]
x1 = np.array([x[0] for idx, x in enumerate(vectors) if labels[idx]==1])
y1 = np.array([x[1] for idx, x in enumerate(vectors) if labels[idx]==1])
label1 = [1 for idx, x in enumerate(vectors) if labels[idx]==1]


word2 = [sentencesList[idx] for idx, x in enumerate(vectors) if labels[idx]==2]
val2 = [x for idx, x in enumerate(vectors) if labels[idx]==2]
ind2 = [idx for idx, x in enumerate(vectors) if labels[idx]==2]
x2 = np.array([x[0] for idx, x in enumerate(vectors) if labels[idx]==2])
y2 = np.array([x[1] for idx, x in enumerate(vectors) if labels[idx]==2])
label2 = [2 for idx, x in enumerate(vectors) if labels[idx]==2]

word3 = [sentencesList[idx] for idx, x in enumerate(vectors) if labels[idx]==3]
val3 = [x for idx, x in enumerate(vectors) if labels[idx]==3]
ind3 = [idx for idx, x in enumerate(vectors) if labels[idx]==3]
label3 = [3 for idx, x in enumerate(vectors) if labels[idx]==3]
x3 = np.array([x[0] for idx, x in enumerate(vectors) if labels[idx]==3])
y3 = np.array([x[1] for idx, x in enumerate(vectors) if labels[idx]==3])

word4 = [sentencesList[idx] for idx, x in enumerate(vectors) if labels[idx]==4]
val4 = [x for idx, x in enumerate(vectors) if labels[idx]==4]
ind4 = [idx for idx, x in enumerate(vectors) if labels[idx]==4]
label4 = [4 for idx, x in enumerate(vectors) if labels[idx]==4]
x4 = np.array([x[0] for idx, x in enumerate(vectors) if labels[idx]==4])
y4 = np.array([x[1] for idx, x in enumerate(vectors) if labels[idx]==4])

word5 = [sentencesList[idx] for idx, x in enumerate(vectors) if labels[idx]==5]
val5 = [x for idx, x in enumerate(vectors) if labels[idx]==5]
ind5 = [idx for idx, x in enumerate(vectors) if labels[idx]==5]
label5 = [5 for idx, x in enumerate(vectors) if labels[idx]==5]
x5 = np.array([x[0] for idx, x in enumerate(vectors) if labels[idx]==5])
y5 = np.array([x[1] for idx, x in enumerate(vectors) if labels[idx]==5])

word6 = [sentencesList[idx] for idx, x in enumerate(vectors) if labels[idx]==6]
val6 = [x for idx, x in enumerate(vectors) if labels[idx]==6]
ind6 = [idx for idx, x in enumerate(vectors) if labels[idx]==6]
label6 = [6 for idx, x in enumerate(vectors) if labels[idx]==6]
x6 = np.array([x[0] for idx, x in enumerate(vectors) if labels[idx]==6])
y6 = np.array([x[1] for idx, x in enumerate(vectors) if labels[idx]==6])




#pip install adjustText
# pour ajuster le texte 
from adjustText import adjust_text

# Choose a colormap
import matplotlib.colors as mcolors
import matplotlib.cm as cm
colormap = cm.viridis

# Which parameter should we represent with color?
colorparams = labels
normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

plt.figure()
plt.xlim(-0.2,0.5)
plt.ylim(-1, 0.5)

adjust=[]
for i in range(100,110):
    plt.scatter(x1[i], y1[i],c=colormap(normalize(label1[i])))
    adjust.append(plt.text(x1[i], y1[i], str(word1[i])))
adjust_text(adjust)
plt.title('Cluster1')
plt.show()

adjust=[]
for i in range(100,115):
    plt.scatter(x2[i], y2[i],c=colormap(normalize(label2[i])))
    adjust.append(plt.text(x2[i], y2[i], str(word2[i])))
adjust_text(adjust)
plt.title('Cluster2')
plt.show()

    
    
adjust=[]
for i in range(200,235):
    plt.scatter(x0[i], y0[i],c=colormap(normalize(label0[i])))
    adjust.append(plt.text(x0[i], y0[i], str(word0[i])))
adjust_text(adjust)  
plt.title('Cluster0')
plt.show()  
    

adjust=[]
for i in range(200,235):
    plt.scatter(x3[i], y3[i],c=colormap(normalize(label3[i])))
    adjust.append(plt.text(x3[i], y3[i], str(word3[i])))
adjust_text(adjust)  
plt.title('Cluster3')
plt.show()  

adjust=[]
for i in range(200,235):
    plt.scatter(x4[i], y4[i],c=colormap(normalize(label4[i])))
    adjust.append(plt.text(x4[i], y4[i], str(word4[i])))
adjust_text(adjust)  
plt.title('Cluster4')
plt.show()                      
                         
adjust=[]
for i in range(200,235):
    plt.scatter(x5[i], y5[i],c=colormap(normalize(label5[i])))
    adjust.append(plt.text(x5[i], y5[i], str(word5[i])))
adjust_text(adjust)  
plt.title('Cluster5')
plt.show()  

adjust=[]
for i in range(200,235):
    plt.scatter(x6[i], y6[i],c=colormap(normalize(label6[i])))
    adjust.append(plt.text(x6[i], y6[i], str(word6[i])))
adjust_text(adjust)  
plt.title('Cluster6')
plt.show()  


adjust=[]
for i in range(10,30):
    plt.scatter(X_reduced[i,0], X_reduced[i,1],c=colormap(normalize(labels[i])))
    adjust.append(plt.text(X_reduced[i,0], X_reduced[i,1], str(sentencesList[i])))
adjust_text(adjust)    
plt.title('Echantillon')

# # Colorbar setup
# s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
# s_map.set_array(colorparams)





list(model.wv.key_to_index)
list(model.wv.key_to_index)
len(model.wv.key_to_index)
X.shape
len(sentencesList)


############ Analyse des clusters  ############

# On énumère les tokens présents et on regarde leur fréquence

cluster0 = [ len(listElem) for listElem in word0]
print("Nombre d'éléments de chaque phrase du cluster 0: ", len(cluster0), "la taille min: ", min(cluster0), "la taille max: ",max(cluster0),"\n\n" )
#Nombre d'éléments de chaque phrase du cluster 0:  21631 la taille min:  1 la taille max:  37 

cluster1 = [ len(listElem) for listElem in word1]
print("Nombre d'éléments de chaque phrase du cluster 1: ", len(cluster1), "la taille min: ", min(cluster1), "la taille max: ",max(cluster1),"\n\n" )
#Nombre d'éléments de chaque phrase du cluster 1:  46571 la taille min:  1 la taille max:  103 

cluster2 = [ len(listElem) for listElem in word2]
print("Nombre d'éléments de chaque phrase du cluster 2: ", len(cluster2), "la taille min: ", min(cluster2), "la taille max: ",max(cluster2),"\n\n" )
#Nombre d'éléments de chaque phrase du cluster 2:  42170 la taille min:  1 la taille max:  65 

cluster3 = [ len(listElem) for listElem in word3]
print("Nombre d'éléments de chaque phrase du cluster 3: ", len(cluster3), "la taille min: ", min(cluster3), "la taille max: ",max(cluster3),"\n\n" )
#Nombre d'éléments de chaque phrase du cluster 3:  9375 la taille min:  1 la taille max:  14 


cluster4 = [ len(listElem) for listElem in word4]
print("Nombre d'éléments de chaque phrase du cluster 4: ", len(cluster4), "la taille min: ", min(cluster4), "la taille max: ",max(cluster4),"\n\n" )
# Nombre d'éléments de chaque phrase du cluster 4:  11380 la taille min:  1 la taille max:  9 

cluster5 = [ len(listElem) for listElem in word5]
print("Nombre d'éléments de chaque phrase du cluster 5: ", len(cluster5), "la taille min: ", min(cluster5), "la taille max: ",max(cluster5),"\n\n" )
# Nombre d'éléments de chaque phrase du cluster 5:  13729 la taille min:  1 la taille max:  29 

cluster6 = [ len(listElem) for listElem in word6]
print("Nombre d'éléments de chaque phrase du cluster 6: ", len(cluster6), "la taille min: ", min(cluster6), "la taille max: ",max(cluster6),"\n\n" )
# Nombre d'éléments de chaque phrase du cluster 6:  45115 la taille min:  1 la taille max:  92 



# Nombre de phrases par clusters:
nbPhrasesParCluster =[len(cluster0),len(cluster1),len(cluster2),len(cluster3),len(cluster4),len(cluster5),len(cluster6)]
#nbPhrasesParCluster =[len(cluster4),len(cluster3),len(cluster1),len(cluster6),len(cluster5),len(cluster2),len(cluster0)]
nomdesClusters =['Cluster0','Cluster1','Cluster2','Cluster3','Cluster4','Cluster5','Cluster6']


c= [0, 1, 2, 3, 4, 5, 6]
plt.bar(nomdesClusters, nbPhrasesParCluster,color=colormap(normalize(c)))
plt.title('Nombre de phrases par cluster')
plt.show()
plt.show()




# statistical analysis on the text
word_freq0 = Counter(cluster0)
print("Nombre de différentes tailles des phrases dans le corpus | cluster0 : ", len(word_freq0),"\n")   
word_freq1 = Counter(cluster1)
print("Nombre de différentes tailles des phrases dans le corpus  | cluster1 : ", len(word_freq1),"\n")
word_freq2 = Counter(cluster2)
print("Nombre de différentes tailles des phrases dans le corpus  | cluster2  : ", len(word_freq2),"\n")
word_freq3 = Counter(cluster3)
print("Nombre de différentes tailles des phrases dans le corpus  | cluster3 : ", len(word_freq3),"\n")
word_freq4 = Counter(cluster4)
print("Nombre de différentes tailles des phrases dans le corpus  | cluster4  : ", len(word_freq4),"\n")
word_freq5 = Counter(cluster5)
print("Nombre de différentes tailles des phrases dans le corpus  | cluster5  : ", len(word_freq5),"\n")
word_freq6 = Counter(cluster6)
print("Nombre de différentes tailles des phrases dans le corpus  | cluster6  : ", len(word_freq6),"\n")



# Affcihe la taille des phrases et le nombre de phrases à cette dimension
names = list(word_freq0.keys())
values = list(word_freq0.values())

plt.bar(range(len(word_freq0)), values, tick_label=names)
plt.title('Cluster0')
plt.title('Cluster6')
plt.xlabel('Nombre de tokens')
plt.ylabel("Nombre de phrase")
plt.show()


names = list(word_freq1.keys())
values = list(word_freq1.values())
#names=names[1:10]
#values=values[1:10]
plt.bar(range(len(word_freq1)), values, tick_label=names)
#plt.bar(range(len(values)), values, tick_label=names)
plt.title('Cluster1')
#plt.title('Cluster2')
plt.show()

names = list(word_freq2.keys())
values = list(word_freq2.values())

plt.bar(range(len(word_freq2)), values, tick_label=names)
plt.title('Cluster2')
plt.title('Cluster5')
plt.show()

names = list(word_freq3.keys())
values = list(word_freq3.values())
#names = names[0:10]
#values = values[0:10]
plt.bar(range(len(word_freq3)), values, tick_label=names)
plt.bar(range(len(names)), values, tick_label=names)
plt.title('Cluster3')
#plt.title('Cluster1')
plt.show()

names = list(word_freq4.keys())
values = list(word_freq4.values())

plt.bar(range(len(word_freq4)), values, tick_label=names)
plt.title('Cluster4')
plt.show()

names = list(word_freq5.keys())
values = list(word_freq5.values())
plt.bar(range(len(word_freq5)), values, tick_label=names)
plt.title('Cluster5')
plt.title('Cluster4')
plt.xlabel('Nombre de tokens')
plt.ylabel("Nombre de phrase")
plt.show()

names = list(word_freq6.keys())
values = list(word_freq6.values())

plt.bar(range(len(word_freq6)), values, tick_label=names)
plt.title('Cluster6')
plt.title('Cluster3')
plt.xlabel('Nombre de tokens')
plt.ylabel("Nombre de phrase")
plt.show()



# Pour récupérer les quantités:
[ listElem[0] for listElem in word3 if len(listElem) == 2 and not(listElem[0].isalpha())]    


listIngredientsToSave=[]

#### Cluster 0 : extraction de qq ingrédients
excluList=list(set(['autr','liquid','grillad','mixeur','papillot','moulin','li','ingrédientss','parfum','cocott','assiet','wok','eventuel','mousseux','neutr','calibr','telefono','centimetr','briq','pav','réhydrat','clair','farc','bouchon','amer','pour','min','regl','quatr','se','dl','de','douc','el','tripl','rempl','cà','parur','patidoux','concass','couteau','gel','viv','mix','gît','couch','cher','bain','fix','pelur','tablet','musiqu','assiet','arrier','rich','strong','robot','copeau','fumet','copeau','bomb','refroid','ex','dégerm','bloc','dilu','précuit','verr','vit','boir','vingtain','ne','étrill','gaz','maximum','concentr','tabl','tub','cas','seul','pont','volum','tant','vi','gratin','fouet','saveur','concass','debr','empereur','écheleur','désydrat','garn','avec','or','divers','moulu','du','càc','cisel','plat','vapeur','cc','brindill','soupçon','poignet','racin','queu','goutt','capsul','bulb','pluch','rameau','en','pinc','dos','beaucoup','plutôt','encor','ménag','fluid','pré','cluster','congel','poid','trompet','onctueux','minimun','déjà','brut','boit','liss','conserv','lu','oui','couvertur','spécial','n','ménag','point','décortiqu','trois','gramm','lingot','boulanger','cuit','light','gourmand','frâîch','rapp','préalabl','ramoll','températur','format','soit','complet','royal','dentel','allegépour','t','voir','râpépour','extra','boulang','clarifi','alumet','patissi','condens','dessert','court','wok','litr','un','fr','sélect','caisset','chaudepour','feuill','spiral','tig','m','adapt','glaçag','pic','saut','idé','caramélis','spatul','rien','tour','rest','barquet','matériel','tajin','prépar','simpl','vid','bouch','sauteux','granul','on','gazeux','fenêtr','march','épicer','veil','naturel','croût','réserv','cuill','réserv','film','ador','moelleux','bouteil','végétarien','naturel','gourmandis','équivalent','crémeux','smart','faut','voulu','souhait','chacun','égoutt','veut','minut','part','villageois','ramequin','ordinair','person','émiet','pot','kilos','besoin','indispens','individuel','obligatoir','conven','aiment','douzain','assaison','dorur','aim','finit','dégust','classiqu','fluidifi','person','auss','intérieur','supermarch','fourrag','intérieur','surfac','person','servic','désir','égal','recet','etc','peau','préfer','une','besoin','vos','comestibl','soigneux','j','doubl','façon','ma','est','il','dégraiss','alsacien','aérosol','four','paris','désoss','filtr','conseil','bas','déroul','alimentair','lill','dessous','fort','mond','déglac','brésilien','solid','nettoye','dessus','mexicain','assiet','guadeloup','fourr','présent','verrin','quinzain','h','essor','rouleau','désépaiss','goûut','italien','assidat','decor','avez','serv','exempl','gr','méxicain','sachet','dizain','prêt','exempl','commerc','plaqu','concass','coeur','feu','four''nid','soyeux','précuisson','option','fum','accompagn','aluminium','gout','paquet','leg','cuisson','écaill','couvercl','import','garnitur','poign','nécessair','accompagn','possibl','envi','déco','traditionnel','anniversair','personnalis','emploi','feuillet','cuisson','déco','rectangulair','alcoolis','lyophilis','styl','rap','qui','choix','entre','tout','quelconqu','bois','décor','st','haut','aigr','quantit','fait','kg','par','typ','moin','sous','moin','que','dit','al','vos','sur','assort','quarti','parchemin','ingrédient','différent','si','maison','papi','cur','concentr','rond','pâtissi','trop','x','dem','oriental','ici','sinon','trop','selon','env','tres','aux','assez','bel','deux','oiseau','eminc','boît','morceau','bott','préférent','trait','espagn','boul','gouss','têt','cub','unit','plus','dé','eminc','taill','fendu','lav','falcut','déshydrat','pil','surgel','cm','bocal','grosseur','droit','tricolor','cisel','râp','pel','juteux','roug','piqu','couleur','cru','lamel','pay','hache','écras','breton','printemp','écras','éventuel','rondel','grill','nantais','menu','branch','thaï','découp','allong','goût','éminc','dur','confit','facult','grand','douc','pet','moyen','press','coup','grossi','gross','long','nouvel','nouveau','balnc','chinois','brun','fris','jeun','hach','hâch','sec','mix','tendr','épluch','fin','longu','•','sech','longu','chinois','s','moulu','gros','vert','gris','noir','entir','entier','beau','à','petit','crus','battu','battus','optionnel','poch','facult','moyen','battu','enti','','quelqu','dénoyaut','niçois','peu','blanc','hâch','chaud','du','provençal','goût','fort','effil','meilleur','culinair','cuiller','cuir','anciennn','épaiss','volont','environ','froid','bio','fraich','poêl','frir','épais','sech','hach','plat','fondu','moudr','tied','ancien','roux','vi','mélang','mêm','verr','de','d','en','l','des','à','et','fin','la','le','ou','fin','cl','mais','fair','ml','car','doux','vieux','fond','c','2c','mi','pur','pas','g','l','alleg','non','a','léger','bon','a','frais','non','votr','fraîch','pour','vous','vert','cinq','gros','noir','petit','natur','au','défaut','bien','grand','quatr','moiti']))


# phrases de dim 7 => 170 ingrédients
listIngredientsToSave +=  list(set([ listElem[5] for listElem in word0 if len(listElem) == 7 and listElem[5] not in excluList  and (listElem[5].isalpha())]))



# phrases de dim 8 => 199 ingérdients
#[ listElem[5:8] for listElem in word0 if len(listElem) == 8]
listIngredientsToSave += list(set([ listElem[5] for listElem in word0 if len(listElem) == 8 and listElem[5] not in excluList and listElem[5].isalpha()]))

# phrases de dim 6 => 261 ingérdients
# ['1', 'cuiller', 'à', 'soup', 'de', 'gingembr'],
listIngredientsToSave += list(set([ listElem[5] for listElem in word0 if len(listElem) == 6 and listElem[5] not in excluList and listElem[5].isalpha()]))


# phrases de dim 3 => 199 ingérdients

# si la phrase commence par un nombre => 17
# le nombre: [ listElem[0] for listElem in word0 if len(listElem) == 3 and not(listElem[0].isalpha())]
#['1l', 'd', 'eau'],
listIngredientsToSave += list(set([ listElem[2] for listElem in word0 if len(listElem) == 3 and listElem[2] not in excluList and not(listElem[0].isalpha())]))

# nb = 2 [ listElem for listElem in word0 if len(listElem) == 3 and not(listElem[0].isalpha())]
# ['36', 'biscuit', 'cuiller'],
# ['10', 'oliv', 'noir'],


listIngredientsToSave += list(set([ listElem[1] for listElem in word0 if len(listElem) == 3 and listElem[1] not in excluList  and (listElem[1].isalpha()) and not(listElem[0].isalpha())]))





# si la phrase ne commence pas par un nombre
# 54
listIngredientsToSave += list(set([ listElem[0] for listElem in word0 if len(listElem) == 3 and listElem[0] not in excluList and listElem[0].isalpha()]))

listIngredientsToSave +=list(set([ listElem[1] for listElem in word0 if len(listElem) == 3 and listElem[1] not in excluList and listElem[1].isalpha() and listElem[0].isalpha()]))
listIngredientsToSave +=list(set([ listElem[2] for listElem in word0 if len(listElem) == 3 and listElem[2] not in excluList and listElem[2].isalpha() and listElem[0].isalpha()]))

#### Cluster 3 : extraction de qq ingrédients


# phrases de dim 2 => 1 ingrédient
listIngredientsToSave += list(set([ listElem[1] for listElem in word6 if len(listElem) == 2 and listElem[1] not in excluList and not(listElem[0].isalpha())]))

# phrases de dim 3 => 7 ingrédient
listIngredientsToSave += list(set([ listElem[1] for listElem in word6 if len(listElem) == 3 and listElem[1] not in excluList and (listElem[1].isalpha()) and not(listElem[0].isalpha())]))


#### Cluster 4 : extraction de qq ingrédients 

# 26 ingr
listIngredientsToSave += list(set([ listElem[0] for listElem in word5 if len(listElem) == 3 and listElem[0] not in excluList]))

# 45 ingr
listIngredientsToSave += list(set([ listElem[2] for listElem in word5 if len(listElem) == 3 and listElem[2] not in excluList]))



#### Cluster 0 : extraction de qq ingrédients 
# quand on ne commence pas par un num: [ listElem for listElem in word4 if  (listElem[0].isalpha()) ]


listIngredientsToSave += list(set([ listElem[2] for listElem in word4 if len(listElem) == 3 and listElem[2] not in excluList and listElem[2].isalpha()]))

listIngredientsToSave += list(set([ listElem[3] for listElem in word4 if len(listElem) == 4 and listElem[3] not in excluList and listElem[0].isalpha() and listElem[3].isalpha()] ))

listIngredientsToSave += list(set([ listElem[1] for listElem in word4 if len(listElem) == 2 and listElem[1] not in excluList and listElem[1].isalpha()]))

listIngredientsToSave += list(set([ listElem[1] for listElem in word4 if len(listElem) == 2 and listElem[1] not in excluList and listElem[1].isalpha()]))

listIngredientsToSave += list(set([ listElem[4] for listElem in word4 if len(listElem) == 5 and listElem[4] not in excluList and listElem[4].isalpha()]))

listIngredientsToSave += list(set([ listElem[3] for listElem in word4 if len(listElem) == 5 and listElem[3] not in excluList and listElem[3].isalpha()]))






#### Cluster 1 : extraction de qq ingrédients 

# phrase avec 2 token dont le premier est un chiffre => 272 ingrédients
listIngredientsToSave += list(set([ listElem[1] for listElem in word3 if len(listElem) == 2 and not listElem[0].isalpha() and listElem[1] not in excluList ]))

#108
listIngredientsToSave += list(set([ listElem[0] for listElem in word3 if len(listElem) == 2 and listElem[0] not in excluList and listElem[0].isalpha()]))
#104
listIngredientsToSave += list(set([ listElem[0] for listElem in word3 if len(listElem) == 2 and listElem[0] not in excluList and listElem[0].isalpha()]))
#143
listIngredientsToSave += list(set([ listElem[0] for listElem in word3 if len(listElem) == 5 and listElem[0] not in excluList and listElem[0].isalpha() ]))
#139
listIngredientsToSave += list(set([ listElem[3] for listElem in word3 if len(listElem) == 5  and listElem[3].isalpha() and listElem[3] not in excluList and listElem[0].isalpha() ]))
#320
listIngredientsToSave += list(set([ listElem[4] for listElem in word3 if len(listElem) == 5  and listElem[4].isalpha() and listElem[4] not in excluList and listElem[0].isalpha() ]))
#108
listIngredientsToSave += list(set([ listElem[4] for listElem in word3 if len(listElem) == 6  and listElem[4].isalpha() and listElem[4] not in excluList and listElem[0].isalpha() ]))
#197
listIngredientsToSave += list(set([ listElem[5] for listElem in word3 if len(listElem) == 6  and listElem[5].isalpha() and listElem[5] not in excluList and listElem[0].isalpha()]))
#164
listIngredientsToSave += list(set([ listElem[0] for listElem in word3 if len(listElem) == 3  and listElem[0].isalpha() and listElem[0] not in excluList and listElem[0].isalpha()]))



#### Cluster 2 : extraction de qq ingrédients 

#380
listIngredientsToSave += list(set([ listElem[3] for listElem in word1 if len(listElem) == 6 and listElem[3] not in excluList and listElem[3].isalpha()]))
#140
listIngredientsToSave += list(set([ listElem[4] for listElem in word1 if len(listElem) == 6 and listElem[4] not in excluList and listElem[4].isalpha()]))
#90
listIngredientsToSave += list(set([ listElem[2] for listElem in word1 if len(listElem) == 5 and listElem[2] not in excluList and listElem[2].isalpha()]))
#276
listIngredientsToSave += list(set([ listElem[3] for listElem in word1 if len(listElem) == 7 and listElem[3] not in excluList and listElem[3].isalpha()]))
#193
listIngredientsToSave += list(set([ listElem[5] for listElem in word1 if len(listElem) == 7 and listElem[5] not in excluList and listElem[5].isalpha()]))



#### Cluster 5 : extraction de qq ingrédients 
#42
listIngredientsToSave += list(set([ listElem[0] for listElem in word2 if len(listElem) == 1 and listElem[0] not in excluList ] ))
#9
listIngredientsToSave += list(set( [ listElem[1] for listElem in word2 if len(listElem) == 3 and not(listElem[0].isalpha()) and listElem[1] not in excluList and listElem[1].isalpha()]))
#78
listIngredientsToSave += list(set([ listElem[3] for listElem in word2 if len(listElem) == 4 and (listElem[3].isalpha()) and listElem[3] not in excluList]))
#55
listIngredientsToSave += list(set([ listElem[1] for listElem in word2 if len(listElem) == 4 and (listElem[1].isalpha()) and listElem[1] not in excluList]))
#39
listIngredientsToSave += list(set([ listElem[0] for listElem in word2 if len(listElem) == 2 and (listElem[0].isalpha()) and listElem[0] not in excluList]))
#44
listIngredientsToSave += list(set([ listElem[1] for listElem in word2 if len(listElem) == 2 and (listElem[1].isalpha()) and listElem[1] not in excluList]))


# On obtient une liste de 1127 ingrédients
listIngredientsToSave = list(set(listIngredientsToSave))
print("Nombre d'ingrédients récupérés: ",len(listIngredientsToSave),"\n\n")

df = pd.DataFrame(listIngredientsToSave)
df.to_csv('listIngredientsToSave.csv', index=False,header=False)


##########################################################################   
###########         Fin  Analyse des clusters        #####################
##########################################################################   


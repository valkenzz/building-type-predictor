##############info###########################
#script permettant l'entraînement d'un modèle et l'évaluation de cet entraînement.
#les paramètres à modifier sont situés dans la zone notée comme paramètre, elle est juste ci-dessous.
#############################################


#####################PARAMETRE###################
#chemin du jeu de données des empreintes de bâtiment.
pathEmp="./donnerQuebec/batiment/dfBatiQC.shp"
#chemin du jeu de donnée featuré.
PatchFeature="./YAMASKA/Feature.csv"
PatchFeature="./Quebec/Feature.csv"
#fichier pour enregistrer les resultats de l'entraînement.
PathResulta="./resultaEntrainement/"
#nom du modèle.
ModelPath = './resultaEntrainement/modelQuebec2.sav'
#chemin du scaler.
scaler_filename = './resultaEntrainement/scalerQuebec2.sav'
#####################FIN PARAMETRE###############


#on commence par les importations.
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from joblib import dump
import joblib

#on importe les données avec features.
df=pd.read_csv(PatchFeature).drop(columns=["Unnamed: 0"])




'''
#ZONE A UTILISER AVEC LES DONNER DONNER

#colone suprimer
df=df.drop(columns=["ratioDistRelativ","equidistance"])


#pour tester sur les donner de Yamaska
#df=df[df['type_batim']!=7]
#df=df[df['type_batim']!=2]
#df=df[df['type_batim']!=4]
#df['type_batim']=df['type_batim'].apply(lambda x : 4 if x==8 else x)
#df['type_batim']=df['type_batim'].apply(lambda x : 2 if x==3 else x)
#df['type_batim']=df['type_batim'].apply(lambda x : 3 if x==5 else x)
#df['type_batim']=df['type_batim'].apply(lambda x : 5 if x==6 else x)


#pour tester sur les 2 jeux de donner
#PatchFeature="./ContratFinal/Quebec/Feature.csv"
#df2=pd.read_csv(PatchFeature).drop(columns=["Unnamed: 0"])
#df2=df2.drop(columns=["ratioDistRelativ","equidistance"])
#df = pd.concat([df, df2], axis=0, join="inner",ignore_index=True)
'''




#petite visualisation du jeu de données (pour un code qui visualise plus en détail, contacter VAMEO@ulaval.ca).
print("diferente classe:")
print(df['type_batim'].value_counts())

#on importe les empreintes raw.
dfEmp = gpd.read_file(pathEmp)

#########################preprosesing###########################
#on prépare les données pour l'entraînement.
x=df.drop(columns=['type_batim'])
y=df['type_batim']

#Scaler
scaler = StandardScaler()
scaler.fit(x.drop(columns=('id_batimen')))
#on sauvegarde le Scaler.
joblib.dump(scaler, scaler_filename) 

#on crée le jeu d'entraînements et de tests.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1776)

#on mémorise les Id des bâtiments pour une analyse detaillée des erreurs.
testID=x_test["id_batimen"]
x_train=x_train.drop(columns=['id_batimen'])
x_test=x_test.drop(columns=['id_batimen'])

#on applique le scaler
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#plus besoin de sampling si les données sont assez nombreuses comme ici.

#########################entrainement#########################

#on teste plusieurs modèles.
clasifier=[KNeighborsClassifier(metric='canberra'),svm.SVC(),GaussianNB(),DecisionTreeClassifier(),MLPClassifier(),RandomForestClassifier(random_state=1776),
          LogisticRegression(),GradientBoostingClassifier(random_state=1778),AdaBoostClassifier()]
result=[]
for i in clasifier:
    print(i)
    i.fit(x_train, y_train)
    y_pred=i.predict(x_test)
    score=i.score(x_test,y_test)
    print(score)
    result.append([score,i])
    
#on regarde les details du meilleur modèle; ici Random forest
#matrice de confusion : 
print("matrice de confusion du model :")
cf=RandomForestClassifier(random_state=42)#ur best here
print(cf)
cf.fit(x_train, y_train)
y_pred=cf.predict(x_test)
score=cf.score(x_test,y_test)
print(score)
print(confusion_matrix(y_test, y_pred))

#on sauvegarde le modèle.
dump(cf, ModelPath) 

#on génère un shp pour visualiser les erreurs dans un logiciel adapté.
df = pd.DataFrame(testID)
a=y_pred==y_test
unique, counts = np.unique(a, return_counts=True)
print("nombre d'erreur total")
print(dict(zip(unique, counts)))
df['bon']=a.values
dfEmp=dfEmp.set_index('id_batimen')
dfEmp=dfEmp[["geometry","type_batim"]]
df=df.set_index('id_batimen')
result = pd.concat([df, dfEmp], axis=1, join="inner")
result = gpd.GeoDataFrame(result, geometry='geometry')

#on sauvegarde les resultats.
result.to_file(PathResulta + str("ResultaEntrainement.shp"))


#################################################################################

#analyse de l'impact des features.
rfc= RandomForestClassifier(random_state=42)
rfc.fit(x_train, y_train)
feats = {}
for feature, importance in zip(x.columns.drop(columns=('id_batimen')), rfc.feature_importances_):
    feats[feature] = importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
importances = importances.sort_values(by='Gini-Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})
sns.set(font_scale = 5)
sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
fig, ax = plt.subplots()
fig.set_size_inches(30,15)
sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
plt.xlabel('Importance', fontsize=25, weight = 'bold')
plt.ylabel('Features', fontsize=25, weight = 'bold')
plt.title('Feature Importance', fontsize=25, weight = 'bold')
plt.savefig(PathResulta + str("importance.png"))
print('importance des diferente feature')
print(importances)
plt.close(fig)
pca_test = PCA(n_components=30)
pca_test.fit(x_train)
sns.set(style='whitegrid')
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
plt.savefig(PathResulta + str("PCA.png"))
evr = pca_test.explained_variance_ratio_
cvr = np.cumsum(pca_test.explained_variance_ratio_)
pca_df = pd.DataFrame()
pca_df['Cumulative Variance Ratio'] = cvr
pca_df['Explained Variance Ratio'] = evr
print("pca")
print(pca_df.head(30))


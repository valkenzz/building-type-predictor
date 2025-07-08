##############info###########################
#script permettant l'entraînement d'un modèle et l'évaluation de cet entraînement.
#les paramètres à modifier sont situés dans la zone notée comme paramètre, elle est juste ci-dessous.
#############################################


#####################PARAMETRE###################
#chemin du jeu de données des empreintes de bâtiment afin de visualiser les resultat.
pathEmpquebec="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/predictiondelusagedesbatiments-main/donnerQuebec/batiment/dfBatiQC.shp"
pathEmp="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/predictiondelusagedesbatiments-main/donnerYamaska/batiment/dt_batiment v9 sans attributs.shp"

#chemin du jeu de donnée featuré.
PatchFeature="C:/Users/Valentin/Desktop/CleanWorkflow/TrainingDataSet.csv"

#fichier pour enregistrer les resultats de l'entraînement.
PathResulta="C:/Users/Valentin/Desktop/CleanWorkflow/ModelTrainingResult"

#nom du modèle.
ModelPath = 'C:/Users/Valentin/Desktop/CleanWorkflow/ModelTrainingResult/model.sav'
#chemin du scaler.
scaler_filename = "C:/Users/Valentin/Desktop/CleanWorkflow/ModelTrainingResult/scaler.sav"
#chemin du label encoder
LabelEncoderPath="C:/Users/Valentin/Desktop/CleanWorkflow/ModelTrainingResult/"
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
import joblib
from joblib import dump
from sklearn import preprocessing



#on importe les données avec features.
df=pd.read_csv(PatchFeature).drop(columns=["Unnamed: 0"])

#petite visualisation du jeu de données (pour un code qui analyse les features plus en détail, contacter VAMEO@ulaval.ca).
print("diferente classe:")
print(df['type_batim'].value_counts())

#on traite les classe
def Option(x):
    if x==0 or x==10:
        return "Residencielle"
    elif x==1 or x==11:
        return "Cabanon/Garage"
    elif x==2 or x==13 or x==12 or x==14:
        return "Comercial"
    elif x==3 or x==15:
        return "Industrielle"
    elif x==4 or x==27:
        return "Institutionelle"
    elif x==16:
        return "BatimentAgricole"
    else:
        return x

#12 == centre comercial
#14== bureau
#15==indu
#13=comercial
#27== eglise ou universiter ou institutionelle
#10==maison
#11=cabanon
#16==agricole
#0=maison
#1=cabanon
#2=comercial
#3=indu
    
df["type_batim"]=df["type_batim"].apply(lambda x : Option(x))

#on importe les empreintes de yamaska
dfEmp=gpd.read_file(pathEmp)
#YAMASKA########################################################################
#on rajoute les type de batiment categoriser manuellement
dfff= gpd.read_file("C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerYamaska/TYPE_BATIMEN/resulta2.shp")
dfEmp['type_batim']=dfEmp['id_batimen'].apply(lambda x :20 if len(dfff[dfff['FID']==x]['ClasePredi'])==0 else dfff[dfff['FID']==x]['ClasePredi'].values[0]) 
dfEmp=dfEmp[dfEmp['type_batim']!=20]

#on mémorise le crs.
crss=dfEmp.crs
#on importe les empreintes de quebec
dfEmpquebec=gpd.read_file(pathEmpquebec)
#on merge les dataframe
#on utilise le même crs que pour les adresses. 
dfEmp=dfEmp.to_crs(crss)
#on utilise le même crs que pour les adresses. 
dfEmpquebec=dfEmpquebec.to_crs(crss)
dfEmpF = pd.concat([dfEmp, dfEmpquebec], axis=0, join="inner",ignore_index=True)
dfEmp=dfEmpF


#on encode les colone.

le = preprocessing.LabelEncoder()
le.fit(df["type_batim"])
df["type_batim"]=le.transform(df["type_batim"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
#on sauvegarde le labelle encoder code
dump(le, LabelEncoderPath+str('classe.sav')) 

df["categPred"]=df["categPred"].apply(lambda x: str(x))
le = preprocessing.LabelEncoder()
le.fit(df["categPred"])
df["categPred"]=le.transform(df["categPred"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
#on sauvegarde le labelle encoder code
dump(le, LabelEncoderPath+str('categPred.sav')) 

df["desc_usage"]=df["desc_usage"].apply(lambda x: str(x))
le = preprocessing.LabelEncoder()
le.fit(df["desc_usage"])
df["desc_usage"]=le.transform(df["desc_usage"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
#on sauvegarde le labelle encoder code
dump(le, LabelEncoderPath+str('desc_usage.sav')) 

le = preprocessing.LabelEncoder()
le.fit(df["categAdr"])
df["categAdr"]=le.transform(df["categAdr"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
#on sauvegarde le labelle encoder code
dump(le, LabelEncoderPath+str('categAdr.sav')) 

le = preprocessing.LabelEncoder()
le.fit(df["rt_generic"])
df["rt_generic"]=le.transform(df["rt_generic"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
#on sauvegarde le labelle encoder code
dump(le, LabelEncoderPath+str('rt_generic.sav')) 

le = preprocessing.LabelEncoder()
le.fit(df["cls_foncti"])
df["cls_foncti"]=le.transform(df["cls_foncti"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
#on sauvegarde le labelle encoder code
dump(le, LabelEncoderPath+str('cls_foncti.sav')) 

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
#clasifier=[KNeighborsClassifier(metric='canberra'),svm.SVC(),GaussianNB(),DecisionTreeClassifier(),MLPClassifier(),RandomForestClassifier(random_state=1776),
 #         LogisticRegression(),GradientBoostingClassifier(random_state=1778),AdaBoostClassifier()]
#result=[]
#for i in clasifier:
#    print(i)
#    i.fit(x_train, y_train)
#    y_pred=i.predict(x_test)
#    score=i.score(x_test,y_test)
#    print(score)
#    result.append([score,i])
    
#on regarde les details du meilleur modèle; ici Random forest
#matrice de confusion : 
print("matrice de confusion du model :")
cf=RandomForestClassifier(random_state=42)
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
for feature, importance in zip(x.drop(columns=('id_batimen')).columns, rfc.feature_importances_):
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
pca_test = PCA(n_components=20)
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


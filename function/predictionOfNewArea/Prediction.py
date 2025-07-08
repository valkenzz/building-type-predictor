##############info###########################
#script permettant d'effectuer des predictions avec les features.
#les parametres a  modifier sont situees dans la zone note comme parametre, elle est juste ci-dessous.
#############################################


#####################PARAMETRE###################
#chemin du jeu de donnes featurer.
PatchFeature="C:/Users/valentin/Desktop/workflow2.0/tmpResult/ResultatArea.csv"
#fichier pour enregistrer le resultat.
PathResulta="C:/Users/valentin/Desktop/workflow2.0/prediction.csv"
#nom du modele.
filename= "C:/Users/valentin/Desktop/workflow2.0/model/model.sav"
#chemin du scaler.
scaler_filename = "C:/Users/valentin/Desktop/workflow2.0/model/scaler.sav"
#endroit oÃ¹ a ete sauvegarde le label encodeur qui a encode les classes.
LabelEncoderPath='C:/Users/valentin/Desktop/workflow2.0/model/'
#chemin du jeux de données des empreintes de bâtiment.
pathEmp="C:/Users/valentin/Desktop/workflow2.0/predictionOfNewArea/Données pour BV Chaudière (transmis juin 2023)/dt_batiment_v19_2023_06_05_11_27_43.gpkg"
#folder pour stocker le resultat avec les empreinte
ResultFolderPath='C:/Users/valentin/Desktop/workflow2.0/tmpResult/pred.shp'
#####################FIN PARAMETRE###############


#on commence par les importations.
import pandas as pd
import joblib
from joblib import load
import geopandas as gpd
import numpy as np


#on importe les donnees.
df=pd.read_csv(PatchFeature).drop(columns=["Unnamed: 0"])

df=df[['id_batiment', 'distRout', 'surface', 'D', 'I', 'R', 'C', 'O',
       'nbBatimentDansLaParcelle', 'Surfaceparcelle', 'ZONEindustrielle',
       'rt_generic', 'perimetre', 'cls_foncti', 'alignement',
       'ZONEinstitutionelle', 'Sommet', 'densiter', 'nbAdressDansParcelle',
       'densiterProche', 'elongement', 'categAdr', 'categPred',
       'distAdrCentroid', 'densitertresProche', 'desc_usage',
       'surfaceAutreBatiment']]


#fonction pr encoder les colum, mais en gerer les valeur qui on pas encor ete vus
def lateEncod(colName):
    pat=LabelEncoderPath+colName
    le=joblib.load(pat+'.sav')
    # Get the unique values in the 'rt_generic' column
    unique_labels = df[colName].unique()
    # Create a mask to filter out unseen labels
    mask = np.isin(df[colName], le.classes_)
    # Map the labels in the 'rt_generic' column, excluding unseen labels
    x = le.transform(df.loc[mask, colName])
    maximum=max(x)+1
    df[colName]=df[colName].apply(lambda x : maximum if x not in le.classes_ else x)
    df.loc[mask, colName] = le.transform(df.loc[mask, colName])
    return df[colName]


df['rt_generic']=lateEncod('rt_generic')

df['categPred']=lateEncod('categPred')

df['cls_foncti']=lateEncod('cls_foncti')

df['desc_usage']=lateEncod('desc_usage')

df['categAdr']=lateEncod('categAdr')


x=df.drop(columns=['id_batiment'])
#on supprime la colonne id pas utile ici.

#########################preprosesiong#########################################
#on charge le scaler.
scaler = joblib.load(scaler_filename)  
###############################################################################   

#on cherche le modele.  
cf = load(filename)
#on predit.
y_pred=cf.predict(scaler.transform(x))

#on sauvegarde les predictions.
pred = pd.DataFrame(y_pred, columns = ['Prediction'])
result = pd.concat([df['id_batiment'], pred], axis=1, join="inner",ignore_index=True)
#on rend les predictions lisibles.
result["Classe"]=result[1]
result["ID_bat"]=result[0]
result=result.drop(columns=[0,1])
#on sauvegarde le resultat.
result.to_csv(PathResulta)

#si jamais on veut assosier les ids avec les empreinte
#on importe les empreintes.
dfEmp=gpd.read_file(pathEmp)
#on recup que ce que on veut classifier
#dfEmp=dfEmp[dfEmp['version']==666]


dfEmp['ID_bat']=dfEmp['id_batiment']
dfEmp=dfEmp.drop(columns=['id_batiment'])
dfEmp['ID_bat']=dfEmp['ID_bat'].apply(lambda x: int(x))
dfEmp['ID_bat']=dfEmp['ID_bat'].astype(int)

result['ID_bat']=result['ID_bat'].apply(lambda x: int(x))
result['ID_bat']=result['ID_bat'].astype(int)

merg=dfEmp.merge(result, on='ID_bat', how='left')
print(merg['Classe'].value_counts())
merg.to_file(ResultFolderPath)  





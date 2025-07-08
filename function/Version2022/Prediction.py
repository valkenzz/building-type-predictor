##############info###########################
#script permettant d'effectuer des predictions avec les features.
#les parametres a† modifier sont situees dans la zone note comme parametre, elle est juste ci-dessous.
#############################################


#####################PARAMETRE###################
#chemin du jeu de donnes featurer.
PatchFeature="./Feature.csv"
#fichier pour enregistrer le resultat.
PathResulta="./prediction.csv"
#nom du modele.
filename = './resultaEntrainement/model.sav'
#chemin du scaler.
scaler_filename = './resultaEntrainement/scaler.sav'
#endroit o√π a ete sauvegarde le label encodeur qui a encode les classes.
LabelEncoderPath="./YAMASKA/LabelEncd.sav"
#####################FIN PARAMETRE###############


#on commence par les importations.
import pandas as pd
import joblib
from joblib import load

#on importe les donnees.
df=pd.read_csv(PatchFeature).drop(columns=["Unnamed: 0"])
x=df
#on supprime la colonne id pas utile ici.
x=x.drop(columns=('id_batimen'))

#SI LES DONNEES ON ETE GENEREES POUR UN ENTRAINEMENT, ON SUPPRIME LA COLONNE TYPE
#x=df.drop(columns=['type_batim'])

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
result = pd.concat([df['id_batimen'], pred], axis=1, join="inner",ignore_index=True)
#on rend les predictions lisibles.
le = load(LabelEncoderPath)
result["Classe"]=le.inverse_transform(result[1])#.to_numpy()
result["ID_bat"]=result[0]
result=result.drop(columns=[0,1])
#on sauvegarde le resultat.
result.to_csv(PathResulta)


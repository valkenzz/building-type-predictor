le fichier generateurFeature2022.py est le script permettant de générer les features nécessaires à 
l'entraînement ou la prediction.
le fichier Entrainement.py est le script utilisé pour entrainer l'algorithme.
le fichier Prediction.py permet de prédire les catégories.
./resultaEntrainement/scaler est le scaler préentrainé sur Québec et sur Saint-Hyacinthe et ses environs.
./resultaEntrainement/scalerQuebec est le scaler préentraîné sur Québec. 
./resultaEntrainement/scalerCampagne est le scaler préentraîné sur Saint-Hyacinthe et ses environs.
./resultaEntrainement/Model est le modèle préentraîné sur Québec et sur Saint-Hyacinthe et ses environs.
./resultaEntrainement/ModelQuebec est le modèle préentraîné sur Québec.
./resultaEntrainement/ModelCampagne est le modèle préentraîné sur Saint-Hyacinthe et ses environs.

Le dossier "resulta entrainement" contient les exemples de résultats d'un entraînement.
FormaData.pdf est la documentation sur le format des données nécessaires.

les donner brute utiliser pour generer les feature sont dans donnerQuebec et donnerYamaska et dt_aq_route2020 pour les route
les feature sont dans le docier quebec pour yamaska et yamaska pour la region de quebec xd

un example de resulta de prediction peut etre trouver dans ExampleDeResultaDePrediction

Note : tous les exemples de resulta/modelPrenhentrainer/Feature ont quelques colonnes supprimées depuis : pour utiliser ses donner dans un entrainement il faut donc decomanter les ligne presiser dans le fichier entrainement
les donner YAMASKA nessesite egalement un traitement particulier pour les utiliser lort du featuring ou de l'entrainement des ligne sont a decomanter : tout est indiquer dans les fichier

#Les optimisations ont donné de très très faibles résultats. Ils ne sont donc pas présents ici.
#Pour des analyses plus détaillées des features, n'hésitez pas à me contacter à vameo@ulaval.ca 


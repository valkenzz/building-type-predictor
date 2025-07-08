Ce dossier contient le workflow de prédiction des types de bâtiment.

APERÇU : Ce travail a pour but de prédire les principaux types de bâtiments à partir des empreintes de bâtiment et de quelques autres données facilement accessibles au Québec. Pour cela, un modèle Random Forest est utilisé. Ce modèle est feeder après une étape de création de feature couteuse en temps de calcul. Le choix de ces caractéristiques a été effectué après une exploration approfondie. Ce modèle a montré des performances largement supérieures à ce qui existe dans la littérature.

Le code et les données ayant servi à entraîner le modèle sont situés dans le sous-dossier "Entraînement".
Le modèle en tant que tel et tous les encodeurs/scaler se trouvent dans le dossier "Modèle". 
Enfin, le sous-dossier predictionOfNewArea contient le workflow utile à la prédiction de nouvelles zones. Ce dernier sous-dossier contient un script utilisé pour générer les caractéristiques pour chaque empreinte de bâtiment. L'en-tête du script peut être modifié pour pointer vers les données voulues. De plus, un script est utilisé pour effectuer la prédiction à partir des empreintes dotées de caractéristiques. Ce modèle utilise le modèle préalablement entraîné et stocke les résultats dans le dossier "RésultatsTemporaires".

Les détails sur l'analyse des différentes caractéristiques peuvent être trouvés dans la version précédente du workflow. Si vous avez des questions, n'hésitez pas à me contacter : vameo@ulaval.ca.
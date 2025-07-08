
##############info###################
#Ce script prend en entrée les données brutes (cf documentation sur le format des données),
#calcul les features,
#et enregistre un csv utilisable pour les predictions ou l'entraînement.
#Tous les chemins d'accès où se trouvent vos données sont à modifier dans la partie ZONE A MODIFIER .
#La localisation du fichier d'enregistrement est aussi à préciser.
#Il est possible de créer des features pour l'entraînement ou pour la prédiction.
####################################


########################ZONE A MODIFIER###################################
#chemin du jeu de données d'adresse.
pathADR="./donnerQuebec/adresse/adresses_vdq.shp"
#chemin du jeux de données des empreintes de bâtiment.
pathEmp="./donnerQuebec/batiment/dfBatiQC.shp"
#chemin du jeu de données cadastral.
pathCadastre="./donnerQuebec/cadastre/cadastre_vdq.shp"
#chemin du jeu den données routier.
pathRoute="./dt_aq_route_2020/dt_aq_route_2020.shp"
#output
OutpoutFilename="./Resulta.csv"
#pour un jeu d'entraînement ou de test?
entrainement=True
#endroit où sauvegarder le label encodeur qui a encodé les classes.
LabelEncoderPath="./LabelEncd.sav"
####Optionnel
#permet d'avoir des labels explicites si les labels sont numeriques avec 20 pour les erreurs de détection.
#utilisation de l'option?
option=False
#fonction à modifier selon les catégories
def Option(x):
    if x==0:
        return "Residencielle"
    elif x==2:
        return "Cabanon/Garage"
    elif x==4:
        return "Comercial"
    elif x==3:
        return "Industrielle"
    elif x==4:
        return "Institutionelle"
    elif x==3:
        return "BatimentAgricole"
    else:
        return x
########################FIN ZONE A MODIFIER################################


#on importe les librairies
import geopandas as gpd
import numpy as np
from sklearn import preprocessing
import rtree
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import nearest_points
from joblib import dump

########################################################
#on importe les adresses.
adresse = gpd.read_file(pathADR)
#on mémorise le crs.
crss=adresse.crs

#on importe les empreintes.
dfEmp=gpd.read_file(pathEmp)

#si l'option est activée, on transforme les labels en des labels plus lisibles.
if option==True:
    dfEmp["type_batim"]=dfEmp["type_batim"].apply(lambda x : Option(x))
    #si c un entraniement on suprime les erreur de detection
    if entrainement==True:
        dfEmp=dfEmp[dfEmp['type_batim']!=20]
#ici votre traitement sur les classe







    
#on utilise le même crs que pour les adresses. 
dfEmp=dfEmp.to_crs(crss)
#on crée une zone avec laquelle on pourra filtrer les autres jeux de données utilisées (pour gagner en performances).
h=dfEmp.buffer(1000).unary_union

#on ne garde que les adresses proches des empreintes, inutile de calculer les autres adresses.
adresse=adresse[adresse.within(h)]

#on importe le cadastre. 
cadastre = gpd.read_file(pathCadastre)
#on utilise le même crs que pour les adresses.
cadastre=cadastre.to_crs(crss)
#on ne garde que les données proches des empreintes.
cadastre=cadastre[cadastre.within(h)]

#on importe les routes.
route = gpd.read_file(pathRoute)
#on utilise le même crs que pour les adresses.
route=route.to_crs(crss)
#on garde que les données proches des empreintes.
route=route[route.within(h)]
########################################################


################testing mode##################
#pour tester sans trop de calcul.
#dfEmp=dfEmp[0:100]
##############################################


###########################pré-traitement#######################################
#on verifie qu'il n'y a pas de duplicata.
print("longueur avant de supprimer les duplicatas sur la colonne FID : "+str(len(dfEmp)))
dfEmp=dfEmp.drop_duplicates(subset=['id_batimen'])
print("longueur après avoir supprimé les duplicatas sur la colonne FID : "+str(len(dfEmp)))
#étant donné qu'il y a quasiment uniquement des adresses certifiées, on ne va garder que celles-là.
adresse=adresse[adresse["Etat"]=="Certifiée"]
#on réindexe les jeux de données à cause de la suppression précédente. Cela est nécessaire pour la suite.
route=route.reset_index(drop=True)
cadastre=cadastre.reset_index(drop=True)
adresse=adresse.reset_index(drop=True)
#######################FUNCTION########################################################
#on définit les fonctions.

#fonction de calcul de l'élément le plus proche et la distance aassociée à partir d'une géométrie et d'un geodaframe d'éléments. Le dernier paramètre a un intérêt esthétique ^^
def min_dist(point, gpd2,ColName):  
    VariableSansNom=gpd2.sindex.query(point.buffer(100))
    VariableSansNom=gpd2.loc[VariableSansNom]
    if len(VariableSansNom)==0:
        VariableSansNom=gpd2
    VariableSansNom[ColName] = VariableSansNom.apply(lambda row:  point.distance(row.geometry),axis=1)
    geoseries = VariableSansNom.iloc[VariableSansNom[ColName].argmin()]
    return geoseries

#fonction retournant les informations de l'élément qui contient la geometrie. #usuellement, les informations de la parcelle de l'empreinte.
def parcelleGet(point,gpd2,ColName):
    rt=gpd2.sindex.query(point.buffer(100))
    gpd2=gpd2.loc[rt]
    return gpd2[gpd2['geometry'].intersects(point)][ColName].values
    
#calcule de densité. La fonction prend en entrée la géométrie (qui servira de centre pour le calcul de la densité) et un geodataframe de géométrie. Le 3ème paramètre definit l'unité d'espace pour calculer la densité (*1.5 = rayon du cercle). 
def Densiter(point,gpd2,dist):
    rt=gpd2.sindex.query(point.buffer(dist*1.5))
    gpd3=gpd2.loc[rt]
    return len(gpd3[gpd3['geometry'].intersects(point.buffer(dist))])
    
#comme son nom l'indique, elle donne le nombre de trucs (inclus dans la géométrie).
def nombreDeTruc(point,gpd2):
    rt=gpd2.sindex.query(point.buffer(50))
    gpd3=gpd2.loc[rt]
    return len(gpd3[gpd3['geometry'].intersects(point)])

#renvoit l'élongement de la géométrie donnée en entrée.
def allongement(poly):
    box = poly.minimum_rotated_rectangle
    x, y = box.exterior.coords.xy
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
    length = max(edge_length)
    width = min(edge_length)
    if width>=length:
        result=length/width
    else:
        result=width/length
    return result

#fonction permettant de calculer le nombre d'adresses dans une géométrie donnée.
def CategCounter(gdf2,parcel):
    r=gdf2.sindex.query(parcel, predicate="contains")#intersection    , cap_style = 3
    gpd2=gdf2.loc[r]
    return (len(gpd2))

#calcule la taille moyenne des elements (d) qui touchent la géométrie(point)
def MoyenCalcultateur(d,point):
    VariableSansNom=d.sindex.query(point.buffer(1))
    VariableSansNom=d.loc[VariableSansNom]
    return VariableSansNom[VariableSansNom["geometry"].touches(point)].area.mean()

#fonction retournant la categorie d'adresses la plus présente dans les parcelles adjacentes à la géométrie.
def MoyenCalcultateur2(d,point,d2):
    VariableSansNom=d.sindex.query(point.buffer(1))
    VariableSansNom=d.loc[VariableSansNom]
    h=VariableSansNom[VariableSansNom["geometry"].touches(point)].geometry.buffer(1).unary_union
    d3=d2.sindex.query(h)
    d3=d2.loc[d3]
    return d3['Categorie'].value_counts().index

#fonction retournant le nombre de types d'adresse unique présents dans les parcelles adjacentes.
def MoyenCalcultateur3(d,point,d2):
    VariableSansNom=d.sindex.query(point.buffer(1))
    VariableSansNom=d.loc[VariableSansNom]
    h=VariableSansNom[VariableSansNom["geometry"].touches(point)].geometry.buffer(1).unary_union
    d3=d2.sindex.query(h)
    d3=d2.loc[d3]
    return d3['Categorie'].unique()

#fonction retournant l'alignement à la route la plus proche de la géométrie fournie en entrée.
def alignement(poly):
    rout=min_dist(poly.centroid,route,'distAdr')["geometry"]
    nearest_point = nearest_points(rout, poly.centroid)[0]
    line_points_except_nearest = MultiPoint([point for point in rout.coords 
                                         if point != (nearest_point.x, nearest_point.y)])
    nearest_point2 = nearest_points(line_points_except_nearest, poly.centroid)[0]
    rout= LineString([nearest_point2, nearest_point])
    x, y = poly.exterior.coords.xy
    x1x0 = Point(x[0], y[0]).distance(Point(x[1], y[1]))
    x1x2 = Point(x[1], y[1]).distance(Point(x[2], y[2]))
    if x1x0>x1x2:
        ligne = LineString([Point(x[0], y[0]), Point(x[1], y[1])])     
    elif x1x0<x1x2:
        ligne = LineString([Point(x[2], y[2]), Point(x[1], y[1])])
    else:
        return 0
    xs1, xe1 = ligne.xy[0]
    ys1, ye1 = ligne.xy[1]
    xs2, xe2 = rout.xy[0]
    ys2, ye2 = rout.xy[1]
    if xs1 == xe1:
        angle1 = np.pi/2
    else:
        angle1 = np.arctan((ye1-ys1)/(xe1-xs1))
    if xs2 == xe2:
        angle2 = np.pi/2
    else:
        angle2 = np.arctan((ye2-ys2)/(xe2-xs2))

    return abs(angle2-angle1)

#fonction pour augmenter la lisibilité des différente catérgorie.
#entrée : string ,
#utiliser ici pour le label du type de bâtiment prédit par la base line.
def fonction(x):
    if x=='R' or x=='V':
        return "R"
    elif x=='N/D' or x==0 or x=="D":
        return "Indus"
    elif x=='O':
        return "Artefact"
    elif x=='G':
        return "cabanon"
    else:
        return x

#même fonction que min_dist mais avec une réindexation.
def min_distIndex(point, gpd2,ColName):  
    VariableSansNom=gpd2.sindex.query(point.buffer(100))
    gpd2=gpd2.reset_index(drop=True)
    VariableSansNom=gpd2.loc[VariableSansNom]
    if len(VariableSansNom)==0:
        VariableSansNom=gpd2
    VariableSansNom[ColName] = VariableSansNom.apply(lambda row:  point.distance(row.geometry),axis=1)
    geoseries = VariableSansNom.iloc[VariableSansNom[ColName].argmin()]
    return geoseries

#fonction permettant de definir une borne supérieure à une colonne (réduit la dispersion des valeurs aberrante )
def filtre(col,valeur,dff):
    print(len(dff))
    print('-----------')
    print(len(dff[dff[col]<valeur]))
    d=dff
    x=d[col].apply(lambda x : x if x<valeur else valeur)
    print(len(x))
    return x
###############################################################################











###############################################################################
###############################################################################
#UNIQUEMENT POUR LES DONNER DE YAMASKA DANS LE DOSIER JOIN
#on importe la hauteur
#hauteur = gpd.read_file("./donnerYamaska/hauteur/lidar_yamaska_building_stat.shp")


#on commence par ajouter les information de hauteur a chaque empreinte
#si une empreinte na pas de donner assosier on affecte la valeur -1
#dfEmp['h_mean']=dfEmp['id_batimen'].apply(lambda x : -1 if len(hauteur[hauteur['fid']==x]['h_mean'])==0 else hauteur[hauteur['fid']==x]['h_mean'].values[0]) 
#dfEmp['h_median']=dfEmp['id_batimen'].apply(lambda x :-1 if len(hauteur[hauteur['fid']==x]['h_median'])==0 else hauteur[hauteur['fid']==x]['h_median'].values[0]) 
#dfEmp['h_max']=dfEmp['id_batimen'].apply(lambda x : -1 if len(hauteur[hauteur['fid']==x]['h_max'])==0 else hauteur[hauteur['fid']==x]['h_max'].values[0]) 
#dfEmp['h_sol_mean']=dfEmp['id_batimen'].apply(lambda x :-1 if len(hauteur[hauteur['fid']==x]['h_sol_mean'])==0 else hauteur[hauteur['fid']==x]['h_sol_mean'].values[0]) 
#dfEmp['type_toit']=dfEmp['id_batimen'].apply(lambda x :-1 if len(hauteur[hauteur['fid']==x]['type_toit'])==0 else hauteur[hauteur['fid']==x]['type_toit'].values[0]) 
#dfEmp['nb_points']=dfEmp['id_batimen'].apply(lambda x :-1 if len(hauteur[hauteur['fid']==x]['nb_points'])==0 else hauteur[hauteur['fid']==x]['nb_points'].values[0]) 

#on remplace les valeur manquante par la moyen de la colone assosier ou la valeur la plus frequente
#quand l'empreinte n'avais pas de ligne dans le dataFrame hauteur
#dfEmp['h_mean']=dfEmp['h_mean'].replace(-1, hauteur['h_mean'].mean())
#dfEmp['h_median']=dfEmp['h_median'].replace(-1, hauteur['h_median'].mean())
#dfEmp['h_max']=dfEmp['h_max'].replace(-1, hauteur['h_max'].mean())
#dfEmp['h_sol_mean']=dfEmp['h_sol_mean'].replace(-1, hauteur['h_sol_mean'].mean())
#dfEmp['type_toit']=dfEmp['type_toit'].replace(-1, hauteur['type_toit'].value_counts().idxmax())



#on rajoute les type de batiment categoriser manuellement
#df = gpd.read_file("./donnerYamaska/TYPE_BATIMEN/resulta2.shp")
#dfEmp['type_batim']=dfEmp['id_batimen'].apply(lambda x :20 if len(df[df['FID']==x]['ClasePredi'])==0 else df[df['FID']==x]['ClasePredi'].values[0]) 




###############################################################################
###############################################################################














##########HAUTEUR######################

#hauteur : il y a beaucoup de valeurs manquantes donc un traitement spécial est effectué :

#on ne garde que les empreintes avec plus de 30 points.
print(len(dfEmp))
dfEmp=dfEmp[dfEmp["nb_points"]>30]
dfEmp=dfEmp.drop(columns=("nb_points"))
print(len(dfEmp))
dfEmp=dfEmp.reset_index(drop=True)

#on calcule une différence clés pour accélérer le machine learning.
dfEmp['h_median']=dfEmp['h_median']-dfEmp['h_sol_mean']

#valeurs manquantes : on remplace par la moyenne des autres valeurs.
dfEmp['h_median']=dfEmp['h_median'].fillna(dfEmp['h_median'].mean())
dfEmp['h_sol_mean']=dfEmp['h_sol_mean'].fillna(dfEmp['h_sol_mean'].mean())
dfEmp['type_toit']=dfEmp['type_toit'].fillna(dfEmp['type_toit'].value_counts().idxmax())

#si il y a d'autres valeurs manquantes dans les données, elles sont supprimées.
dfEmp=dfEmp.dropna()
dfEmp=dfEmp.reset_index(drop=True)

#######################################
#Calcul de features géométriques simples.

#centre de l'empreinte.
dfEmp["centre"]=dfEmp["geometry"].centroid

#surface de l'empreinte.
dfEmp["surface"]=dfEmp.area

#périmètre de l'empreinte.
dfEmp["perimetre"]=dfEmp["geometry"].length

#nombre de sommets.
dfEmp=dfEmp[dfEmp['geometry'].apply(lambda x : x.geom_type != 'MultiPolygon')]
dfEmp["Sommet"]=dfEmp['geometry'].apply(lambda x : len(x.exterior.coords.xy[0]))

################################
#calcul des features géométriques plus complexes.

#distance à la route la plus proche.
dfEmp["distRout"]=dfEmp["centre"].apply(lambda x: min_dist(x,route,"distRout")["distRout"])

#ID de la principals parcelle de l'empreinte.
dfEmp["parcelle"]=dfEmp["centre"].apply(lambda x:parcelleGet(x,cadastre,"NO_LOT"))
dfEmp["parcelle"]=dfEmp["parcelle"].apply(lambda x : x[0] if x.shape[0]==1 else -1)
dfEmp=dfEmp[dfEmp["parcelle"]!=-1]
dfEmp=dfEmp.reset_index(drop=True) 

#géométrie de la parcelle de l'empreinte.
dfEmp["geometriDeLaparcelle"]=dfEmp["centre"].apply(lambda x:parcelleGet(x,cadastre,"geometry"))
dfEmp["geometriDeLaparcelle"]=dfEmp["geometriDeLaparcelle"].apply(lambda x : x[0])

#nombre d'empreintes détectées dans la parcelle de l'empreinte.
dfEmp["nbBatimentDansLaParcelle"]=dfEmp["parcelle"].apply(lambda x : len(dfEmp[dfEmp["parcelle"]==x])) 

#surface de la parcelle de l'empreinte.
dfEmp["Surfaceparcelle"]=dfEmp["geometriDeLaparcelle"].apply(lambda x : x.area) 
      
#densité d'empreintes dans une unité de surface circulaire de 600 mètres de rayon autour du centre de l'empreinte.
dfEmp["densiter"]=dfEmp["centre"].apply(lambda x : Densiter(x,dfEmp,600))

#densité d'empreinte dans une unité de surface circulaire de 50 mètres de rayon autour du centre de l'empreinte.
dfEmp["densiterProche"]=dfEmp["centre"].apply(lambda x : Densiter(x,dfEmp,50))

#densité d'empreinte dans une unité de surface circulaire de 50 mètres de rayon autour du centre de l'empreinte.
dfEmp["densitertresProche"]=dfEmp["centre"].apply(lambda x : Densiter(x,dfEmp,10))

#nombre d'adresses dans la parcelle associée à l'empreinte.
dfEmp["nbAdressDansParcelle"]=dfEmp["geometriDeLaparcelle"].apply(lambda x : nombreDeTruc(x,adresse))

#élongement de la géométrie de l'empreinte : ratio entre 0 et1 (1 une ligne et 0 un carré)
dfEmp["elongement"]=dfEmp["geometry"].apply(lambda x : allongement(x))

#distance de l'empreinte à l'adresse la plus proche.
dfEmp["distAdr"]=dfEmp["geometry"].apply(lambda x: min_dist(x,adresse,'distAdr')["distAdr"])

#id de l'adresse la plus proche de l'empreinte.
dfEmp["idadr"]=dfEmp["geometry"].apply(lambda x: min_dist(x,adresse,'distAdr')["IdAdr"])

#catégorie de l'adresse la plus proche de l'empreinte.
dfEmp["categAdr"]=dfEmp["idadr"].apply(lambda x : adresse[adresse["IdAdr"]==x]['Categorie'].values[0])

#moyen des distances du bâtiment aux autres bâtiments de la parcelle.
dfEmp['distanceMoyenBatimentParcelle']=dfEmp.apply(lambda x : (dfEmp[dfEmp['parcelle']==x['parcelle']].apply(lambda y: x['centre'].distance(y.geometry),axis=1).mean()) if len(dfEmp[dfEmp['parcelle']==x['parcelle']])>1 else 0,axis=1)

#distance du batiment le plus proche dans la même parcelle.
dfEmp["DistBatCloserDansParcelle"]=dfEmp.apply(lambda x: min_distIndex(x["geometry"],dfEmp[(dfEmp['parcelle']==x['parcelle'])  & (dfEmp['id_batimen']!=x['id_batimen'])],'distAdr')["distAdr"] if len(dfEmp[(dfEmp['parcelle']==x['parcelle'])& (dfEmp['id_batimen']!=x['id_batimen'])])>1 else 0,axis=1)

#distance moyenne des autre batiment de la parcelle au autre batiment
dfEmp["DistMoyenDesAutreBatimentDelaParcelle"]=dfEmp.apply(lambda x:  dfEmp[(dfEmp['parcelle']==x['parcelle'])  & (dfEmp['id_batimen']!=x['id_batimen'])]["distanceMoyenBatimentParcelle"].mean(),axis=1)
dfEmp["DistMoyenDesAutreBatimentDelaParcelle"]=dfEmp["DistMoyenDesAutreBatimentDelaParcelle"].fillna(0)

#moyen de la distance au batiment le plus proche des autre batiment de la parcelle
dfEmp["BatimentPlusProcheMoyenParcelle"]=dfEmp.apply(lambda x:  dfEmp[(dfEmp['parcelle']==x['parcelle'])  & (dfEmp['id_batimen']!=x['id_batimen'])]["DistBatCloserDansParcelle"].mean(),axis=1)
dfEmp["BatimentPlusProcheMoyenParcelle"]=dfEmp["BatimentPlusProcheMoyenParcelle"].fillna(0)

#calcule de quelque ratio pour aider le machine learning notament le randomForest
#calcul d'un ratio permettant d'exprimer la proximité du bâtiment le plus proche par rapport aux autres de la parcelle (plus c'est proche de 0 , plus il est proche de son voisin par rapport aux autres, et 0.5 un isoloment par rapport aux autres qui sont forme un group en moyenne a equidistant)
dfEmp["Isolement"]=dfEmp.apply(lambda x: x['distanceMoyenBatimentParcelle']/(x['distanceMoyenBatimentParcelle']+x["DistBatCloserDansParcelle"]) if x['distanceMoyenBatimentParcelle']!=0 else 0.5 ,axis=1)

#calcul d'un ratio permettant d'exprimer la proximité du bâtiment le plus proche en fonction de la proximité des autres bâtiments entre eux : plus c'est grand(proche de 1), plus la distance relative des bâtiments les plus proches est petite par rapport à la distance du bâtiment le plus proche, plus c'est proche de 0, plus on se raproche d'une situation où les deux bâtiments sont très proches par rapport à la moyenne et plus c'est proche de 0.5, plus on se rapproche d'une équidistance (en moyenne) 
dfEmp["proxy"]=dfEmp.apply(lambda x:  x["DistBatCloserDansParcelle"]/(dfEmp[(dfEmp['parcelle']==x['parcelle'])  & (dfEmp['id_batimen']!=x['id_batimen'])]["DistBatCloserDansParcelle"].mean()+x["DistBatCloserDansParcelle"]) if dfEmp[(dfEmp['parcelle']==x['parcelle'])& (dfEmp['id_batimen']!=x['id_batimen'])]["DistBatCloserDansParcelle"].mean()!=0 else 0   ,axis=1)
dfEmp["proxy"]=dfEmp["proxy"].fillna(1)

#calcul d'un ratio permettant d'exprimer la proximité de l'adresse la plus proche du bâtiment par rapport à la distance de cette adresse à son bâtiment le plus proche.
n=adresse
n['PolyCloser']=n['geometry'].apply(lambda x: min_dist(x,dfEmp,"Poly")["Poly"])
dfEmp["ratioMinDist"]=dfEmp['idadr'].apply(lambda x : (n[n["IdAdr"]==x]['PolyCloser'].values[0])/(dfEmp[dfEmp["idadr"]==x]['distAdr'].values[0]))
dfEmp["ratioMinDist"]=dfEmp["ratioMinDist"].fillna(1)

#calcul des différents nombres d'adresses pour chaque catégorie d'adresse dans la parcelle de l'empreinte. 
n=adresse[(adresse['Categorie']=="R")| (adresse['Categorie']=="V")].reset_index(drop=True)
dfEmp["R"]=dfEmp["geometriDeLaparcelle"].apply(lambda x : CategCounter(n , x))
n=adresse[adresse['Categorie']=="C"].reset_index(drop=True)
dfEmp["C"]=dfEmp["geometriDeLaparcelle"].apply(lambda x : CategCounter(n , x))
n=adresse[adresse['Categorie']=="I"].reset_index(drop=True)
dfEmp["I"]=dfEmp["geometriDeLaparcelle"].apply(lambda x : CategCounter(n , x))
n=adresse[adresse['Categorie']=="O"].reset_index(drop=True)
dfEmp["O"]=dfEmp["geometriDeLaparcelle"].apply(lambda x : CategCounter(n , x))
n=adresse[adresse['Categorie']=="D"].reset_index(drop=True)
dfEmp["D"]=dfEmp["geometriDeLaparcelle"].apply(lambda x : CategCounter(n , x))

#calcul d'une feature permettant de mesurer la densité d'adresses de type industriel dans une unité de surface circulaire de 1000 mètres de rayon autour du centre de la géométrie.
distAdr=1000
AdresseIndustrielle=(adresse[adresse['Categorie']=='D'])
dfEmp["ZONEindustrielle"]=dfEmp["centre"].apply(lambda x : len(AdresseIndustrielle[AdresseIndustrielle["geometry"].intersects(x.buffer(distAdr))])/Densiter(x,dfEmp,1000))

#calcul d'une feature permettant de mesurer la densité d'adresses de type institutionel dans une unité de surface circulaire de 1000 mètres de rayon autour du centre de la géométrie.
AdresseInsitutionelle=(adresse[adresse['Categorie']=='I'])
dfEmp["ZONEinstitutionelle"]=dfEmp["centre"].apply(lambda x : len(AdresseInsitutionelle[AdresseInsitutionelle["geometry"].intersects(x.buffer(distAdr))])/Densiter(x,dfEmp,1000))

#générique de la route la plus proche de l'empreinte.
dfEmp['rt_generic']=dfEmp["centre"].apply(lambda x: min_dist(x,route,'distRout')["ga_odo_gen"])

#fonction de la route la plus proche de l'empreinte.
dfEmp['cls_foncti']=dfEmp["centre"].apply(lambda x: min_dist(x,route,'distRout')["cls_foncti"])#clas route a plus de detaille , mais moin d'inpact

#calcul de la moyenne de taille des parcelles adjacente à la parcelle de l'empreinte.
dfEmp["moyenTailleParcelleColer"]=dfEmp["geometriDeLaparcelle"].apply(lambda x :MoyenCalcultateur(cadastre,x))

#calcul de la catégorie la plus présente dans les parcelles adjacentes à la parcelle de l'empreinte.
dfEmp["categoriePluspresenteDansLesParcelleColer"]=dfEmp["geometriDeLaparcelle"].apply(lambda x :MoyenCalcultateur2(cadastre,x,adresse))
dfEmp["categoriePluspresenteDansLesParcelleColer"]=dfEmp["categoriePluspresenteDansLesParcelleColer"].apply(lambda x: 0 if len(x)==0 else x[0])

#calcul de la distance à l'adresse la plus proche mais en prennant le centre de l'empreinte (permet de regarder comment est placée l'adresse dans le bâtiment).
dfEmp["distAdrCentroid"]=dfEmp["centre"].apply(lambda x: min_dist(x,adresse,"distAdr")["distAdr"])

#calcul de la géométrie englobant l'empreinte.
dfEmp['GeometryEnglobante']=dfEmp['geometry'].apply(lambda x: x.minimum_rotated_rectangle)

#calcul de l'alignement de l'empreinte à la route la plus proche.
dfEmp['alignement']=dfEmp['GeometryEnglobante'].apply(lambda x : alignement(x))

#calcule de la surface moyenne des autres batiments
dfEmp["surfaceAutreBatiment"]=dfEmp.apply(lambda x : dfEmp[(dfEmp['parcelle']==x['parcelle'])]['surface'].mean(),axis=1)

#calcul d'une feature hardcodant un arbre de décision basique pour catégoriser les empreintes. A noter qu'elle sert aussi de base line, avec la prédiction naïve.
dfEmp["categPred"]=dfEmp['nbAdressDansParcelle'].apply(lambda x : 0 if x==0 else "a")
dfEmp["categPred"]=dfEmp.apply(lambda x : x["categAdr"] if x['distAdr']==0 else x["categPred"],axis=1)
dfEmp["categPred"]=dfEmp.apply(lambda x : x["categAdr"] if ((x['nbAdressDansParcelle']==1) & (x['nbBatimentDansLaParcelle']==1))else x["categPred"],axis=1)
dfEmp['categPred']=dfEmp.apply(lambda x : len(MoyenCalcultateur3(cadastre,x["geometriDeLaparcelle"],adresse)) if (x['categPred']=='a' and x['nbBatimentDansLaParcelle']==1) else x['categPred'],axis=1)
dfEmp['categPred']=dfEmp.apply(lambda x : MoyenCalcultateur3(cadastre,x["geometriDeLaparcelle"],adresse)  if x['categPred']==1 else x['categPred'],axis=1)
dfEmp['categPred']=dfEmp.apply(lambda x : "M2" if ((x['categPred']=='a') and (x['nbBatimentDansLaParcelle']==2) and (x['nbAdressDansParcelle']==1)) else x['categPred'],axis=1)
#calcul dans le cadre de la feature précédemment cité de la taille relative par rapport à l'autre bâtiment de la parcelle.
dfEmp["batimentDansPArcelleSurface"]=dfEmp.apply(lambda x : dfEmp[(dfEmp['parcelle']==x['parcelle']) & (dfEmp['id_batimen']!=x['id_batimen'])]['surface'].values if (x['categPred']=="M2") else 0,axis=1)
dfEmp['categPred']=dfEmp.apply(lambda x: "G" if ((x['categPred']=='M2') and (x["batimentDansPArcelleSurface"]>x['surface']) and (MoyenCalcultateur3(cadastre,x["geometriDeLaparcelle"],adresse)[0] in ("R","V"))) else x['categPred']   ,axis=1)
dfEmp['categPred']=dfEmp.apply(lambda x: "R" if ((x['categPred']=='M2') and (x["batimentDansPArcelleSurface"]<x['surface']) and (MoyenCalcultateur3(cadastre,x["geometriDeLaparcelle"],adresse)[0] in ("R","V"))) else x['categPred']   ,axis=1)
dfEmp['categPred']=dfEmp.apply(lambda x: MoyenCalcultateur3(cadastre,x["geometriDeLaparcelle"],adresse)[0] if x['categPred']=='M2' else x['categPred'],axis=1)
#calcul de la distance à la route de l'empreinte de bâtiment de la parcelle de l'empreinte la plus proche de la route la plus proche de l'empreinte (permet de savoir si l'empreinte est la plus proche de la route dans sa parcelle et de savoir la distance).
dfEmp["batimentDansPArcelleDistRoute"]=dfEmp.apply(lambda x : dfEmp[dfEmp['parcelle']==x['parcelle']]['distRout'].min(),axis=1)
dfEmp['categPred']=dfEmp.apply(lambda x : 'R' if ((x['categPred']=="a") and (x['nbBatimentDansLaParcelle']>=4) and (x['nbAdressDansParcelle']==1) and (x["batimentDansPArcelleDistRoute"]==x["distRout"])) else x['categPred'] ,axis=1)
dfEmp['categPred']=dfEmp.apply(lambda x : 'D' if ((x['categPred']=="a") and (x['nbAdressDansParcelle']==1) and (x['nbBatimentDansLaParcelle']>=4) and (x["batimentDansPArcelleDistRoute"]!=x["distRout"])) else x['categPred'] ,axis=1)
t=dfEmp[(dfEmp['categPred']=='a')]
dfEmp["categPred"]=dfEmp["categPred"].apply(lambda x : fonction(x))
####################################
#ajout de nouvelles features






####################################

#on vérifie qu'il n'y a pas de valeurs manquantes dans tous les features précédemment calculés.
for i in dfEmp:
    print('valeur manquante pour la colone : ')
    print(i)
    print(len(dfEmp)-len(dfEmp.dropna(subset=[i])))

#on supprime les valeurs manquantes.
print("longueur avant drop na :")
print(len(dfEmp))
print("longueur apres dropna :")
dfEmp=dfEmp.dropna()
print(len(dfEmp))

#on garde les colonnes pertinentes.
if entrainement==True:
    dfEmp=dfEmp[['parcelle', 'id_batimen', 'distRout', 'surface', 'D', 'I', 'R', 'C',"O",
           'nbBatimentDansLaParcelle', 'Surfaceparcelle', 'ZONEindustrielle', 'rt_generic','perimetre',
           'cls_foncti', 'moyenTailleParcelleColer','type_batim','distanceMoyenBatimentParcelle','ratioDistRelativ',
           'categoriePluspresenteDansLesParcelleColer','alignement','ZONEinstitutionelle','Sommet',
           'densiter', 'nbAdressDansParcelle', 'densiterProche',"ratioMinDist",#ajouter ici le nom des nouvelle feature
           'elongement',"categAdr","categPred", 'distAdrCentroid','densitertresProche',"Isolement",
           "proxy","DistMoyenDesAutreBatimentDelaParcelle","BatimentPlusProcheMoyenParcelle","equidistance",
            'batimentDansPArcelleDistRoute', 'type_toit', 'h_median','h_sol_mean',"surfaceAutreBatiment"]]
else:
    dfEmp=dfEmp[['parcelle', 'id_batimen', 'distRout', 'surface', 'D', 'I', 'R', 'C',"O",
           'nbBatimentDansLaParcelle', 'Surfaceparcelle', 'ZONEindustrielle', 'rt_generic','perimetre',
           'cls_foncti', 'moyenTailleParcelleColer','distanceMoyenBatimentParcelle','ratioDistRelativ',
           'categoriePluspresenteDansLesParcelleColer','alignement','ZONEinstitutionelle','Sommet',
           'densiter', 'nbAdressDansParcelle', 'densiterProche',"ratioMinDist",#ajouter ici le nom des nouvelle feature
           'elongement',"categAdr","categPred", 'distAdrCentroid','densitertresProche',"Isolement",
           "proxy","DistMoyenDesAutreBatimentDelaParcelle","BatimentPlusProcheMoyenParcelle","equidistance",
           'batimentDansPArcelleDistRoute','type_toit', 'h_median','h_sol_mean',"surfaceAutreBatiment"]]
dfEmp=dfEmp.drop(columns=["parcelle"])

#on encode les valeur non numeriques (catégories/labels).
if entrainement==True:
    le = preprocessing.LabelEncoder()
    le.fit(dfEmp["type_batim"])
    dfEmp["type_batim"]=le.transform(dfEmp["type_batim"])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    #on sauvegarde le labelle encoder code
    dump(le, LabelEncoderPath) 

dfEmp["categPred"]=dfEmp["categPred"].apply(lambda x: str(x))
le = preprocessing.LabelEncoder()
le.fit(dfEmp["categPred"])
dfEmp["categPred"]=le.transform(dfEmp["categPred"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

le = preprocessing.LabelEncoder()
le.fit(dfEmp["categAdr"])
dfEmp["categAdr"]=le.transform(dfEmp["categAdr"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

le = preprocessing.LabelEncoder()
le.fit(dfEmp["rt_generic"])
dfEmp["rt_generic"]=le.transform(dfEmp["rt_generic"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)


le = preprocessing.LabelEncoder()
le.fit(dfEmp["cls_foncti"])
dfEmp["cls_foncti"]=le.transform(dfEmp["cls_foncti"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

dfEmp["categoriePluspresenteDansLesParcelleColer"]=dfEmp["categoriePluspresenteDansLesParcelleColer"].apply(lambda x: "X" if x==0 else x)

le = preprocessing.LabelEncoder()
le.fit(dfEmp["categoriePluspresenteDansLesParcelleColer"])
dfEmp["categoriePluspresenteDansLesParcelleColer"]=le.transform(dfEmp["categoriePluspresenteDansLesParcelleColer"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)    

le = preprocessing.LabelEncoder()
le.fit(dfEmp["type_toit"])
dfEmp["type_toit"]=le.transform(dfEmp["type_toit"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)  


#on ajoute une borne superieur
dfEmp['surface']=filtre('surface',1000,dfEmp)
dfEmp['distRout']=filtre('distRout',100,dfEmp)
dfEmp['nbBatimentDansLaParcelle']=filtre('nbBatimentDansLaParcelle',10,dfEmp)
dfEmp['Surfaceparcelle']=filtre('Surfaceparcelle',10000,dfEmp)
dfEmp['moyenTailleParcelleColer']=filtre('moyenTailleParcelleColer',90000,dfEmp)
dfEmp['nbAdressDansParcelle']=filtre('nbAdressDansParcelle',15,dfEmp)
dfEmp['distAdrCentroid']=filtre('distAdrCentroid',70,dfEmp)
dfEmp['batimentDansPArcelleDistRoute']=filtre('batimentDansPArcelleDistRoute',100,dfEmp)
dfEmp['R']=filtre('R',15,dfEmp)
dfEmp["surfaceAutreBatiment"]=filtre('surface',1000,dfEmp)

#on sauvegarde le resultat.
dfEmp.to_csv(OutpoutFilename)

























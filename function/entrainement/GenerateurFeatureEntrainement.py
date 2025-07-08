
##############info###################
#Ce script prend en entrée les données brutes (cf documentation sur le format des données),
#calcul les features,
#et enregistre un csv utilisable pour l'entraînement.
#Tous les chemins d'accès où se trouvent vos données sont à modifier dans la partie ZONE A MODIFIER .
#La localisation du fichier d'enregistrement est aussi à préciser.
####################################


########################ZONE A MODIFIER###################################
#chemin du jeu de données d'adresse.
pathADR="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerYamaska/Adresses brutes Yamaska/Adresses brutes Yamaska.shp"
pathADRquebec="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerQuebec/adresse/adresses_vdq.shp"

#chemin du jeux de données des empreintes de bâtiment.
pathEmp="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerYamaska/batiment/dt_batiment v9 sans attributs.shp"
pathEmpquebec="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerQuebec/batiment/dfBatiQC.shp"

#chemin du jeu de données cadastral.
pathCadastre="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerYamaska/cadastre/Cadastre Yamaska.shp"
pathCadastrequebec="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerQuebec/cadastre/Cadastre_vdq.shp"

#chemin du jeu den données routier.
pathRoute="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/dt_aq_route_2020/dt_aq_route_2020.shp"

#output
OutpoutFilename="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/ResultaTrainingFeatureSansLidar.csv"

#pour un jeu d'entraînement ou de test?
entrainement=True

#endroid ou le role est stocker
PAthRolequebec="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerQuebec/do_role_up_quebec.gpkg"
PAthRole="C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerYamaska/do_role_up_yamaska.gpkg"
    
########################FIN ZONE A MODIFIER################################

#on importe les librairies
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import nearest_points
import pandas as pd

########################################################


#on importe les empreintes de yamaska
dfEmp=gpd.read_file(pathEmp)
#YAMASKA########################################################################
#on rajoute les type de batiment categoriser manuellement
dfff= gpd.read_file("C:/Users/Valentin/Desktop/predictiondelusagedesbatiments-main/CleanWorkflow/donnerYamaska/TYPE_BATIMEN/resulta2.shp")
dfEmp['type_batim']=dfEmp['id_batimen'].apply(lambda x :20 if len(dfff[dfff['FID']==x]['ClasePredi'])==0 else dfff[dfff['FID']==x]['ClasePredi'].values[0]) 
dfEmp=dfEmp[dfEmp['type_batim']!=20]
################################################################################################################################################

#on importe les adresses de yamaska
adresse = gpd.read_file(pathADR)
adresse=adresse.drop(columns=['version_aq', 'date_modif', 'no_civq',
       'no_civq_su', 'seq_odo', 'odo_gener', 'odo_parti', 'odo_speci',
       'odo_orien', 'odo_rec_lo', 'no_arr'])
adresse=adresse.drop(columns=['version', 'code_mun', 'norm_suffi', 'norm_gener',
       'norm_odony', 'norm_orien','nb_unite', 'id_rte',
       'cote_rte', 'qualif'])


#on mémorise le crs.
crss=adresse.crs


#on importe les empreintes de quebec
dfEmpquebec=gpd.read_file(pathEmpquebec)
#on merge les dataframe
#on utilise le même crs que pour les adresses. 
dfEmp=dfEmp.to_crs(crss)
#on utilise le même crs que pour les adresses. 
dfEmpquebec=dfEmpquebec.to_crs(crss)
dfEmpF = pd.concat([dfEmp, dfEmpquebec], axis=0, join="inner",ignore_index=True)
dfEmp=dfEmpF

#on importe les adresses de yamaska
adressequebec = gpd.read_file(pathADRquebec)
adressequebec=adressequebec.drop(columns=['OBJECTID','Seqodo', 'CodeMun', 'CodeArr', 'NoLot','NoSqNoCivq', 'DateModif', 'NoCivq', 'NoCivqSuf'])
adressequebec=adressequebec.drop(columns=['Version', 'Statut', 'CaractAdr', 'NbUnite',
       'IdRte', 'CoteRte'])
adressequebec=adressequebec.drop(columns=['Qualif'])
adressequebec=adressequebec.to_crs(crss)
#on merge le tt
adresse.columns=adressequebec.columns
adresse = pd.concat([adresse, adressequebec], axis=0, join="inner",ignore_index=True)

#on crée une zone avec laquelle on pourra filtrer les autres jeux de données utilisées (pour gagner en performances).
h=dfEmp.buffer(1000).unary_union

#on ne garde que les adresses proches des empreintes, inutile de calculer les autres adresses.
adresse=adresse[adresse.within(h)]

#on importe le cadastre. 
cadastre = gpd.read_file(pathCadastre)
cadastre=cadastre.drop(columns=['ogc_fid', 'fid'])
cadastre=cadastre.to_crs(crss)


#on importe le cadastre. 
cadastrequebec = gpd.read_file(pathCadastrequebec)
cadastrequebec=cadastrequebec.drop(columns=['OBJECTID', 'CO_TYPE_PO', 'CO_TYPE_DI', 'CO_ECHEL_C',
       'CO_ECHEL_R', 'CO_ECHEL_1', 'NB_COORD_X', 'NB_COORD_Y', 'NB_ANGLE_N',
       'CO_INDIC_F', 'VA_SUPRF_L', 'VA_SUPRF_1', 'CO_TYPE_UN', 'NO_PLAN_CO',
       'CO_CIRCN_F', 'NM_CIRCN_F', 'DA_DEPOT_C', 'NO_FEUIL_C', 'DA_MISE_VI',
       'DH_DERNR_M', 'SHAPE_Leng', 'SHAPE_Area'])
cadastrequebec=cadastrequebec.to_crs(crss)
cadastre = pd.concat([cadastre, cadastrequebec], axis=0, join="inner",ignore_index=True)
#on utilise le même crs que pour les adresses.
cadastre=cadastre.to_crs(crss)
#on ne garde que les données proches des empreintes.
cadastre=cadastre[cadastre.within(h)]

#on charge le role
roleData=gpd.read_file(PAthRole)
roleData=roleData.to_crs(crss)
roleData=roleData.drop(columns=['id_up', 'vers', 'no_lot', 'nb_lot', 'co_util', 'id_ue',
       'desc_up', 'id_prop', 'type_prop'])


roleDataquebec=gpd.read_file(PAthRolequebec)
roleDataquebec=roleDataquebec.to_crs(crss)
roleDataquebec=roleDataquebec.drop(columns=['id_up', 'vers', 'no_lot', 'nb_lot', 'co_util', 'id_ue',
       'desc_up', 'id_prop', 'type_prop'])


roleData = pd.concat([roleData, roleDataquebec], axis=0, join="inner",ignore_index=True)
#on ne garde que les données proches des empreintes.
roleData=roleData[roleData.within(h)]

#on importe les routes.
route = gpd.read_file(pathRoute)

#on utilise le même crs que pour les adresses.
route=route.to_crs(crss)
#on garde que les données proches des empreintes.
route=route[route.within(h)]
########################################################

#fonction retournant les informations de l'élément qui contient la geometrie. #usuellement, les informations de la parcelle de l'empreinte.
def parcelleGet(point,gpd2,ColName):
    rt=gpd2.sindex.query(point.buffer(100))
    gpd2=gpd2.loc[rt]
    return gpd2[gpd2['geometry'].intersects(point)][ColName].values
    

#on merge le cadastre et le role :
cadastre=cadastre.reset_index(drop=True)
roleData=roleData.reset_index(drop=True)
cadastre["centre"]=cadastre["geometry"].centroid
cadastre['desc_usage']=cadastre["centre"].apply(lambda x:parcelleGet(x,roleData,"desc_usage"))
cadastre['desc_usage']=cadastre['desc_usage'].apply(lambda x: x[0] if len(x)>0 else 'nane')

################testing mode##################
#pour tester sans trop de calcul.
#dfEmp=dfEmp[0:1000]
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
    if d3['Categorie'].unique().size == 0:
        return ['OMG']
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
dfEmp=dfEmp.reset_index(drop=True)
dfEmp=dfEmp[['id_batimen','geometry','type_batim']]

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

#on recupere l usage
dfEmp["desc_usage"]=dfEmp["parcelle"].apply(lambda x : cadastre[x==cadastre['NO_LOT']]['desc_usage'].iloc[0] if len(x)>0 else 'nane')

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


adresse['categorie']=adresse['Categorie']

#catégorie de l'adresse la plus proche de l'empreinte.
dfEmp["categAdr"]=dfEmp["idadr"].apply(lambda x : adresse[adresse["IdAdr"]==x]['Categorie'].values[0])

#distance du batiment le plus proche dans la même parcelle.
dfEmp["DistBatCloserDansParcelle"]=dfEmp.apply(lambda x: min_distIndex(x["geometry"],dfEmp[(dfEmp['parcelle']==x['parcelle'])  & (dfEmp['id_batimen']!=x['id_batimen'])],'distAdr')["distAdr"] if len(dfEmp[(dfEmp['parcelle']==x['parcelle'])& (dfEmp['id_batimen']!=x['id_batimen'])])>1 else 0,axis=1)


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
dfEmp['categPred']=dfEmp.apply(lambda x: "G" if ((x['categPred']=='M2') and (x["batimentDansPArcelleSurface"]>x['surface']) and (MoyenCalcultateur3(cadastre,x["geometriDeLaparcelle"],adresse)[0] in ("R","V"))) else x['categPred']   ,axis=1)#(MoyenCalcultateur3(cadastre,x["geometriDeLaparcelle"],adresse)[0]
dfEmp['categPred']=dfEmp.apply(lambda x: "R" if ((x['categPred']=='M2') and (x["batimentDansPArcelleSurface"]<x['surface']) and (MoyenCalcultateur3(cadastre,x["geometriDeLaparcelle"],adresse)[0] in ("R","V"))) else x['categPred']   ,axis=1)#MoyenCalcultateur3(cadastre,x["geometriDeLaparcelle"],adresse)[0]
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
           'cls_foncti','type_batim',
            'alignement','ZONEinstitutionelle','Sommet',
           'densiter', 'nbAdressDansParcelle', 'densiterProche',#ajouter ici le nom des nouvelle feature
           'elongement',"categAdr","categPred", 'distAdrCentroid','densitertresProche',
            "desc_usage"]]
dfEmp=dfEmp.drop(columns=["parcelle"])

#on ajoute une borne superieur
dfEmp['surface']=filtre('surface',1000,dfEmp)
dfEmp['distRout']=filtre('distRout',100,dfEmp)
dfEmp['nbBatimentDansLaParcelle']=filtre('nbBatimentDansLaParcelle',10,dfEmp)
dfEmp['Surfaceparcelle']=filtre('Surfaceparcelle',10000,dfEmp)
dfEmp['nbAdressDansParcelle']=filtre('nbAdressDansParcelle',15,dfEmp)
dfEmp['distAdrCentroid']=filtre('distAdrCentroid',70,dfEmp)
dfEmp['R']=filtre('R',15,dfEmp)
dfEmp["surfaceAutreBatiment"]=filtre('surface',1000,dfEmp)

#on sauvegarde le resultat.
dfEmp.to_csv(OutpoutFilename)

























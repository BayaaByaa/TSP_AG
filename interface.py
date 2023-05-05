import streamlit as st
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#st.title("L’algorithme génétique pour la résolution du TSP ")
st.write("<center><h1>L’algorithme génétique pour la résolution du TSP</h1></center>", unsafe_allow_html=True)

generations=[]
dist=[]  

def population_innitiale (n, TAILLE_POP):
      population = [random.sample(range(1,n), n-1) for _ in range(TAILLE_POP)]
      return population

def fitness(x, distances):
  d= distances[0][x[0]]
  for i in range(1,len(x) - 1):
    d += distances[x[i]][x[i+1]]
  #distance entre la derniere ville et celle de départ
  d += distances[x[-1]][x[0]]
  return (d) 

def selection (population, distances):
    # Sélection par roulette
    scores_fitness = [1/fitness(x, distances) for x in population]
    # Calcul de la proba de sélection
    sum_fitness = sum(scores_fitness)
    proba_fitness = [score/sum_fitness for score in scores_fitness]
    # Sélection des parents selon la proba_fitness
    parent1 = random.choices(population, weights=proba_fitness)[0] 
    parent2 = random.choices(population, weights=proba_fitness)[0]
    return parent1, parent2

def croisement (parent1, parent2):
    # Croisement uniforme (le point de croisement est aléatoire)
    pt_aleatoire = random.randint(1, len(parent1) - 1)
    # Création des enfants 
    enfant1 = parent1[:pt_aleatoire] + [gene for gene in parent2 if gene not in parent1[:pt_aleatoire]]
    enfant2 = parent2[:pt_aleatoire] + [gene for gene in parent1 if gene not in parent2[:pt_aleatoire]]
    return enfant1, enfant2

def mutation (x,TAUX_MUTATION):
  #methode de permutation
    for i in range(len(x)):
        if random.random() < TAUX_MUTATION: # vérification si l'individu actuel doit subir une mutation
            j = random.randint(0, len(x) - 1)
            x[i], x[j] = x[j], x[i] 
    return x

# Fonction de recherche de la meilleure solution (distance la plus courte)
def bestSolution(pop,d): 
  bestValue=fitness(pop[0],d)
  xBest=pop[0]
  for x in pop: 
    if bestValue>fitness(x,d): 
        bestValue=fitness(x,d)
        xBest=x
  return bestValue,xBest

def AG_TSP (n,distances, TAILLE_POP, NBR_GENERATIONS, TAUX_MUTATION):
  
  # Création de la population initiale
  population= population_innitiale (n, TAILLE_POP)
  # Générations
  for generation in range(NBR_GENERATIONS):
      bestValue,xBest=bestSolution(population,distances)
      #print("La meilleure valeur de la generation",generation," est:",bestValue)
      generations.append(generation)
      dist.append(bestValue)
      nouvelle_population=[]
      while len(nouvelle_population) < TAILLE_POP:
        #phase de selection
          parent1, parent2 = selection (population, distances)
        #phase de croisement
          enfant1, enfant2 = croisement (parent1, parent2)
        #phase de mutaion 
          enfant1 = mutation(enfant1,TAUX_MUTATION)
          enfant2 = mutation(enfant2,TAUX_MUTATION)
        #création de la nouvelle génération
          nouvelle_population.append(enfant1)
          nouvelle_population.append(enfant2)
      population=population+nouvelle_population
      pop=[]
      for generation in range(NBR_GENERATIONS//2):
          p1,p2=selection (population, distances)
          pop.append(p1)
          pop.append(p2)
      pop.append(xBest)
      population=pop
  return bestSolution(nouvelle_population,distances)
  
# Définir les options de choix
options = ["Distances aléatoires", "Distances via fichier CSV"]

# Demander à l'utilisateur de choisir une option
choix = st.sidebar.radio("Choisir le mode", options)

# Afficher la page sélectionnée
if choix == "Distances aléatoires":
    st.subheader('Données:')
    col1,col2,col3,col4 = st.columns([1,1,1,1])
    with col1:
            st.number_input(':blue[NOMBRE DE VILLES:]',key='n', value=0, step=1, format="%d")
    with col2:
            st.number_input(':blue[TAILLE DE LA POPULATION:]',key='TAILLE_POP', value=0, step=1, format="%d")
    with col3:
            st.number_input(':blue[NOMBRE DE GENERATIONS:]',key='NBR_GENERATIONS',value=0, step=1, format="%d")
    with col4:
            st.number_input(':blue[TAUX DE MUTATION:]',key='TAUX_MUTATION',value=0.0, step=0.1)

    
    distances = np.zeros(( st.session_state['n'], st.session_state['n']))
    for i in range( st.session_state['n']):
        for j in range( st.session_state['n']):
            if i == j:
                distances[i][j] = 0
            elif i < j:
                distances[i][j] = random.randint(20, 200)
                distances[j][i] = distances[i][j]
    distances = distances.astype(int)

    def affichage():
        st.subheader(":blue[La meilleure solution est:]"+" "+ str(dist[-1])+"km")
        st.subheader("Evolution de la meilleure solution en fonction des générations:")
        fig, ax = plt.subplots()
        ax.plot(generations, dist)
        ax.set_xlabel('Générations')
        ax.set_ylabel('Meilleures solutions (en km)')
        st.pyplot(fig)
        st.subheader("Matrice des distances:")
        st.table(distances)

    if st.button('START'):
        AG_TSP(st.session_state['n'],distances, st.session_state['TAILLE_POP'], st.session_state['NBR_GENERATIONS'], st.session_state['TAUX_MUTATION'])
        affichage()
else:
    # Titre de la page
    st.subheader('Données:')

    # Demander à l'utilisateur de télécharger le fichier CSV
    file = st.file_uploader(':blue[IMPORTER UN FICHIER CSV]', type=['csv'])

    # Vérifier si un fichier a été téléchargé
    if file is not None:
        # Lire le fichier CSV en utilisant Pandas
        d = pd.read_csv(file)

        # Afficher les données dans un DataFrame
        st.subheader('Matrice des distances:')
        st.write(d)
        n = d.shape[0]
        distances = d.values.tolist()

        col2,col3,col4 = st.columns([1,1,1])
        with col2:
                st.number_input(':blue[TAILLE DE LA POPULATION:]',key='TAILLE_POP', value=0, step=1, format="%d")
        with col3:
                st.number_input(':blue[NOMBRE DE GENERATIONS:]',key='NBR_GENERATIONS',value=0, step=1, format="%d")
        with col4:
                st.number_input(':blue[TAUX DE MUTATION:]',key='TAUX_MUTATION',value=0.0, step=0.1)

        def affichage():
            st.subheader(":blue[La meilleure solution est:]"+" "+ str(dist[-1])+"km")
            st.subheader("Evolution de la meilleure solution en fonction des générations:")
            fig, ax = plt.subplots()
            ax.plot(generations, dist)
            ax.set_xlabel('Générations')
            ax.set_ylabel('Meilleures solutions (en km)')
            st.pyplot(fig)
            
        if st.button('START'):
            AG_TSP(n,distances, st.session_state['TAILLE_POP'], st.session_state['NBR_GENERATIONS'], st.session_state['TAUX_MUTATION'])
            affichage()
    
        
          
        
    









        









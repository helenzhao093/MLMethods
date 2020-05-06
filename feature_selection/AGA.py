import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import base 
import random
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

class AGA():
    def __init__(self, clf, m, cv=None, metric='f1', crossover='single', p_crossover=0.6, p_mutation=0.1, max_generations=5, tol=0.01, adaptive=True):
        self.base_clf = clf
        self.m = m
        self.cv = cv
        self.metric = metric
        self.crossover = crossover
        self.max_generations = max_generations
        self.tol = tol
        self.adaptive = adaptive
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
    
    def init_population(self, num_features):
        # initialize m binary feature selections
        feature_indices = range(num_features)
        self.population = []
        for i in range(self.m):
            selected = np.random.choice(feature_indices, 
                                        int(random.uniform(0.3, 0.7) * num_features),
                                        replace=False)
            self.population.append(np.array([True if i in selected else False for i in range(num_features)]))
    
    def calculate_fitness_cv(self, X, y):
        self.fitnesses = []
        for indiv in self.population:
            scores = []
            for train_index, test_index in self.cv.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf = base.clone(self.base_clf)
                clf.fit(X_train[:,indiv], y_train)
                y_pred = clf.predict(X_test[:,indiv])
                scores.append(self.get_score(self.metric, y_test, y_pred))
            self.fitnesses.append(np.mean(scores)) 
        self.fitnesses = np.array(self.fitnesses)
        self.aggregate_fitness()
    
    def calculate_fitness(self, X_train, X_test, y_train, y_test):
        # can change this to 5 fold cross validation fitness 
        self.fitnesses = []
        for indiv in self.population:
            clf = base.clone(self.base_clf)
            clf.fit(X_train[:,indiv], y_train)
            y_pred = clf.predict(X_test[:,indiv])
            self.fitnesses.append(self.get_score(self.metric, y_test, y_pred))
        self.fitnesses = np.array(self.fitnesses)
        self.aggregate_fitness()
        
    def aggregate_fitness(self):
        self.max_fitness = np.max(self.fitnesses)
        self.sum_fitness = np.sum(self.fitnesses)
        self.avg_fitness = np.mean(self.fitnesses)
        ranking = self.fitnesses.argsort().argsort() * 1.5
        sum_rank = ranking.sum() * 1.0
        self.proba = np.array([rank/sum_rank for rank in ranking])
        
    def cal_crossover_proba(self, i, j):
        if self.adaptive == False:
            return self.p_crossover
        k1 = max(0.9, self.p_crossover)
        k2 = min(0.9, self.p_crossover)
        f = self.fitnesses[i] if self.fitnesses[i] > self.fitnesses[j] else self.fitnesses[j]
        if f >= self.avg_fitness:
            return k1 * (self.max_fitness - f) / (self.max_fitness - self.avg_fitness)
        else:
            return k2
    
    def cal_mutation_proba(self, i, j):
        if self.adaptive == False:
            return self.p_mutation
        k3 = max(self.p_mutation, 0.001)
        k4 = min(self.p_mutation, 0.001)
        f = self.fitnesses[i] if self.fitnesses[i] > self.fitnesses[j] else self.fitnesses[j]
        if f >= self.avg_fitness:
            return k3 * (self.max_fitness - f) / (self.max_fitness - self.avg_fitness)
        else:
            return k4
        
    def single_point_crossover(self, parent1, parent2):
        crossover_index = random.randint(0, len(parent1)-1)
        offspring1 = np.concatenate((parent1[:crossover_index], 
                               parent2[crossover_index:])),
        offspring2 = np.concatenate((parent2[:crossover_index], 
                               parent1[crossover_index:]))
        return [offspring1[0], offspring2]
    
    def two_point_crossover(self, parent1, parent2):
        crossover_index_1 = random.randint(0, len(parent1)-1)
        crossover_index_2 = random.randint(0, len(parent1)-1)
        index1 = crossover_index_1 if crossover_index_2 > crossover_index_1 else crossover_index_2
        index2 = crossover_index_2 if crossover_index_2 > crossover_index_1 else crossover_index_1
        offspring1 = np.concatenate((parent1[:index1], 
                               parent2[index1:index2],
                               parent1[index2:]))
        offspring2 = np.concatenate((parent2[:index1], 
                               parent1[index1:index2],
                               parent2[index2:]))
        return [offspring1, offspring2]

    def uniform_crossover(self, parent1, parent2):
        for i in range(len(parent1)):
            if random.random() < 0.5:
                temp = parent1[i]
                parent1[i] = parent2[i]
                parent2[i] = temp
        return [parent1, parent2]
    
    def rank_selection(self, population_indices):
        return np.random.choice(population_indices, 2, p=self.proba)
    
    def roulette_selection(self):
        select_sum = random.randint(int(self.sum_fitness))
        
    def mutate(self, p_m, offspring):
        for j in range(len(offspring)):
            if random.random() <= p_m:
                offspring[j] = not offspring[j]
        return offspring
        
    def create_new_population(self, num_features):
        population = []
        population_indices = range(len(self.population))
        for _ in range(int(self.m/2)):
            parent_indices = self.rank_selection(population_indices)
            p_c = self.cal_crossover_proba(parent_indices[0], parent_indices[1])
            p_m = self.cal_mutation_proba(parent_indices[0], parent_indices[1])
            # crossover
            if random.random() <= p_c:
                if self.crossover == 'single':
                    offsprings = self.single_point_crossover(self.population[parent_indices[0]], self.population[parent_indices[1]])
                elif self.crossover == 'two':
                    offsprings = self.two_point_crossover(self.population[parent_indices[0]], self.population[parent_indices[1]])
                elif self.crossover == 'uniform':
                    offsprings = self.uniform_crossover(self.population[parent_indices[0]], self.population[parent_indices[1]])
            else:
                offsprings = [self.population[parent_indices[0]], self.population[parent_indices[1]]]
                
            # mutation
            for offspring in offsprings:
                self.mutate(p_m, offspring)
                population.append(offspring)
            
        self.population = population 
    
    def fit(self, X, y):
        num_features = len(X[0])
        self.init_population(num_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        if self.cv is None:
            self.calculate_fitness(X_train, X_test, y_train, y_test)
        else:
            self.calculate_fitness_cv(X, y)
        
        current_max_fitness = self.max_fitness
        num_tol = 0
        for _ in range(self.max_generations):
            self.create_new_population(num_features)
            if self.cv is None:
                self.calculate_fitness(X_train, X_test, y_train, y_test)
            else:
                self.calculate_fitness_cv(X, y)
            
            # stopping criteria
            if (self.max_fitness > 0.93):
                return 
            
            # performance did not increase
            if (self.max_fitness - current_max_fitness < self.tol):
                num_tol += 1 
                current_max_fitness = self.max_fitness
                if num_tol == 5:
                    return
    
    def transform(self, X):
        max_fitness_index = np.argmax(self.fitnesses)
        print(max_fitness_index)
        return X[:,self.population[max_fitness_index]]
            
    def get_score(self, metric, y, y_pred):
        if metric == 'auc':
            return roc_auc_score(y, y_pred)
        if metric == 'f1':
            return f1_score(y, y_pred)
        return accuracy_score(y, y_pred)
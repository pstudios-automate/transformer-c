// evolutionary_weight_loader.h
#ifndef EVOLUTIONARY_WEIGHT_LOADER_H
#define EVOLUTIONARY_WEIGHT_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Evolutionary parameters
#define MUTATION_RATE 0.01
#define MUTATION_STRENGTH 0.1
#define CROSSOVER_RATE 0.3
#define GENERATION_SIZE 10
#define SELECTION_RATE 0.2  // Keep top 20% for next generation

typedef struct {
    double *weights;
    int count;
    double fitness;
} Chromosome;

typedef struct {
    Chromosome *population;
    int population_size;
    int generation;
    double best_fitness;
    Chromosome best_chromosome;
} EvolutionContext;

// Core evolutionary functions
Chromosome create_random_chromosome(int weight_count);
void mutate_chromosome(Chromosome *chromo, double rate, double strength);
Chromosome crossover_chromosomes(Chromosome *parent1, Chromosome *parent2, double rate);
void evaluate_fitness(Chromosome *chromo);  // You'll need to implement this
void evolve_population(EvolutionContext *context);
void save_best_chromosome(EvolutionContext *context, const char *base_filename);
Chromosome load_or_create_chromosome(const char *base_filename, int weight_count);

#endif
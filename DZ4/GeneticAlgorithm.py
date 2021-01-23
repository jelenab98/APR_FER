import numpy as np
import random


class GeneticAlgorithm:
    def __init__(self, function, crossovers, mutations, k=3, population_size=1000, gene_size=2, p_mutation=0.2,
                 max_iter=100000, epsilon=1e-6, interval=(-50, 50), precision=1, printing=True, num_of_bits=None):
        self.f = function
        self.gene_size = gene_size
        self.size = population_size
        self.k = k
        self.p_mutation = p_mutation
        self.max_iters = max_iter
        self.epsilon = epsilon
        self.crossovers = crossovers
        self.mutations = mutations
        self.interval = interval
        self.precision = precision
        if num_of_bits is None:
            self.num_of_bits = np.ceil((np.log10(np.floor(1 + (interval[1]-interval[0])*10**precision)))/(np.log10(2)))
            self.num_of_bits = int(self.num_of_bits)

        else:
            self.num_of_bits = num_of_bits
        self.population = np.zeros((population_size, gene_size))
        self.evals = None
        self.best = 0
        self.min_error = np.inf
        self.printing = printing

    def select_and_reproduce(self):
        raise NotImplementedError

    def generate_population(self):
        raise NotImplementedError

    def calculate(self):
        self.generate_population()
        i = 0
        while i < self.max_iters and self.min_error > self.epsilon:
            self.select_and_reproduce()
            self.best = np.argmin(self.evals)
            self.min_error = self.evals[self.best]
            if i % 10000 == 0 and self.printing:
                print("Iter: {}, min_error: {}\n\tbest_gene: {}".format(i, self.min_error,
                                                                      self.population[self.best]))
            i += 1

        print("Finished! Iter: {}, min_error:{}\n\tbest_gene: {}".format(i, self.min_error, self.population[self.best]))


class BinaryGA(GeneticAlgorithm):

    def calculate(self):
        self.generate_population()
        i = 0
        while i < self.max_iters and self.min_error > self.epsilon:
            self.select_and_reproduce()
            self.best = np.argmin(self.evals)
            self.min_error = self.evals[self.best]
            if i % 10000 == 0 and self.printing:
                print("Iter: {}, min_error: {}\nbest_gene: {}".format(i, self.min_error,
                                                                      self.fp_to_bin(self.population[self.best])))
            i += 1

        print("Finished! Iter: {}, min_error:{}\nbest_gene: {}\n best_gene in FP:{}".format(
            i, self.min_error, self.fp_to_bin(self.population[self.best]), self.population[self.best]))

    def single_crossover(self, idx_parent1, idx_parent2):
        parent1 = self.population[idx_parent1, :]
        parent2 = self.population[idx_parent2, :]
        parent1_bin = self.fp_to_bin(parent1)
        parent2_bin = self.fp_to_bin(parent2)
        break_index = np.random.randint(0, self.gene_size-1)
        child1_bin = parent1_bin.copy()
        child2_bin = parent2_bin.copy()
        child1_bin[break_index:] = parent2_bin[break_index:]
        child2_bin[break_index:] = parent1_bin[break_index:]
        child1 = self.bin_to_fp(child1_bin)
        child2 = self.bin_to_fp(child2_bin)
        if self.f.get_fitness(np.expand_dims(child1, axis=0)) < self.f.get_fitness(np.expand_dims(child2, axis=0)):
            return child1_bin
        else:
            return child2_bin

    def multi_point_crossover(self, idx_parent1, idx_parent2):
        parent1 = self.population[idx_parent1, :]
        parent2 = self.population[idx_parent2, :]
        parent1_bin = self.fp_to_bin(parent1)
        parent2_bin = self.fp_to_bin(parent2)
        number_of_breaks = np.random.randint(1, self.gene_size)
        break_index = np.random.randint(0, self.gene_size-1, number_of_breaks)
        child1_bin = parent1_bin.copy()
        child2_bin = parent2_bin.copy()
        for i in range(number_of_breaks):
            child1_bin[break_index[i:i+1]] = parent2_bin[break_index[i:i+1]]
            child2_bin[break_index[i:i+1]] = parent1_bin[break_index[i:i+1]]
        child1 = self.bin_to_fp(child1_bin)
        child2 = self.bin_to_fp(child2_bin)
        if self.f.get_fitness(np.expand_dims(child1, axis=0)) < self.f.get_fitness(np.expand_dims(child2, axis=0)):
            return child1_bin
        else:
            return child2_bin

    def uniform_crossover(self, idx_parent1, idx_parent2):
        parent1 = self.population[idx_parent1, :]
        parent2 = self.population[idx_parent2, :]
        parent1_bin = self.fp_to_bin(parent1)
        parent2_bin = self.fp_to_bin(parent2)
        r = np.random.randint(0, 1, (self.num_of_bits, ))
        a_xor_b = np.logical_xor(parent1_bin, parent2_bin).astype(int)
        child = np.add(np.multiply(parent1_bin, parent2_bin), np.multiply(r, a_xor_b))
        return child

    def simple_mutation(self, gene):
        index = np.random.randint(0, self.num_of_bits-1)
        if np.random.uniform(0, 1) < self.p_mutation:
            for d in range(self.gene_size):
                gene[d, index] = (gene[d, index] + 1) % 2
        return self.bin_to_fp(gene)

    def uniform_mutation(self, gene):
        for d in range(self.gene_size):
            for i in range(self.num_of_bits):
                if np.random.uniform(0, 1) < self.p_mutation:
                    gene[d, i] = (gene[d, i] + 1) % 2
        return self.bin_to_fp(gene)

    def generate_population(self):
        self.population = np.random.uniform(self.interval[0], self.interval[1], (self.size, self.gene_size))
        self.evals = self.f.get_fitness(self.population)

    def select_and_reproduce(self):
        indices = random.sample(range(self.size), self.k)
        f_i = self.evals[indices]
        idx_worst = indices[np.argmax(f_i)]
        f_i = np.reshape(f_i, (self.k,))
        idx_parent1 = indices[np.argpartition(f_i, 2)[0]]
        idx_parent2 = indices[np.argpartition(f_i, 2)[1]]

        f_cross = random.randint(0, len(self.crossovers) - 1)
        if self.crossovers[f_cross] == 'single':
            child = self.single_crossover(idx_parent1, idx_parent2)
        elif self.crossovers[f_cross] == 'multi':
            child = self.multi_point_crossover(idx_parent1, idx_parent2)
        else:
            child = self.uniform_crossover(idx_parent1, idx_parent2)

        f_mutation = random.randint(0, len(self.mutations) - 1)
        if self.mutations[f_mutation] == "uniform":
            child = self.uniform_mutation(child)
        else:
            child = self.simple_mutation(child)

        child = np.expand_dims(child, axis=0)
        value = self.f.get_fitness(child)

        self.population[idx_worst, :] = child
        self.evals[idx_worst] = value
        return

    def fp_to_bin(self, gene):
        gene_bin = np.zeros((self.gene_size, self.num_of_bits), dtype=int)
        for i in range(self.gene_size):
            b = np.round((2**self.num_of_bits - 1)*(gene[i] - self.interval[0])/(self.interval[1] - self.interval[0])).astype(int)
            b_bin = np.array(list(np.binary_repr(b, width=self.num_of_bits)), dtype=int)
            gene_bin[i] = b_bin.copy()
        return gene_bin

    def bin_to_fp(self, gene):
        gene_fp = np.zeros(shape=(self.gene_size,), dtype=float)
        for i in range(self.gene_size):
            b = gene[i]
            b = b.dot(2**np.arange(b.size)[::-1])
            x = self.interval[0] + (self.interval[1] - self.interval[0])*b/(2**self.num_of_bits - 1)
            gene_fp[i] = x
        return gene_fp


class DiscreteGA(GeneticAlgorithm):
    def generate_population(self):
        self.population = np.random.uniform(self.interval[0], self.interval[1], (self.size, self.gene_size))
        self.evals = self.f.get_fitness(self.population)

    def one_point_crossover(self, idx_parent1, idx_parent2):
        parent1 = self.population[idx_parent1, :]
        parent2 = self.population[idx_parent2, :]
        break_index = np.random.randint(0, self.gene_size)
        child = parent1.copy()
        child[break_index:] = parent2[break_index:]
        return child

    def discrete_recombination(self, idx_parent1, idx_parent2):
        parent1 = self.population[idx_parent1, :]
        parent2 = self.population[idx_parent2, :]
        child = np.array([np.random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)])
        return child

    def simple_arithmetic_recombination(self, idx_parent1, idx_parent2):
        parent1 = self.population[idx_parent1, :]
        parent2 = self.population[idx_parent2, :]
        break_index = np.random.randint(0, self.gene_size)
        child = parent1.copy()
        a = np.random.uniform(0, 1)
        child[break_index:] = a * parent1[break_index:] + (1 - a) * parent2[break_index:]
        return child

    def single_arithmetic_recombination(self, idx_parent1, idx_parent2):
        parent1 = self.population[idx_parent1, :]
        parent2 = self.population[idx_parent2, :]
        child = parent1.copy()
        break_index = np.random.randint(0, self.gene_size)
        a = np.random.uniform(0, 1)
        child[break_index] = a * child[break_index] + (1 - a) * parent2[break_index]
        return child

    def whole_arithmetic_recombination(self, idx_parent1, idx_parent2):
        parent1 = self.population[idx_parent1, :]
        parent2 = self.population[idx_parent2, :]
        a = np.random.uniform(0, 1)
        child = a * parent1 + (1 - a) * parent2
        return child

    def heuristic_recombination(self, idx_parent1, idx_parent2):
        parent1 = self.population[idx_parent1, :]
        parent2 = self.population[idx_parent2, :]
        if self.f.get_fitness(np.expand_dims(parent1, axis=0)) < self.f.get_fitness(np.expand_dims(parent2, axis=0)):
            a = np.random.uniform(0, 1)
            child = a*(parent1 - parent2) + parent1
        else:
            a = np.random.uniform(0, 1)
            child = a*(parent2 - parent1) + parent2
        return child

    def uniform_mutation(self, gene):
        for i in range(self.gene_size):
            if np.random.uniform(0, 1) <= self.p_mutation:
                gene[i] = np.random.uniform(self.interval[0], self.interval[1])
        return gene

    def gauss_mutation(self, gene):
        for i in range(self.gene_size):
            if np.random.uniform(0, 1) <= self.p_mutation:
                gene[i] = np.random.normal(gene[i], 1)
        return gene

    def select_and_reproduce(self):
        indices = random.sample(range(self.size), self.k)
        f_i = self.evals[indices]
        idx_worst = indices[np.argmax(f_i)]
        f_i = np.reshape(f_i, (self.k,))
        idx_parent1 = indices[np.argpartition(f_i, 2)[0]]
        idx_parent2 = indices[np.argpartition(f_i, 2)[1]]

        f_cross = random.randint(0, len(self.crossovers)-1)
        if self.crossovers[f_cross] == 'discrete':
            child = self.discrete_recombination(idx_parent1, idx_parent2)
        elif self.crossovers[f_cross] == 'whole_arithmetic':
            child = self.whole_arithmetic_recombination(idx_parent1, idx_parent2)
        elif self.crossovers[f_cross] == 'single':
            child = self.single_arithmetic_recombination(idx_parent1, idx_parent2)
        elif self.crossovers[f_cross] == 'simple':
            child = self.simple_arithmetic_recombination(idx_parent1, idx_parent2)
        else:
            child = self.one_point_crossover(idx_parent1, idx_parent2)
        f_mutation = random.randint(0, len(self.mutations)-1)
        if self.mutations[f_mutation] == "uniform":
            child = self.uniform_mutation(child)
        else:
            child = self.gauss_mutation(child)
        child = np.expand_dims(child, axis=0)
        value = self.f.get_fitness(child)

        self.population[idx_worst, :] = child
        self.evals[idx_worst] = value

        return

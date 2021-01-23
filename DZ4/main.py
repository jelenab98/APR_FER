from GeneticAlgorithm import *
from Functions import *
import json


def get_configs(input_file):
    with open(input_file, 'r') as f:
        data_configs = json.load(f)
    return data_configs


def write_configs(output_file):
    data = {"type": "discrete", "function": 2,
            "crossovers": ["whole-arithmetic", "single", "simple", "discrete", "one-point"],
            "mutations": ["uniform", "gauss"],
            "k": 3, "population": 1000, "variables": 3, "p_mut": 0.5, "max_iter": 1e5, "epsilon": 1e-6,
            "interval": (-50, 150), "printing": True}

    with open(output_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':

    # write_configs('configs.json')

    configs = get_configs('configs.json')

    if configs['type'] == 'bin':
        fn = configs['function']
        if fn == 1:
            f = F1()
        elif fn == 2:
            f = F2()
        elif fn == 6:
            f = F6()
        else:
            f = F7()
        ga = BinaryGA(function=f, crossovers=configs['crossovers'], mutations=configs['mutations'], k=configs['k'],
                      population_size=configs['population'], gene_size=configs['variables'],
                      p_mutation=configs['p_mut'], max_iter=configs['max_iter'], epsilon=configs['epsilon'],
                      interval=configs['interval'], precision=configs['precision'], printing=configs['printing'])
    else:
        fn = configs['function']
        if fn == 1:
            f = F1()
        elif fn == 2:
            f = F2()
        elif fn == 6:
            f = F6()
        else:
            f = F7()
        ga = DiscreteGA(function=f, crossovers=configs['crossovers'], mutations=configs['mutations'], k=configs['k'],
                        population_size=configs['population'], gene_size=configs['variables'],
                        p_mutation=configs['p_mut'], max_iter=configs['max_iter'], epsilon=configs['epsilon'],
                        interval=configs['interval'], printing=configs['printing'])

    ga.calculate()

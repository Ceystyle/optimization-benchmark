"""Compare performance of different optimization algorithms."""

import numpy as np
import matplotlib.pyplot as plt
import functions
import optimizers
import inspect

np.seterr(all='raise')

algorithms = []
for name, obj in inspect.getmembers(optimizers):
    if inspect.isclass(obj):
        algorithms.append(name)

benchmarks = []
for name, obj in inspect.getmembers(functions):
    if inspect.isclass(obj):
        benchmarks.append(name)

dimensions = [2, 3, 4]
parameters = {
    'StupidGradientDescent': [0.001], 'MomentumGradientDescent': [0.001, 0.9],
    'NesterovGradientDescent': [0.001, 0.9]
}

MAX_BUDGET = 5000
EPSILON = 0.1
performance = dict()

# Get minimal steps for all benchmark / algorithm combinations
for algorithm in algorithms:
    performance[algorithm] = dict()
    for dim in dimensions:
        performance[algorithm][dim] = dict()
        for benchmark in benchmarks:
            bm = getattr(functions, benchmark)(dim=dim)
            np.random.seed(seed=42)
            initstate = 10*(np.random.rand(dim,  1)-1)[:, 0]
            alg = getattr(optimizers, algorithm)(*parameters[algorithm], initstate, bm.grad)
            # alg = getattr(optimizers, algorithm)(initstate, bm.grad)
            for step in range(MAX_BUDGET * dim):
                try:
                    alg.step()
                except FloatingPointError:
                    performance[algorithm][dim][benchmark] = np.inf
                    break

                if np.max(np.linalg.norm(alg.state - bm.optimal()[0])) < EPSILON:
                    performance[algorithm][dim][benchmark] = step
                    break
            # If no entry for algorithm is available, it didn't converge
            try:
                performance[algorithm][dim][benchmark]
            except KeyError:
                performance[algorithm][dim][benchmark] = np.inf

#print(performance)

# Create overall ranking
ranking = dict()
for algorithm in algorithms:
    ranking[algorithm] = dict()
    for dim in dimensions:
        ranking[algorithm][dim] = np.zeros(MAX_BUDGET)
        for benchmark in benchmarks:
            for step in range(MAX_BUDGET):
                if performance[algorithm][dim][benchmark] <= step * dim:
                    ranking[algorithm][dim][step] += 1./len(benchmarks)
        plt.plot(ranking[algorithm][dim], label=f'{algorithm}, {dim}')

plt.xlabel('Steps / dimension')
plt.ylabel('Percentage')
plt.legend()
#plt.show()
plt.savefig("ranking.png", dpi=200)
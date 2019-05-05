"""Compare performance of different optimization algorithms."""

import numpy as np
import matplotlib.pyplot as plt
import functions
import optimizers
import inspect

algorithms = []
for name, obj in inspect.getmembers(optimizers):
    if inspect.isclass(obj):
        algorithms.append(name)

benchmarks = ['Rosenbrock', 'StyblinskiTang']
dimensions = {'Rosenbrock': 2, 'StyblinskiTang': 2}
parameters = {'StupidGradientDescent': [0.0001], 'MomentumGradientDescent': [0.0001, 0.1]}

MAX_BUDGET = 10000
EPSILON = 0.1
performance = dict()

# Get minimal steps for all benchmark / algorithm combinations
for benchmark in benchmarks:
    bm = getattr(functions, benchmark)(dim=dimensions[benchmark])
    performance[benchmark] = dict()
    for algorithm in algorithms:
        np.random.seed(seed=42)
        initstate = 10*(np.random.rand(dimensions[benchmark],  1)-1)[:, 0]
        alg = getattr(optimizers, algorithm)(*parameters[algorithm], initstate, bm.grad)
        for step in range(MAX_BUDGET * dimensions[benchmark]):
            alg.step()
            if np.max(np.linalg.norm(alg.state - bm.optimal()[0])) < EPSILON:
                performance[benchmark][algorithm] = (step, dimensions[benchmark])
                break
        # If no entry for algorithm is available, it didn't converge
        try:
            performance[benchmark][algorithm]
        except KeyError:
            performance[benchmark][algorithm] = (MAX_BUDGET * dimensions[benchmark] + 1, dimensions[benchmark])

print(performance)

# Create overall ranking
ranking = dict()
for algorithm in algorithms:
    ranking[algorithm] = np.zeros(MAX_BUDGET)
    for benchmark in benchmarks:
        for step in range(MAX_BUDGET):
            if performance[benchmark][algorithm][0] <= step * dimensions[benchmark]:
                ranking[algorithm][step] += 1./len(benchmarks)
    plt.plot(ranking[algorithm], label=algorithm)

plt.xlabel('Steps / dimension')
plt.ylabel('Percentage')
plt.legend()
plt.show()
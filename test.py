import numpy as np
from optimizers import StupidGradientDescent
from functions import Rosenbrock

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')

BUDGET = 500

if __name__=="__main__":
    # Initialize benchmark function object
    fr = Rosenbrock()
    # Use deterministic randomness
    np.random.seed(seed=42)
    # Initialize optimization algorithm object
    initstate = 10*(np.random.rand(2,  1)-1)[:,  0]
    gd = StupidGradientDescent(0.0001,  initstate, fr.grad)

    # Numpy array for logging
    performance = np.zeros((BUDGET, ))
    
    for i in range(BUDGET):
        # Take a gradient step with constant alpha
        gd.step()
        # Save f value
        performance[i] = fr.eval(gd.state[0], gd.state[1])

    plt.plot(np.log10(performance), 'bo')
    plt.savefig("rosenbrock.png", dpi=200)
    #plt.show()

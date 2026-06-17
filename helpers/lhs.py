from scipy.stats import qmc
import pandas as pd

def lhs(n, d, l_bounds, u_bounds):

    assert len(l_bounds) == d, "Number of bounds must match dimension"
    assert len(u_bounds) == d, "Number of bounds must match dimension"

    sampler = qmc.LatinHypercube(d)
    sample = sampler.random(n)

    scaled = qmc.scale(sample, l_bounds, u_bounds)
    return scaled

if __name__ == "__main__":
    n = 10
    d = 2

    l_bounds = [0, 0]
    u_bounds = [3, 4]

    sample = lhs(n, d, l_bounds, u_bounds)
    print(pd.DataFrame(sample))
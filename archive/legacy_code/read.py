import numpy as np

def read_shock_tube(path: str):
    data = np.load(path)
    return data

if __name__ == "__main__":
    data = read_shock_tube("./data/shock_tube.npz")
    print(data.files)

    print(data["U"][0][0])
    
    print(data["gamma"])
    print(data["p_low"])
    print(data["p_high"])
    print(data["rho_low"])
    print(data["rho_high"])
    print(data["x0"])
    print(data["dt"])
    print(data["n_steps"])
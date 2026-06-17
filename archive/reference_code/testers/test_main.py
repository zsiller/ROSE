import asyncio
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

from radical.asyncflow import WorkflowEngine
from radical.asyncflow.logging import init_default_logger
from rhapsody.backends import ConcurrentExecutionBackend

from rose.al.active_learner import SequentialActiveLearner

from simulations.CDR_1D.sim_wrapper import run_cdr
from surrogate.POD import POD
from surrogate.train import Surrogate

from active_learning.lhs import lhs


N_COMPONENTS = 5
QUERY_BATCH_SIZE = 10
N_SELECT = 1
L_BOUNDS = [0.0, 0.0]
U_BOUNDS = [2.0, 0.25]

class Manager:

    def __init__(self):
        self.X_pod = None
        self.X_gp = None

    def construct_training_data(self):
        betas = [.5, 1.0, 1.5, 2.0]

        X_pod_rows = []
        X_gp_rows = []

        for beta in betas:
            df, meta = run_cdr(theta=beta, t_final=0.25, write_every=1000)

            times = df.columns.to_numpy()

            for t in times:
                state_vec = df[t].to_numpy()
                X_pod_rows.append(state_vec.reshape(1, -1))
                X_gp_rows.append([beta, float(t)])

        self.X_pod = np.concatenate(X_pod_rows, axis=0)
        self.X_gp = np.asarray(X_gp_rows, dtype=float)

    def add_training_data(self, df, meta):
        times = df.columns.to_numpy()

        X_pod_rows = []
        X_gp_rows = []

        for t in times:
            state_vec = df[t].to_numpy()    
            X_pod_rows.append(state_vec.reshape(1, -1))
            X_gp_rows.append([meta.beta, float(t)])

        self.X_pod = np.vstack([self.X_pod, np.concatenate(X_pod_rows, axis=0)])
        self.X_gp = np.vstack([self.X_gp, np.asarray(X_gp_rows, dtype=float)])

def construct_training_data():

    betas = [.5, 1.0, 1.5, 2.0]

    X_pod_rows = []
    X_gp_rows = []

    for beta in betas:
        df, meta = run_cdr(theta=beta, t_final=0.25, write_every=1000)

        times = df.columns.to_numpy()
        
        for t in times:
            state_vec = df[t].to_numpy()
            X_pod_rows.append(state_vec.reshape(1, -1))
            X_gp_rows.append([beta, float(t)])

    X_pod = np.concatenate(X_pod_rows, axis=0)
    X_gp = np.asarray(X_gp_rows, dtype=float)
        
    return X_pod, X_gp

def max_uncertainty_index(std):

    return np.argsort(np.max(std, axis=1))[-N_SELECT:]

def random_query():

    beta = np.random.uniform(0.25, 2.0, size=QUERY_BATCH_SIZE)
    #t = np.random.uniform(0.0, 0.25, size=QUERY_BATCH_SIZE)

    X_test = np.column_stack([beta, np.full(QUERY_BATCH_SIZE, 0.25)])

    return X_test

def lhs_query():
    X_test = lhs(QUERY_BATCH_SIZE, 2, L_BOUNDS, U_BOUNDS)
    return X_test

def 

if __name__ == "__main__":

    manager = Manager()
    manager.construct_training_data()

    print(manager.X_pod.shape)
    print(manager.X_gp.shape)

    pod = POD(n_components=N_COMPONENTS)
    pod.fit(pd.DataFrame(manager.X_pod))

    coeffs = pod.svd.transform(pd.DataFrame(manager.X_pod))

    model = Surrogate()
    model.train(manager.X_gp, coeffs)

    print(model.evaluate(manager.X_gp, coeffs))

    X_test = lhs_query()

    coeffs_pred, std = model.predict(X_test, return_std=True)

    print(pd.DataFrame(std))

    max_uncertainty_indices = max_uncertainty_index(std)

    for index in max_uncertainty_indices:
        df_r, meta_r = run_cdr(theta=X_test[index, 0], t_final=X_test[index, 1], write_every=1000)
        manager.add_training_data(df_r, meta_r)

    print(manager.X_pod.shape)
    print(manager.X_gp.shape)

    pod.fit(pd.DataFrame(manager.X_pod))

    coeffs = pod.svd.transform(pd.DataFrame(manager.X_pod))

    model.train(manager.X_gp, coeffs)

    print(model.evaluate(manager.X_gp, coeffs))

    

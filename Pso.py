#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import numpy as np
from sko.tools import func_transformer
from .base import SkoBase
from .operators import crossover, mutation, ranking, selection
from .operators import mutation

import random
import time
from functools import wraps

# Set print options to display lists completely
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


def generate_own_X(lower_limit, upper_limit, load_set_standard, fluctuation_range, num_groups, num_dianzhan,
                   data_length):
    """Generate initial 3D array (number of groups, power stations, time series)

    Args:
        lower_limit: Lower bounds array
        upper_limit: Upper bounds array
        load_set_standard: Standard load dataset
        fluctuation_range: Allowed fluctuation range
        num_groups: Number of groups/populations
        num_dianzhan: Number of power stations
        data_length: Time series length (hours)
    """
    # Initialize 3D array (groups, stations, time series)
    data = np.zeros((num_groups, num_dianzhan * data_length), dtype=float)

    for m in range(num_groups):
        for n in range(num_dianzhan):
            # Get initial discharge flow bounds for current station
            low_0 = lower_limit[0 + n * 24]
            high_0 = upper_limit[0 + n * 24]

            # Generate random base discharge flow within bounds
            current_Q_max = round(random.uniform(low_0, high_0), 2)

            for i in range(0, data_length):
                # Calculate discharge flow for each timestep
                current_Q = round(load_set_standard[i] * current_Q_max, 2)

                # Ensure values stay within operational limits
                current_Q = min(upper_limit[i + n * 24],
                                max(lower_limit[i + n * 24], current_Q))

                data[m, i + n * 24] = current_Q
    return data


def timefn(fn):

    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result

    return measure_time


class PSO(SkoBase):
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint. Note: not available yet.
    constraint_ueq : tuple
        unequal constraint
    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration


    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    def __init__(self, func, n_dim=None, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.8, c1=0.5, c2=0.5,
                 constraint_eq=tuple(), constraint_ueq=tuple(), verbose=False
                 , dim=None, fr=10, v_num=0.25, dianzhan=5, load_set_standard=0):

        n_dim = n_dim or dim  # support the earlier version

        self.func = func_transformer(func)
        self.w = w  # inertia coefficient
        self.cp, self.cg = c1, c2  # cognitive and social coefficients
        self.pop = pop  # population size (number of particles)
        self.n_dim = n_dim  # dimension of search space
        self.max_iter = max_iter  # maximum iterations
        self.verbose = verbose  # whether to print progress
        self.fr = fr  # fluctuation range
        self.v_num = v_num  # velocity coefficient
        self.dianzhan = dianzhan  # number of power stations
        self.load_set_standard = load_set_standard  # load setting standard

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * pop)

        self.X = generate_own_X(lower_limit=self.lb, upper_limit=self.ub, load_set_standard=self.load_set_standard,
                                fluctuation_range=self.fr, num_groups=self.pop, num_dianzhan=self.dianzhan,
                                data_length=int(self.n_dim / self.dianzhan))

        v_high = (self.ub - self.lb) * self.v_num
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # velocity matrix
        self.Y = self.cal_y()  # calculate fitness values
        self.pbest_x = self.X.copy()  # personal best positions
        self.pbest_y = np.array([[np.inf]] * pop)  # personal best fitness values
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best position
        self.gbest_y = np.inf  # global best fitness value
        self.gbest_y_hist = []  # history of global best values
        self.update_gbest()

        # Recording configuration
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # legacy attributes

    def check_constraint(self, x):
        # Check inequality constraints
        for constraint_func in self.constraint_ueq:
            if (constraint_func(x) > 0).any():
                return False
        return True

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # Calculate fitness for all particles
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        Update personal best positions and values
        '''
        self.need_update = self.pbest_y > self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)
        print(np.round(self.pbest_x, 2), 'Historical personal best positions')
        print(self.pbest_y, 'Historical personal best values\n')

    def update_gbest(self):
        '''
        Update global best position and value
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]
        print('')
        print(np.round(self.gbest_x, 2), 'Global best position')
        print(self.gbest_y, 'Global best value\n')

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    @timefn
    def run(self, max_iter=None, precision=None, N=20):
        '''
        precision: None or float
            None -> run max_iter iterations
            float -> stop when fitness improvement < precision for N consecutive iterations
        N: convergence check window size
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0  # convergence counter
        for iter_num in range(self.max_iter):
            print('------------------------------------------- Iterating -------------------------------------------')
            print(f'Iteration {iter_num + 1}:')
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.verbose:
                print('')
                self.gbest_x = np.round(self.gbest_x, 2)
                print('Iter: {}, Best fit: {} at {}\n'.format(iter_num + 1, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    fit = run


class PSO_TSP(SkoBase):
    def __init__(self, func, n_dim, size_pop=50, max_iter=200, w=0.8, c1=0.1, c2=0.1):
        self.func = func_transformer(func)
        self.func_raw = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.max_iter = max_iter

        self.w = w
        self.cp = c1
        self.cg = c2

        self.X = self.crt_X()
        self.Y = self.cal_y()
        self.pbest_x = self.X.copy()
        self.pbest_y = np.array([[np.inf]] * self.size_pop)

        self.gbest_x = self.pbest_x[0, :]
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_gbest()
        self.update_pbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.verbose = False

    def crt_X(self):
        tmp = np.random.rand(self.size_pop, self.n_dim)
        return tmp.argsort(axis=1)

    def pso_add(self, c, x1, x2):
        x1, x2 = x1.tolist(), x2.tolist()
        ind1, ind2 = np.random.randint(0, self.n_dim - 1, 2)
        if ind1 >= ind2:
            ind1, ind2 = ind2, ind1 + 1

        part1 = x2[ind1:ind2]
        part2 = [i for i in x1 if i not in part1]  # this is very slow

        return np.array(part1 + part2)

    def update_X(self):
        for i in range(self.size_pop):
            x = self.X[i, :]
            x = self.pso_add(self.cp, x, self.pbest_x[i])
            self.X[i, :] = x

        self.cal_y()
        self.update_pbest()
        self.update_gbest()

        for i in range(self.size_pop):
            x = self.X[i, :]
            x = self.pso_add(self.cg, x, self.gbest_x)
            self.X[i, :] = x

        self.cal_y()
        self.update_pbest()
        self.update_gbest()

        for i in range(self.size_pop):
            x = self.X[i, :]
            new_x_strategy = np.random.randint(3)
            if new_x_strategy == 0:
                x = mutation.swap(x)
            elif new_x_strategy == 1:
                x = mutation.reverse(x)
            elif new_x_strategy == 2:
                x = mutation.transpose(x)

            self.X[i, :] = x

        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.Y

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            self.recorder()
            self.update_X()

            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

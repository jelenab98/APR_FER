import numpy as np


class F:
    @staticmethod
    def f(x):
        raise NotImplementedError

    def get_fitness(self, x):
        return np.abs(self.f(x))


class F1(F):
    @staticmethod
    def f(x):
        return np.add(100*np.power(x[:, 1] - np.power(x[:, 0], 2), 2), np.power(1 - x[:, 0], 2))


class F2(F):
    @staticmethod
    def f(x):
        res = np.zeros((x.shape[0],))
        ones = np.ones((x.shape[0],))
        for dim in range(x.shape[1]):
            x_tmp = x[:, dim]
            el_su = np.sum(np.power(x_tmp - (dim+1)*ones, 2), axis=0)
            res += el_su
        return res


class F6(F):
    @staticmethod
    def f(x):
        el_sum = np.sum(np.power(x, 2), axis=1)
        return np.add(0.5, np.multiply(np.subtract(np.power(np.sin(np.sqrt(el_sum)), 2), 0.5),
                                       np.reciprocal(np.power(np.add(1, np.multiply(0.001, el_sum)), 2))))


class F7(F):
    @staticmethod
    def f(x):
        el_sum = np.sum(np.power(x, 2), axis=1)
        return np.multiply(np.power(el_sum, 0.25),
                           np.add(1,
                                  np.power(np.sin(np.multiply(50, np.power(el_sum, 0.1))), 2)))


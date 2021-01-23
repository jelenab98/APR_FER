import math


class F1:
    def f(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


class F2:
    def f(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return (x1 - 4)**2 + 4 * (x2 - 2)**2


class F3:
    def __init__(self, i=None):
        self.i = i

    def f(self, x):
        y = 0
        if self.i is None:
            self.i = x.copy()
            for idx in range(x.n):
                self.i[idx, 0] = idx + 1
        for idx in range(x.n):
            x_i = x[idx, 0]
            i_i = self.i[idx, 0]
            y += (x_i - i_i)**2
        return y


class F4:
    def f(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return abs((x1 - x2) * (x1 + x2)) + math.sqrt(x1**2 + x2**2)


class F6:
    def f(self, x):
        square_sum = 0
        for i in range(x.n):
            square_sum += x[i, 0]**2
        return 0.5 + ((math.sin(math.sqrt(square_sum)))**2 - 0.5)/(1 + 0.001*square_sum)**2

from Matrica import *


class F1:
    def f(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    def dx1(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 2 * (200 * x1**3 - 200*x1*x2 + x1 - 1)

    def dx2(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 200 * (x2 - x1**2)

    def dx1dx1(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 1200 * (x1**2) - 400*x2 + 2

    def dx2dx2(self, x):
        return 200

    def dx1dx2(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return -400*x1

    def hessian(self, x):
        new_matrix = Matrix(n=2, m=2)
        new_matrix[0, 0] = self.dx1dx1(x)
        new_matrix[1, 1] = self.dx2dx2(x)
        new_matrix[0, 1] = self.dx1dx2(x)
        new_matrix[1, 0] = self.dx1dx2(x)
        return new_matrix

    def gradients(self, x):
        new_matrix = Matrix(n=2, m=1)
        new_matrix[0, 0] = self.dx1(x)
        new_matrix[1, 0] = self.dx2(x)
        return new_matrix


class F2:
    def f(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return (x1 - 4)**2 + 4 * (x2 - 2)**2

    def dx1(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 2*(x1-4)

    def dx2(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 8*(x2 - 2)

    def dx1dx1(self, x):
        return 2

    def dx2dx2(self, x):
        return 8

    def dx1dx2(self, x):
        return 0

    def gradients(self, x):
        new_matrix = Matrix(n=2, m=1)
        new_matrix[0, 0] = self.dx1(x)
        new_matrix[1, 0] = self.dx2(x)
        return new_matrix

    def hessian(self, x):
        new_matrix = Matrix(n=2, m=2)
        new_matrix[0, 0] = self.dx1dx1(x)
        new_matrix[1, 1] = self.dx2dx2(x)
        new_matrix[0, 1] = self.dx1dx2(x)
        new_matrix[1, 0] = self.dx1dx2(x)
        return new_matrix


class F3:
    def f(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return (x1 - 2)**2 + (x2 + 3)**2

    def dx1(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 2*(x1 - 2)

    def dx2(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 2*(x2 + 3)

    def dx1dx1(self, x):
        return 2

    def dx2dx2(self, x):
        return 2

    def dx1dx2(self, x):
        return 0

    def gradients(self, x):
        new_matrix = Matrix(n=2, m=1)
        new_matrix[0, 0] = self.dx1(x)
        new_matrix[1, 0] = self.dx2(x)
        return new_matrix

    def hessian(self, x):
        new_matrix = Matrix(n=2, m=2)
        new_matrix[0, 0] = self.dx1dx1(x)
        new_matrix[1, 1] = self.dx2dx2(x)
        new_matrix[0, 1] = self.dx1dx2(x)
        new_matrix[1, 0] = self.dx1dx2(x)
        return new_matrix


class F4:
    def f(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return (x1 - 3)**2 + x2**2

    def dx1(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 2*(x1 - 3)

    def dx2(self, x):
        x1, x2 = x[0, 0], x[1, 0]
        return 2*x2

    def dx1dx1(self, x):
        return 2

    def dx2dx2(self, x):
        return 2

    def dx1dx2(self, x):
        return 0

    def gradients(self, x):
        new_matrix = Matrix(n=2, m=1)
        new_matrix[0, 0] = self.dx1(x)
        new_matrix[1, 0] = self.dx2(x)
        return new_matrix

    def hessian(self, x):
        new_matrix = Matrix(n=2, m=2)
        new_matrix[0, 0] = self.dx1dx1(x)
        new_matrix[1, 1] = self.dx2dx2(x)
        new_matrix[0, 1] = self.dx1dx2(x)
        new_matrix[1, 0] = self.dx1dx2(x)
        return new_matrix

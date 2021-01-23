import copy
import math


class Matrix:
    def __init__(self, n=None, m=None, input_file=None, epsilon=1e-10):
        self.matrix = []
        self.epsilon = epsilon
        if n is None and m is None and input_file is not None:
            self.load(input_file)
        elif n is None and m is None:
            self.n = 0
            self.m = 0
        else:
            self.n = n
            self.m = m
            self.initialize()

    def __eq__(self, other):
        if self.n != other.n or self.m != other.m:
            return False
        for i in range(0, self.n):
            for j in range(0, self.m):
                if self[i, j] != other[i, j]:
                    return False
        return True

    def __str__(self):
        tmp_string = ''
        for i in range(self.n):
            for j in range(self.m - 1):
                tmp_el = self[i, j]
                if tmp_el == -0.0:
                    tmp_el = 0.0
                tmp_string += "{} ".format(tmp_el)
            tmp_el = self[i, self.m - 1]
            if tmp_el == -0.0:
                tmp_el = 0.0
            tmp_string += "{}\n".format(tmp_el)
        return tmp_string

    def __add__(self, other):
        if not self.check_dimensions(other):
            print("Dimenzije moraju biti iste!")
            return None
        new_matrix = Matrix(n=self.n, m=self.m, epsilon=self.epsilon)
        for i in range(self.n):
            for j in range(self.m):
                new_matrix[i, j] = self[i, j] + other[i, j]
        return new_matrix

    def __iadd__(self, other):
        if not self.check_dimensions(other):
            print("Dimenzije moraju biti iste!")
            return None
        for i in range(self.n):
            for j in range(self.m):
                self[i, j] += other[i, j]
        return self

    def __sub__(self, other):
        if not self.check_dimensions(other):
            print("Dimenzije moraju biti iste!")
            return None
        new_matrix = Matrix(n=self.n, m=self.m, epsilon=self.epsilon)
        for i in range(self.n):
            for j in range(self.m):
                new_matrix[i, j] = self[i, j] - other[i, j]
        return new_matrix

    def __isub__(self, other):
        if not self.check_dimensions(other):
            print("Dimenzije moraju biti iste!")
            return None
        for i in range(self.n):
            for j in range(self.m):
                self[i, j] -= other[i, j]
        return self

    def __mul__(self, other):
        if type(other) is Matrix:
            if not self.check_multiply_dimensions(other):
                print("Dimenzije se ne podudaraju za množenje!")
                return None
            new_matrix = Matrix(n=self.n, m=other.m, epsilon=self.epsilon)
            for i in range(self.n):
                for j in range(other.m):
                    for k in range(other.n):
                        new_matrix[i, j] += self[i, k] * other[k, j]
            return new_matrix

        elif type(other) in (float, int):
            new_matrix = Matrix(n=self.n, m=self.m, epsilon=self.epsilon)
            for i in range(self.n):
                for j in range(self.m):
                    new_matrix[i, j] = other * self[i, j]
            return new_matrix
        else:
            print("Krivi format za množenje!")
            return None

    def __rmul__(self, other):
        if type(other) is Matrix:
            if not self.check_multiply_dimensions(other):
                print("Dimenzije se ne podudaraju za množenje!")
                return None
            new_matrix = Matrix(n=self.n, m=other.m, epsilon=self.epsilon)
            for i in range(self.n):
                for j in range(other.m):
                    for k in range(other.n):
                        new_matrix[i, j] += self[i, k] * other[k, j]
            return new_matrix

        elif type(other) in (float, int):
            new_matrix = Matrix(n=self.n, m=self.m, epsilon=self.epsilon)
            for i in range(self.n):
                for j in range(self.m):
                    new_matrix[i, j] = other * self[i, j]
            return new_matrix
        else:
            print("Krivi format za množenje!")
            return None

    def __getitem__(self, item):
        try:
            i, j = item
            a = self.matrix[i][j]
            return a
        except:
            print("Krivo zadani indeksi za pristupanje!")
            print("Želi se pristupiti {}, a dimenzije su ({}, {})".format(item, self.n, self.m))
            return None

    def __setitem__(self, key, value):
        try:
            i, j = key
            self.matrix[i][j] = value
        except:
            print("Krivo zadani indeksi za pristupanje!")

    def __gt__(self, other):
        for i in range(self.n):
            for j in range(self.m):
                if self[i, j] <= other[i, j]:
                    return False
        return True

    def __ge__(self, other):
        for i in range(self.n):
            for j in range(self.m):
                if self[i, j] < other[i, j]:
                    return False
        return True

    def __lt__(self, other):
        for i in range(self.n):
            for j in range(self.m):
                if self[i, j] >= other[i, j]:
                    return False
        return True

    def __le__(self, other):
        for i in range(self.n):
            for j in range(self.m):
                if self[i, j] > other[i, j]:
                    return False
        return True

    def __abs__(self):
        new_matrix = Matrix(n=self.n, m=self.m, epsilon=self.epsilon)
        for i in range(self.n):
            for j in range(self.m):
                new_matrix[i, j] = abs(self[i, j])
        return new_matrix

    def initialize(self):
        self.matrix = [[0.0 for j in range(0, self.m)] for i in range(0, self.n)]

    def load(self, input_file):
        with open(input_file, 'r') as f:
            lines = f.readlines()
        self.m = len(lines[0].split(' '))
        self.n = len(lines)
        self.initialize()
        for i, line in enumerate(lines):
            elements = line.strip('\n').split(' ')
            for j, element in enumerate(elements):
                self[i, j] = float(element)
        return

    def save(self, output_file):
        with open(output_file, 'w') as f:
            for i in range(self.n):
                for j in range(self.m - 1):
                    tmp_el = self[i, j]
                    if tmp_el == -0.0:
                        tmp_el = 0.0
                    f.write("{} ".format(tmp_el))
                tmp_el = self[i, self.m - 1]
                if tmp_el == -0.0:
                    tmp_el = 0.0
                f.write("{}\n".format(tmp_el))
        return

    def copy(self):
        new_matrix = Matrix(n=self.n, m=self.m, epsilon=self.epsilon)
        new_matrix.matrix = copy.deepcopy(self.matrix)
        return new_matrix

    def set_values(self, matrix):
        self.matrix = matrix
        self.n = len(matrix)
        self.m = len(matrix[0])
        return self

    def set_value(self, value):
        self.initialize()
        for i in range(self.n):
            for j in range(self.m):
                self[i, j] += value
        return self

    def transpose(self):
        new_matrix = Matrix(n=self.m, m=self.n, epsilon=self.epsilon)
        for i in range(self.n):
            for j in range(self.m):
                new_matrix[j, i] = self[i, j]
        return new_matrix

    def identity(self):
        new_matrix = Matrix(n=self.m, m=self.n)
        for i in range(self.n):
            for j in range(self.m):
                if i == j:
                    new_matrix[i, j] = 1.0
        return new_matrix

    def check_dimensions(self, other):
        if self.n != other.n or self.m != other.m:
            return False
        return True

    def check_multiply_dimensions(self, other):
        if self.m != other.n:
            return False
        return True

    def switch_rows(self, index1, index2):
        if index1 >= self.n or index2 >= self.n:
            print("Krivo zadani indeksi za zamjenu!")
            return
        tmp = self.matrix[index1]
        self.matrix[index1] = self.matrix[index2]
        self.matrix[index2] = tmp
        return

    def set_epsilon(self, epsilon):
        self.epsilon = float(epsilon)

    def change_dimensions(self, n, m):
        old_n = self.n
        old_m = self.m
        old_matrix = copy.deepcopy(self.matrix)

        self.n = n
        self.m = m
        self.initialize()

        n_min = min(old_n, self.n)
        m_min = min(old_m, self.m)

        for i in range(0, n_min):
            for j in range(0, m_min):
                self[i, j] = old_matrix[i][j]

        return self

    def LU_decompose(self):
        if self.m != self.n:
            print("Nije kvadratna! Nema dekompozicije!")
            self.a = None
            return

        self.a = self.copy()

        for i in range(0, self.n):
            for j in range(i+1, self.m):
                if math.fabs(self.a[i, i]) < self.epsilon:
                    print("Matrica je singularna! Prekidam dekompoziciju")
                    self.a = None
                    return
                self.a[j, i] = self.a[j, i] / self.a[i, i]
                for k in range(i+1, self.n):
                    self.a[j, k] = self.a[j, k] - self.a[j, i] * self.a[i, k]
        return

    def LUP_decompose(self):
        if self.m != self.n:
            print("Nije kvadratna! Nema dekompozicije!")
            self.a = None
            self.p = None
            return

        self.a = self.copy()
        self.p = self.identity()
        self.s = 0

        for i in range(0, self.n):
            tmp_max = -math.inf
            r = i
            for j in range(i, self.n):
                if math.fabs(self.a[j, i]) > tmp_max:
                    tmp_max = math.fabs(self.a[j, i])
                    r = j
            if tmp_max < self.epsilon:
                print("Matrica je singularna! Prekidam dekompoziciju")
                self.a = None
                self.p = None
                return
            else:
                if r != i:
                    self.a.switch_rows(r, i)
                    self.p.switch_rows(r, i)
                    self.s += 1
            for j in range(i+1, self.m):
                if math.fabs(self.a[i, i]) < self.epsilon:
                    print("Matrica je singularna! Prekidam dekompoziciju")
                    self.a = None
                    self.p = None
                    return
                self.a[j, i] = self.a[j, i] / self.a[i, i]
                for k in range(i+1, self.n):
                    self.a[j, k] = self.a[j, k] - self.a[j, i] * self.a[i, k]
        return

    def backward_substitution(self, y):
        new_matrix = y.copy()
        for i in range(self.n-1, -1, -1):
            if math.fabs(self.a[i, i]) > self.epsilon:
                new_matrix[i, 0] = new_matrix[i, 0] / self.a[i, i]
            else:
                print("Ne mogu pronaci rjesenje! Prekidam rjesavanje sustava")
                return None
            for j in range(0, i):
                new_matrix[j, 0] -= self.a[j, i] * new_matrix[i, 0]

        return new_matrix

    def forward_substitution(self, b):
        new_matrix = b.copy()
        for i in range(0, self.n):
            for j in range(i+1, self.m):
                new_matrix[j, 0] -= self.a[j, i] * new_matrix[i, 0]
        return new_matrix

    def solve(self, b, lup=True):
        if lup:
            self.LUP_decompose()
            if self.a is not None:
                tmp_b = self.p * b
                tmp_b = tmp_b.extract_column(0)
                y = self.forward_substitution(tmp_b)
                x = self.backward_substitution(y)
                return x
            else:
                print("Nema rjesenja!")
                return None
        else:
            self.LU_decompose()
            if self.a is not None:
                y = self.forward_substitution(b)
                x = self.backward_substitution(y)
                return x

    def extract_column(self, column):
        new_matrix = Matrix(self.n, 1)
        for j in range(0, self.m):
            new_matrix[j, 0] = self[j, column]
        return new_matrix

    def concat_column(self, column_index, column):
        for j in range(0, self.m):
            self[j, column_index] = column[j, 0]

    def inverse(self):
        self.LUP_decompose()
        if self.a is not None:
            new_matrix = Matrix(self.n, self.m)
            for i in range(0, self.n):
                e_i = self.p.extract_column(i)
                y_i = self.forward_substitution(e_i)
                x_i = self.backward_substitution(y_i)
                new_matrix.concat_column(i, x_i)
            return new_matrix
        else:
            print("Ne mogu izracunati inverz!")
            return None

    def det(self):
        self.LUP_decompose()
        if self.a is not None:
            det_p = (-1) ** self.s
            det_u = 1
            for i in range(self.n):
                det_u *= self.a[i, i]
            return det_p*det_u

        else:
            print("LUP dekompozicija je neuspješna! Matrica je singularna!")
            return 0

    def norm(self):
        result = 0
        for i in range(0, self.n):
            for j in range(0, self.m):
                result += self[i, j]**2
        return result**0.5

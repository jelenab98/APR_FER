import math

from Matrica import Matrix


class GoldenRatio:
    def __init__(self, f, e=1e-6, a=None, b=None, h=1, x0=None):
        self.k = 0.5 * (5 ** 0.5 - 1)
        self.function = f
        self.function_calls = 0
        if a is None and b is None:
            self.a, self.b = UnimodalInterval(x0, f, h).calculate()
            self.x0 = x0
        else:
            self.a = a
            self.b = b
        self.e = Matrix(self.a.n, self.a.m).set_value(e)
        self.h = Matrix(self.a.n, self.a.m).set_value(h)

    def calculate(self):
        c = self.b - self.k * (self.b - self.a)
        d = self.a + self.k * (self.b - self.a)
        fc = self.function(c)
        fd = self.function(d)
        self.function_calls += 2
        while (self.b - self.a) > self.e:
            if fc < fd:
                self.b = d
                d = c
                c = self.b - self.k * (self.b - self.a)
                fd = fc
                fc = self.function(c)
            else:
                self.a = c
                c = d
                d = self.a + self.k * (self.b - self.a)
                fc = fd
                fd = self.function(d)
            self.function_calls += 1
        return self.a, self.b


class UnimodalInterval:
    def __init__(self, x0, f, h=1):
        self.x0 = x0
        self.function = f
        self.h_raw = h
        self.h = Matrix(x0.n, x0.m).set_value(h)
        self.l = self.x0 - self.h
        self.r = self.x0 + self.h
        self.function_calls = 0

    def calculate(self):
        m = self.x0.copy()

        fm = self.function(self.x0)
        fl = self.function(self.l)
        fr = self.function(self.r)

        self.function_calls += 3
        exp = 1

        if fm < fr and fm < fl:
            return self.l, self.r
        elif fm > fr:
            while fm > fr:
                self.l = m
                m = self.r
                fm = fr
                exp *= 2
                self.r = self.x0 + self.h * exp
                fr = self.function(self.r)
                self.function_calls += 1
        else:
            while fm > fl:
                self.r = m
                m = self.l
                fm = fl
                exp *= 2
                self.l = self.x0 - self.h * exp
                fl = self.function(self.l)
                self.function_calls += 1
        return self.l, self.r


class CoordinateSearch:
    def __init__(self, x0, f, e=1e-6, h=1):
        self.x0 = x0
        self.F = f
        self.e = Matrix(x0.n, x0.m).set_value(e)
        self.h = Matrix(x0.n, x0.m).set_value(h)
        self.h_raw = h
        self.e_raw = e
        self.dimension = x0.n
        self.function_calls = 0

    def calculate(self):
        x = self.x0.copy()
        m = 0
        while True:
            xs = x.copy()
            for i in range(self.dimension):
                e_i = Matrix(self.dimension, 1)
                e_i[i, 0] = 1
                lam = self.find_minimum(x, e_i, i)
                x[i, 0] += lam[0, 0]
            if abs(x - xs) < self.e:
                break
            m += 1
        return x

    def find_minimum(self, x, e_i, i):
        gr = GoldenRatio(lambda y: self.F.f(x + y[0, 0]*e_i), e=self.e_raw, h=self.h_raw,
                         x0=Matrix(1, 1).set_values([[x[i, 0]]]))
        l, r = gr.calculate()
        self.function_calls += gr.function_calls
        return (l + r) * 0.5


class Simplex:
    def __init__(self, x0, f, a=1, b=0.5, g=2, s=0.5, e=1e-6, h=1, print_states=False, print_short=False):
        self.x0 = x0
        self.F = f
        self.a = a
        self.b = b
        self.g = g
        self.s = s
        self.e = e
        self.h = h
        self.printing = print_states
        self.printing1 = print_short
        self.xh = None
        self.xl = None
        self.idx_h = None
        self.idx_l = None
        self.xc = Matrix(self.x0.n, self.x0.m)
        self.xe = Matrix(self.x0.n, self.x0.m)
        self.xr = Matrix(self.x0.n, self.x0.m)
        self.xk = Matrix(self.x0.n, self.x0.m)
        self.dimension = self.x0.n
        self.function_calls = 0
        self.points = [self.x0.copy()]
        for i in range(self.dimension):
            e_i = self.x0.copy()
            e_i[i, 0] += h
            self.points.append(e_i)

    def calculate(self):
        i = 0
        while True:
            flag = False
            f_xh, f_xl = self.find_h_l()
            self.centroid()
            f_xc = self.F.f(self.xc)
            if self.printing1:
                print("F(xc) = {}, za xc >>\n{}".format(f_xc, self.xc))
            self.reflection()
            f_xr = self.F.f(self.xr)
            self.function_calls += 2
            if self.printing:
                print("\n\nIteracija:{}, Početni simpleks je >>".format(i), end='')
                for p in self.points:
                    print("{} ".format(p[0, 0]), end='')
                print('')
                print("\txh: {}, f(xh):{}, xl:{}, f(xl):{}, xc: {}, f(xc): {}, xr:{}, f(xr):{}".format(
                    self.xh[0,0], f_xh, self.xl[0,0], f_xl, self.xc[0,0], f_xc, self.xr[0,0], f_xr))
            if f_xr < f_xl:
                self.expansion()
                if self.printing:
                    print("\tIzračunata ekspanzija, xe:{}, f(xe):{}".format(self.xe[0,0], self.F.f(self.xe)))
                if self.F.f(self.xe) < f_xl:
                    self.points[self.idx_h] = self.xe.copy()
                    self.xh = self.xe.copy()
                else:
                    self.points[self.idx_h] = self.xr.copy()
                    self.xh = self.xr.copy()
                self.function_calls += 1
            else:
                for point in self.points:
                    if point == self.xh:
                        continue
                    self.function_calls += 1
                    if f_xr <= self.F.f(point):
                        flag = True
                        break
                if flag:
                    self.points[self.idx_h] = self.xr.copy()
                    self.xh = self.xr.copy()
                    continue
                else:
                    if f_xr < f_xh:
                        self.points[self.idx_h] = self.xr.copy()
                        self.xh = self.xr.copy()
                    self.contraction()
                    if self.printing:
                        print("\tIzračunata kontrakcija, xk:{}, f(xk):{}".format(self.xk[0,0], self.F.f(self.xk)))
                    if self.F.f(self.xk) < f_xh:
                        self.points[self.idx_h] = self.xk.copy()
                        self.xh = self.xk.copy()
                    else:
                        self.move_points()
                        if self.printing:
                            print("\tTočke su pomaknute, novi simpleks je >>", end='')
                            for p in self.points:
                                print("\t{} ".format(p[0,0]), end='')
                            print('')
                    self.function_calls += 1
            if self.printing:
                print("\tSimpleks na kraju iteracije je >>", end='')
                for p in self.points:
                    print("\t{} ".format(p[0, 0]), end='')
                print('')
            if self.stop_condition(f_xc) <= self.e:
                break
            i += 1
        return self.xc

    def stop_condition(self, f_xc):
        result = 0
        for point in self.points:
            result += (self.F.f(point) - f_xc)**2
            self.function_calls += 1
        result /= len(self.points)
        a = math.sqrt(result)
        return a

    def find_h_l(self):
        max_value = -math.inf
        min_value = math.inf
        f_xh, f_xl = 0, 0
        for idx, point in enumerate(self.points):
            value = self.F.f(point)
            self.function_calls += 1
            if value > max_value:
                max_value = value
                self.xh = point.copy()
                self.idx_h = idx
                f_xh = value
            if value < min_value:
                min_value = value
                self.xl = point.copy()
                self.idx_l = idx
                f_xl = value
        return f_xh, f_xl

    def centroid(self):
        self.xc = Matrix(self.x0.n, self.x0.m)
        for point in self.points:
            if point == self.xh:
                continue
            self.xc += point
        self.xc *= 1/(len(self.points) - 1)

    def reflection(self):
        self.xr = (1 + self.a)*self.xc - self.a*self.xh

    def expansion(self):
        self.xe = (1 - self.g)*self.xc + self.g*self.xr

    def contraction(self):
        self.xk = (1 - self.b)*self.xc + self.b*self.xh

    def move_points(self):
        for i in range(len(self.points)):
            self.points[i] = self.s * (self.points[i] + self.xl)


class HookeJeeves:
    def __init__(self, x0, f, dx=0.5, e=1e-6, printing=False):
        self.x0 = x0
        self.F = f
        self.dx = dx
        self.e = e
        self.printing = printing
        self.function_calls = 0
        self.dimension = x0.n
        self.xp = x0.copy()
        self.xb = x0.copy()

    def calculate(self):
        f_xb = self.F.f(self.xb)
        while self.dx > self.e:
            xn = self.find()
            f_xn = self.F.f(xn)
            self.function_calls += 2
            if self.printing:
                print("Ispis trenutnih točaka")
                print("xb >>\n{}".format(self.xb))
                print("xp >>\n{}".format(self.xp))
                print("xn >>\n{}".format(xn))
                print("f(xn) = {}, f(xb) = {}\n".format(f_xn, f_xb))
            if f_xn < f_xb:
                self.xp = 2 * xn - self.xb
                self.xb = xn.copy()
                f_xb = f_xn
            else:
                self.dx /= 2
                self.xp = self.xb.copy()
        return self.xb

    def find(self):
        x = self.xp.copy()
        for i in range(self.dimension):
            p = self.F.f(x)
            x[i, 0] += self.dx
            n = self.F.f(x)
            self.function_calls += 2
            if n > p:
                x[i, 0] -= 2 * self.dx
                n = self.F.f(x)
                self.function_calls += 1
                if n > p:
                    x[i, 0] += self.dx
        return x

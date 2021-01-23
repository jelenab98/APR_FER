import math
import random
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
            f_xc = self.F(self.xc)
            if self.printing1:
                print("F(xc) = {}, za xc >>\n{}".format(f_xc, self.xc))
            self.reflection()
            f_xr = self.F(self.xr)
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
                if self.F(self.xe) < f_xl:
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
                    if f_xr <= self.F(point):
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
                        print("\tIzračunata kontrakcija, xk:{}, f(xk):{}".format(self.xk[0,0], self.F(self.xk)))
                    if self.F(self.xk) < f_xh:
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
            result += (self.F(point) - f_xc)**2
            self.function_calls += 1
        result /= len(self.points)
        a = math.sqrt(result)
        return a

    def find_h_l(self):
        max_value = -math.inf
        min_value = math.inf
        f_xh, f_xl = 0, 0
        for idx, point in enumerate(self.points):
            value = self.F(point)
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
        f_xb = self.F(self.xb)
        while self.dx > self.e:
            xn = self.find()
            f_xn = self.F(xn)
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
            p = self.F(x)
            x[i, 0] += self.dx
            n = self.F(x)
            self.function_calls += 2
            if n > p:
                x[i, 0] -= 2 * self.dx
                n = self.F(x)
                self.function_calls += 1
                if n > p:
                    x[i, 0] += self.dx
        return x


class GradientDescent:
    def __init__(self, x0, f, e=1e-6, golden_cut=False, p=False):
        self.x0 = x0
        self.F = f
        self.e = Matrix(x0.n, x0.m).set_value(e)
        self.golden_cut = golden_cut
        self.e_raw = e
        self.dimension = x0.n
        self.function_calls = 0
        self.gradient_calls = 0
        self.hessian_calls = 0
        self.p = p

    def calculate(self):
        x = self.x0.copy()
        min_f = None
        i = 0
        while i < 150:

            grads = self.F.gradients(x)
            self.gradient_calls += 1
            grads_norm = grads.norm()

            if grads_norm < self.e_raw:
                break

            coeff = -1*grads
            l_min = 1

            if self.golden_cut:
                coeff *= 1/grads_norm
                l_min = self.find_minimum(x, coeff)

            x = x + l_min*coeff
            f_x = self.F.f(x)
            self.function_calls += 1
            if min_f is None:
                min_f = f_x
            elif f_x < min_f:
                min_f = f_x
                i = 0
            else:
                i += 1
        return x

    def find_minimum(self, x, coeff):
        gr = GoldenRatio(lambda y: self.F.f(x + y[0, 0]*coeff), x0=Matrix(1, 1).set_value(1))
        l, r = gr.calculate()
        self.function_calls += gr.function_calls
        return ((l + r) * 0.5)[0, 0]


class NewtonRaphson:
    def __init__(self, x0, f, e=1e-6, golden_cut=False):
        self.x0 = x0
        self.F = f
        self.e = Matrix(x0.n, x0.m).set_value(e)
        self.golden_cut = golden_cut
        self.e_raw = e
        self.dimension = x0.n
        self.function_calls = 0
        self.gradient_calls = 0
        self.hessian_calls = 0

    def calculate(self):
        x = self.x0.copy()
        min_f = None
        i = 0
        while True:

            grads = self.F.gradients(x)
            hess = self.F.hessian(x)
            hess_inv = hess.inverse()
            self.gradient_calls += 1
            self.hessian_calls += 1
            coeff = -1*(hess_inv * grads)
            l_min = 1

            if self.golden_cut:
                coeff *= 1/coeff.norm()
                l_min = self.find_minimum(x, coeff)

            smjer = l_min*coeff

            if smjer.norm() < self.e_raw:
                return x

            x = x + l_min*coeff
            i += 1

            f_x = self.F.f(x)
            self.function_calls += 1

            if min_f is None:
                min_f = f_x
            elif f_x < min_f:
                min_f = f_x
                i = 0
            else:
                i += 1
            if i >= 150:
                return x

    def find_minimum(self, x, coeff):
        gr = GoldenRatio(lambda y: self.F.f(x + y[0, 0]*coeff), x0=Matrix(1, 1).set_value(1))
        l, r = gr.calculate()
        self.function_calls += gr.function_calls
        return ((l + r) * 0.5)[0, 0]


class Box:
    def __init__(self, x0, f, a=1.3, e=1e-6, g=None, xd=None, xg=None):
        self.x0 = x0
        self.F = f
        self.a = a
        self.e = e
        self.g = g
        self.xd = xd
        self.xg = xg
        self.idx_h = None
        self.idx_h2 = None
        self.xc = Matrix(self.x0.n, self.x0.m)
        self.xr = Matrix(self.x0.n, self.x0.m)
        self.dimension = self.x0.n
        self.function_calls = 0
        self.points = [self.x0.copy()]

    def check_explicit_conditions(self, x):
        for idx in range(self.dimension):
            if not(self.xd[idx, 0] <= x[idx, 0] <= self.xg[idx, 0]):
                return False
        return True

    def check_implicit_conditions(self, x):
        for g_i in self.g:
            if g_i(x) < 0:
                return False
        return True

    def find_h(self):
        max_value = -math.inf
        f_xh2 = 0
        for idx, point in enumerate(self.points):
            value = self.F.f(point)
            self.function_calls += 1
            if value > max_value:
                f_xh2 = max_value
                max_value = value
                self.idx_h2 = self.idx_h
                self.idx_h = idx
        return f_xh2

    def calculate_points(self):
        for t in range(0, 2*self.dimension):
            new_point = Matrix(self.dimension, 1)

            for idx in range(0, self.dimension):
                new_point[idx, 0] = self.xd[idx, 0] + random.random()*(self.xg[idx, 0] - self.xd[idx, 0])

            while self.check_implicit_conditions(new_point) is False:
                new_point = self.move_to_centroid(new_point)

            self.points.append(new_point)
            self.tmp_centroid()
        return

    def tmp_centroid(self):
        self.xc = Matrix(self.x0.n, self.x0.m)
        for idx in range(len(self.points)):
            self.xc += self.points[idx]
        self.xc *= 1/(len(self.points))

    def centroid(self):
        self.xc = Matrix(self.x0.n, self.x0.m)
        for idx in range(len(self.points)):
            if idx == self.idx_h:
                continue
            self.xc += self.points[idx]
        self.xc *= 1/(len(self.points) - 1)

    def reflection(self):
        self.xr = (1 + self.a)*self.xc - self.a*self.points[self.idx_h]

    def move_to_centroid(self, point):
        return 0.5*(self.xc + point)

    def stopping_condition(self):
        for idx in range(self.dimension):
            if abs(self.xc[idx, 0] - self.points[self.idx_h][idx, 0]) < self.e:
                return True
        return False

    def calculate(self):
        if not(self.check_explicit_conditions(self.x0)) or not(self.check_implicit_conditions(self.x0)):
            print("pocetna nisu zadovoljena")
            return None

        self.xc = self.x0.copy()
        self.calculate_points()

        min_f = None
        i = 0

        while i < 150:
            f_h = self.find_h()
            self.centroid()
            self.reflection()

            for idx in range(0, self.dimension):
                if self.xr[idx, 0] < self.xd[idx, 0]:
                    self.xr[idx, 0] = self.xd[idx, 0]
                elif self.xr[idx, 0] > self.xg[idx, 0]:
                    self.xr[idx, 0] = self.xg[idx, 0]

            while self.check_implicit_conditions(self.xr) is False:
                self.xr = self.move_to_centroid(self.xr)

            if self.F.f(self.xr) >= f_h:
                self.xr = self.move_to_centroid(self.xr)

            self.points[self.idx_h] = self.xr.copy()
            if self.stopping_condition():
                break
            f_xc = self.F.f(self.xc)
            if min_f is None:
                min_f = f_xc
            elif f_xc < min_f:
                min_f = f_xc
                i = 0
            else:
                i += 1

        return self.xc


class MixedTransformation:
    def __init__(self, x0, f, e=1e-6, t=1, g=None, h=None):
        self.x0 = x0
        self.F = f
        self.e = e
        self.t = t
        self.gs = g
        self.hs = h
        self.ts = [1 for i in range(len(g))]
        self.dimension = self.x0.n
        self.function_calls = 0

    def f(self, x):
        g_result = 0
        h_result = 0
        for t, g in zip(self.ts, self.gs):
            res = g(x)
            if res <= 0:
                tmp_b = -1e10
            else:
                tmp_b = math.log(res)
            g_result -= t*(1/self.t)*tmp_b
        if self.hs is not None:
            for h in self.hs:
                h_result += self.t * (h(x))**2
        return self.F.f(x) + g_result + h_result

    def calculate(self):
        x_0 = self.x0.copy()
        if self.check_inner_point():
            self.inner_point()
            simp = Simplex(x_0, self.f)
            x_0 = simp.calculate()
            self.function_calls += simp.function_calls
        i = 0
        self.ts = [1 for i in range(len(self.gs))]
        while i < 10000:
            simp = Simplex(x_0, self.f)
            x_min = simp.calculate()
            self.function_calls += simp.function_calls
            if self.stopping_condition(x_0, x_min):
                break
            x_0 = x_min
            self.t *= 10
            i += 1
        return x_0

    def stopping_condition(self, x_0, x_min):
        for idx in range(self.dimension):
            a = abs(x_0[idx, 0] - x_min[idx, 0])
            if a < self.e:
                return True
        return False

    def check_inner_point(self):
        for g in self.gs:
            if g(self.x0) < 0:
                return True
        return False

    def inner_point(self):
        self.ts = []
        for g in self.gs:
            if g(self.x0) < 0:
                self.ts.append(-1)
            else:
                self.ts.append(0)



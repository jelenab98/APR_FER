from Matrica import Matrix
import matplotlib.pyplot as plt


class IntegrationSolver:
    def __init__(self, a, b, t_max, t, r, printing=False, printing_step=100):
        self.a = Matrix().load(a)
        self.b = Matrix().load(b)
        self.r_params = False
        self.check_params(r)
        if self.r_params is False:
            self.r = Matrix().load(r)
        else:
            self.r = Matrix(self.a.n, 1)
        self.t_max = t_max
        self.t = t
        self.n = self.a.n
        self.u = Matrix(self.n, self.n).identity()
        self.printing = printing
        self.printing_step = printing_step
        self.steps = []
        self.P, self.Q = None, None

    def check_params(self, r):
        with open(r, 'r') as f:
            lines = f.readlines()
        if lines[0].strip('\n') == "t":
            self.r_params = True

    def solve(self, x0):
        x_current = Matrix().load(x0)
        self.steps.append(x_current)
        t = self.t
        i = 0
        while t <= self.t_max:
            if self.r_params:
                self.change_r(t)
            x_next = self.P * x_current + self.Q * self.r
            x_current = x_next.copy()
            self.steps.append(x_next)
            if self.printing and i % self.printing_step == 0:
                print("Iter: {}, x: {}".format(i, x_current))
            t += self.t
            i += 1
        return

    def change_r(self, t):
        raise NotImplementedError

    def f(self, x):
        return self.a * x + self.b * self.r

    def save_calculations(self, output_file):
        with open(output_file, 'w') as f:
            for step in self.steps:
                f.write("{}\n".format(step))
        return

    def calculate_error(self, gt_list):
        err = Matrix(self.n, 1)
        for pred, gt in zip(self.steps, gt_list):
            err += abs(pred - gt)
        return err

    def plot_steps(self, title="Vizulaizacija"):
        figure = plt.figure(figsize=(15, 10))
        figure.suptitle(title, fontsize=16)
        t = []
        t_i = 0
        while t_i <= self.t_max:
            t.append(t_i)
            t_i += self.t
        for i in range(self.n):
            plt.subplot(self.n, 1, i + 1)
            tmp_step = []
            for step in self.steps:
                tmp_step.append(step[i, 0])
            plt.scatter(t, tmp_step)
            plt.title("Izračunata vrijednost varijable x{}".format(i + 1))
        plt.show()


class Euler(IntegrationSolver):
    def __init__(self, a, b, t_max, t, r, printing=False, printing_step=100):
        super().__init__(a, b, t_max, t, r, printing, printing_step)
        self.P = self.u + self.a * self.t
        self.Q = self.t * self.b

    def change_r(self, t):
        self.r = self.r.set_value(t)

    def step_predict(self, x_current, t):
        if self.r_params:
            self.change_r(t-self.t)
        return x_current + self.t*self.f(x_current)


class ReversedEuler(IntegrationSolver):
    def __init__(self, a, b, t_max, t, r, printing=False, printing_step=100):
        super().__init__(a, b, t_max, t, r, printing, printing_step)
        self.P = (self.u - self.a * self.t).inverse()
        self.Q = self.P * self.t * self.b

    def change_r(self, t):
        self.r = self.r.set_value(t + self.t)

    def step_predict(self, x_current, t):
        if self.r_params:
            self.change_r(t)
        return self.P * x_current + self.Q * self.r

    def step_correct(self, x_current, x_aproximated, t):
        if self.r_params:
            self.r = self.r.set_value(t-self.t)
        return x_current + self.t*self.f(x_aproximated)


class Trapez(IntegrationSolver):
    def __init__(self, a, b, t_max, t, r, printing=False, printing_step=100, correction=False):
        super().__init__(a, b, t_max, t, r, printing, printing_step)
        self.P = (self.u - self.a * self.t * 0.5).inverse() * (self.u + self.a * self.t * 0.5)
        self.Q = (self.u - self.a * self.t * 0.5).inverse() * self.t * 0.5 * self.b
        if self.r_params is False and correction is False:
            self.r = 2 * self.r

    def change_r(self, t):
        self.r = Matrix(self.n, 1).set_value(t) + Matrix(self.n, 1).set_value(t + self.t)

    def step_predict(self, x_current, t):
        if self.r_params:
            self.change_r(t)
        return self.P * x_current + self.Q * self.r

    def step_correct(self, x_current, x_aproximated, t):
        if self.r_params:
            self.r = self.r.set_value(t - self.t)
            a = self.f(x_current)
            self.r = self.r.set_value(t)
            b = self.f(x_aproximated)
            return x_current + 0.5*self.t*(a + b)
        else:
            return x_current + 0.5*self.t*(self.f(x_current) + self.f(x_aproximated))


class RungeKutta(IntegrationSolver):
    def __init__(self, a, b, t_max, t, r, printing=False, printing_step=100):
        super().__init__(a, b, t_max, t, r, printing, printing_step)

    def solve(self, x0):
        x_current = Matrix().load(x0)
        self.steps.append(x_current)
        t = self.t
        i = 0
        r1, r2, r3, r4 = self.r, self.r, self.r, self.r
        while t <= self.t_max:
            if self.r_params:
                r1 = self.r.set_value(t)
                r2 = self.r.set_value(t + 0.5 * self.t)
                r3 = self.r.set_value(t + 0.5 * self.t)
                r4 = self.r.set_value(t + self.t)
            m1 = self.a * x_current + self.b * r1
            m2 = self.a * (x_current + 0.5 * self.t * m1) + self.b * r2
            m3 = self.a * (x_current + 0.5 * self.t * m2) + self.b * r3
            m4 = self.a * (x_current + self.t * m3) + self.b * r4
            x_next = x_current + (self.t / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
            self.steps.append(x_next)
            x_current = x_next.copy()
            if self.printing and i % self.printing_step == 0:
                print("Iter: {}, x: {}".format(i, x_current))
            i += 1
            t += self.t

        return

    def change_r(self, t):
        return


class PredictCorrect(IntegrationSolver):
    def __init__(self, a, b, t_max, t, r, printing=False, printing_step=100, corrector_iters=1,
                 predictor="Euler", corrector="InverseEuler"):
        super().__init__(a, b, t_max, t, r, printing, printing_step)

        self.c_iter = corrector_iters

        if predictor == "Euler":
            self.predictor = Euler(a, b, t_max, t, r, printing, printing_step)
        elif predictor == "InverseEuler":
            self.predictor = ReversedEuler(a, b, t_max, t, r, printing, printing_step)
        else:
            self.predictor = Trapez(a, b, t_max, t, r, printing, printing_step)

        if corrector == "InverseEuler":
            self.corrector = ReversedEuler(a, b, t_max, t, r, printing, printing_step)
        else:
            self.corrector = Trapez(a, b, t_max, t, r, printing, printing_step, True)

    def change_r(self, t):
        self.r = self.r.set_value(t)

    def solve(self, x0):
        x_current = Matrix().load(x0)
        self.steps.append(x_current)
        t = self.t
        i = 0
        while t <= self.t_max:
            x_predicted = self.predictor.step_predict(x_current, t)
            x_next = self.corrector.step_correct(x_current, x_predicted, t)
            if self.c_iter > 1:
                for j in range(self.c_iter-1):
                    x_next = self.corrector.step_correct(x_current, x_next, t)
            x_current = x_next.copy()
            self.steps.append(x_next)
            if self.printing and i % self.printing_step == 0:
                print("Iter: {}, p: {:.4f}, {:.4f}, c: {:.4f}, {:.4f}".format(i+1, x_predicted[0,0], x_predicted[1,0], x_current[0,0], x_current[1,0]))
            t += self.t
            i += 1
        return

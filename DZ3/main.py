from Functions import *
from Optimization import *


def prvi_zadatak():
    x = Matrix().set_values([[0], [0]])
    f = F3()
    alg = GradientDescent(x, f, golden_cut=False, p=True)
    x_min1 = alg.calculate()
    n_1 = alg.function_calls

    alg2 = GradientDescent(x, f, golden_cut=True, p=True)
    x_min2 = alg2.calculate()
    n_2 = alg2.function_calls

    print("{:^23} | {:^23} | {:^20}".format("Funkcija: 3", "minimum: (2, -3)", "x0: (0, 0)"))
    print("{:^23} | {:^23} | {:^20} | {:^20}".format("algoritam", "minimum", "broj poziva funkcije",
                                                     "broj poziva grads"))
    print("-"*95)
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^20} | {:^20}".format("Gradijentni, GR=False", x_min1[0, 0], x_min1[1, 0],
                                                                    n_1, alg.gradient_calls))
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^20} | {:^20}".format("Gradijentni, GR=True", x_min2[0, 0], x_min2[1, 0],
                                                                    n_2, alg2.gradient_calls))


def drugi_zadatak():
    f1 = F1()
    f2 = F2()
    x0_1 = Matrix().set_values([[-1.9], [2]])
    x0_2 = Matrix().set_values([[0.1], [0.3]])

    grad_1 = GradientDescent(x0_1, f1, golden_cut=True)
    grad_2 = GradientDescent(x0_2, f2, golden_cut=True)

    newton_1 = NewtonRaphson(x0_1, f1, golden_cut=True)
    newton_2 = NewtonRaphson(x0_2, f2, golden_cut=True)

    xg1 = grad_1.calculate()
    ng1 = grad_1.function_calls
    xg2 = grad_2.calculate()
    ng2 = grad_2.function_calls

    xn1 = newton_1.calculate()
    nn1 = newton_1.function_calls
    xn2 = newton_2.calculate()
    nn2 = newton_2.function_calls

    print("{:^23} | {:^23} | {:^20}".format("Funkcija: 1", "minimum: (1, 1)", "x0: (-1.9, 2)"))
    print("{:^23} | {:^23} | {:^20} | {:^20} | {:^20}".format("algoritam", "minimum", "broj poziva funkcije",
                                                              "broj poziva grads", "broj poziva Hessa"))
    print("-"*117)
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^20} | {:^20} | {:^20}".format("Gradijentni spust", xg1[0, 0], xg1[1, 0],
                                                                             ng1, grad_1.gradient_calls,
                                                                             grad_2.hessian_calls))
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^20} | {:^20} | {:^20}".format("Newton-Raphson", xn1[0, 0], xn1[1, 0],
                                                                             nn1, newton_1.gradient_calls,
                                                                             newton_1.hessian_calls))
    print()
    print("{:^23} | {:^23} | {:^20}".format("Funkcija: 2", "minimum: (4, 2)", "x0: (0.1, 0.3)"))
    print("{:^23} | {:^23} | {:^20} | {:^20} | {:^20}".format("algoritam", "minimum", "broj poziva funkcije",
                                                              "broj poziva grads", "broj poziva Hess"))
    print("-"*117)
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^20} | {:^20} | {:^20}".format("Gradijentni spust", xg2[0, 0], xg2[1, 0],
                                                                             ng2, grad_2.gradient_calls,
                                                                             grad_2.hessian_calls))
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^20} | {:^20} | {:^20}".format("Newton-Raphson", xn2[0, 0], xn2[1, 0],
                                                                             nn2, newton_2.gradient_calls,
                                                                             newton_2.hessian_calls))


def treci_zadatak():
    f1 = F1()
    f2 = F2()

    x0_1 = Matrix().set_values([[-1.9], [2]])
    x0_2 = Matrix().set_values([[0.1], [0.3]])

    gs = [lambda x: x[1, 0] - x[0, 0], lambda x: 2 - x[0, 0]]
    xd = Matrix(n=2, m=1).set_value(-100)
    xg = Matrix(2, 1).set_value(100)

    box1 = Box(x0_1, f1, g=gs, xd=xd, xg=xg)
    x1 = box1.calculate()
    n1 = box1.function_calls

    box2 = Box(x0_2, f2, g=gs, xd=xd, xg=xg)
    x2 = box2.calculate()
    n2 = box2.function_calls

    print("{:^23} | {:^23} | {:^20}".format("Funkcija: 1", "minimum: (1, 1)", "x0: (-1.9, 2)"))
    print("{:^23} | {:^23} | {:^23} | {}".format("algoritam", "minimum", "F(minimum)", "broj poziva funkcije"))
    print("-"*98)
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^23.6f}  | {:^20}".format("Box", x1[0, 0], x1[1, 0], f1.f(x1),
                                                                        n1))
    print()
    print("{:^23} | {:^23} | {:^20}".format("Funkcija: 2", "minimum: (4, 2)", "x0: (0.1, 0.3)"))
    print("{:^23} | {:^23} | {:^23} | {}".format("algoritam", "minimum", "F(minimum)", "broj poziva funkcije"))
    print("-"*98)
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^23.6f} | {:^20}".format("Box", x2[0, 0], x2[1, 0], f2.f(x2),
                                                                       n2))


def cetvrti_zadatak():
    f1 = F1()
    f2 = F2()

    x0_1 = Matrix().set_values([[-1.9], [2]])
    x0_2 = Matrix().set_values([[0.1], [0.3]])

    gs = [lambda x: x[1, 0] - x[0, 0], lambda x: 2 - x[0, 0]]

    box1 = MixedTransformation(x0_1, f1, g=gs)
    x1 = box1.calculate()
    n1 = box1.function_calls

    box2 = MixedTransformation(x0_2, f2, g=gs)
    x2 = box2.calculate()
    n2 = box2.function_calls

    print("{:^23} | {:^23} | {:^20}".format("Funkcija: 1", "minimum: (1, 1)", "x0: (-1.9, 2)"))
    print("{:^23} | {:^23} | {:^23} | {}".format("algoritam", "minimum", "F(minimum)", "broj poziva funkcije"))
    print("-"*98)
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^23.6f}  | {:^20}".format("Transforms", x1[0, 0], x1[1, 0], box1.f(x1),
                                                                        n1))
    print()
    print("{:^23} | {:^23} | {:^20}".format("Funkcija: 2", "minimum: (4, 2)", "x0: (0.1, 0.3)"))
    print("{:^23} | {:^23} | {:^23} | {}".format("algoritam", "minimum", "F(minimum)", "broj poziva funkcije"))
    print("-"*98)
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^23.6f} | {:^20}".format("Transforms", x2[0, 0], x2[1, 0], box2.f(x2),
                                                                       n2))
    return


def peti_zadatak():
    f1 = F4()
    f2 = F4()

    x0_1 = Matrix().set_values([[0], [0]])
    x0_2 = Matrix().set_values([[5], [5]])

    gs = [lambda x: 3 - x[1, 0] - x[0, 0], lambda x: 3 + 1.5*x[0, 0] - x[1, 0]]
    hs = [lambda x: x[1, 0] - 1]

    box1 = MixedTransformation(x0_1, f1, g=gs, h=hs)
    x1 = box1.calculate()
    n1 = box1.function_calls

    box2 = MixedTransformation(x0_2, f2, g=gs, h=hs)
    x2 = box2.calculate()
    n2 = box2.function_calls

    print("{:^23} | {:^23} | {:^20}".format("Funkcija: 4", "minimum: (3, 0)", "x0: (0, 0)"))
    print("{:^23} | {:^23} | {:^23} | {}".format("algoritam", "minimum", "F(minimum)","broj poziva funkcije"))
    print("-"*98)
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^23.6f} | {:^20}".format("Transforms", x1[0, 0], x1[1, 0], box1.f(x1),
                                                                        n1))
    print()
    print("{:^23} | {:^23} | {:^20}".format("Funkcija: 4", "minimum: (3, 0)", "x0: (5, 5)"))
    print("{:^23} | {:^23} | {:^23} | {}".format("algoritam", "minimum", "F(minimum)", "broj poziva funkcije"))
    print("-"*98)
    print("{:^23} | ({:>10.6f},{:>10.6f}) | {:^23.6f} | {:^20}".format("Transforms", x2[0, 0], x2[1, 0], box2.f(x2),
                                                                       n2))
    return


def read_from_file(input_file):
    params = dict()
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        elements = line.strip('\n').split(' ')
        params[elements[0]] = elements[1]
    return params


if __name__ == '__main__':
    prvi_zadatak()
    print("\n{}\n".format("="*117))
    drugi_zadatak()
    print("\n{}\n".format("="*117))
    treci_zadatak()
    print("\n{}\n".format("=" * 117))
    cetvrti_zadatak()
    print("\n{}\n".format("=" * 117))
    peti_zadatak()


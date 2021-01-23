from Optimization import *
from Functions import *


def prvi_zadatak():
    p = Matrix().set_values([[3]])
    function = F3(p)
    x0 = Matrix().set_values([[10]])

    unimodal = UnimodalInterval(f=function.f, x0=x0)
    l_uni, r_uni = unimodal.calculate()
    n_uni = unimodal.function_calls

    golden_ratio = GoldenRatio(f=function.f, x0=x0)
    l_gr, r_gr = golden_ratio.calculate()
    n_gr = golden_ratio.function_calls

    simplex = Simplex(x0=x0, f=function)
    min_simp = simplex.calculate()
    n_simp = simplex.function_calls

    hj = HookeJeeves(x0=x0, f=function)
    min_hj = hj.calculate()
    n_hj = hj.function_calls

    coords = CoordinateSearch(x0=x0, f=function)
    min_coords = coords.calculate()
    n_coords = coords.function_calls

    print("Funkcija: 3 | minimum: 3  | x0: 10")
    print("{:^15} | {:^23} | {}".format("algoritam", "minimum", "broj poziva funkcije"))
    print("-"*64)
    print("{:^15} | [{:>10.6f},{:>10.6}] | {:^20}".format("Zlatni rez", l_gr[0, 0], r_gr[0, 0], n_gr))
    print("{:^15} | [{:>10.6f},{:>10.6}] | {:^20}".format("Unimodalni", l_uni[0, 0], r_uni[0, 0], n_uni))
    print("{:^15} | ({:^21.6f}) | {:^20}".format("Simplex", min_simp[0, 0], n_simp))
    print("{:^15} | ({:^21.6f}) | {:^20}".format("Hooke Jeeves", min_hj[0, 0], n_hj))
    print("{:^15} | ({:^21.6f}) | {:^20}".format("Coordinate", min_coords[0, 0], n_coords))

    x0 = Matrix().set_values([[25]])
    unimodal = UnimodalInterval(f=function.f, x0=x0)
    l_uni, r_uni = unimodal.calculate()
    n_uni = unimodal.function_calls

    golden_ratio = GoldenRatio(f=function.f, x0=x0)
    l_gr, r_gr = golden_ratio.calculate()
    n_gr = golden_ratio.function_calls

    simplex = Simplex(x0=x0, f=function)
    min_simp = simplex.calculate()
    n_simp = simplex.function_calls

    hj = HookeJeeves(x0=x0, f=function)
    min_hj = hj.calculate()
    n_hj = hj.function_calls

    coords = CoordinateSearch(x0=x0, f=function)
    min_coords = coords.calculate()
    n_coords = coords.function_calls

    print("\n\nFunkcija: 3 | minimum: (3)  | x0: (25)")
    print("{:^15} | {:^23} | {}".format("algoritam", "minimum", "broj poziva funkcije"))
    print('-'*64)
    print("{:^15} | [{:>10.6f},{:>10.6}] | {:^20}".format("Zlatni rez", l_gr[0, 0], r_gr[0, 0], n_gr))
    print("{:^15} | [{:>10.6f},{:>10.6}] | {:^20}".format("Unimodalni", l_uni[0, 0], r_uni[0, 0], n_uni))
    print("{:^15} | ({:^21.6f}) | {:^20}".format("Simplex", min_simp[0, 0], n_simp))
    print("{:^15} | ({:^21.6f}) | {:^20}".format("Hooke Jeeves", min_hj[0, 0], n_hj))
    print("{:^15} | ({:^21.6f}) | {:^20}".format("Coordinate", min_coords[0, 0], n_coords))

    x0 = Matrix().set_values([[-25]])

    unimodal = UnimodalInterval(f=function.f, x0=x0)
    l_uni, r_uni = unimodal.calculate()
    n_uni = unimodal.function_calls

    golden_ratio = GoldenRatio(f=function.f, x0=x0)
    l_gr, r_gr = golden_ratio.calculate()
    n_gr = golden_ratio.function_calls

    simplex = Simplex(x0=x0, f=function)
    min_simp = simplex.calculate()
    n_simp = simplex.function_calls

    hj = HookeJeeves(x0=x0, f=function)
    min_hj = hj.calculate()
    n_hj = hj.function_calls

    coords = CoordinateSearch(x0=x0, f=function)
    min_coords = coords.calculate()
    n_coords = coords.function_calls

    print("\n\nFunkcija: 3 | minimum: (3)  | x0: (-25)")
    print("{:^15} | {:^23} | {}".format("algoritam", "minimum", "broj poziva funkcije"))
    print("-"*64)
    print("{:^15} | [{:>10.6f},{:>10.6}] | {:^20}".format("Zlatni rez", l_gr[0, 0], r_gr[0, 0], n_gr))
    print("{:^15} | [{:>10.6f},{:>10.6}] | {:^20}".format("Unimodalni", l_uni[0, 0], r_uni[0, 0], n_uni))
    print("{:^15} | ({:^21.6f}) | {:^20}".format("Simplex", min_simp[0, 0], n_simp))
    print("{:^15} | ({:^21.6f}) | {:^20}".format("Hooke Jeeves", min_hj[0, 0], n_hj))
    print("{:^15} | ({:^21.6f}) | {:^20}".format("Coordinate", min_coords[0, 0], n_coords))
    return


def drugi_zadatak():
    x0 = Matrix().set_values([[-1.9], [2]])
    fn = F1()

    simplex = Simplex(x0=x0, f=fn)
    min_simp_1 = simplex.calculate()
    n_simp_1 = simplex.function_calls

    hj = HookeJeeves(x0=x0, f=fn)
    min_hj_1 = hj.calculate()
    n_hj_1 = hj.function_calls

    coords = CoordinateSearch(x0=x0, f=fn)
    min_coords_1 = coords.calculate()
    n_coords_1 = coords.function_calls

    print("Funkcija 1 | minimum: (1,1) | x0: (-1.9, 2)\n")
    print("{:^15} | {:^23} | {}".format("algoritam", "minimum", "broj poziva funkcije"))
    print("-"*64)
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Simplex", min_simp_1[0, 0], min_simp_1[1, 0], n_simp_1))
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Hooke Jeeves", min_hj_1[0, 0], min_hj_1[1, 0], n_hj_1))
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Coordinate", min_coords_1[0, 0], min_coords_1[1, 0],
                                                        n_coords_1))

    x0 = Matrix().set_values([[0.1], [0.3]])
    fn = F2()

    simplex = Simplex(x0=x0, f=fn)
    min_simp_2 = simplex.calculate()
    n_simp_2 = simplex.function_calls

    hj = HookeJeeves(x0=x0, f=fn)
    min_hj_2 = hj.calculate()
    n_hj_2 = hj.function_calls

    coords = CoordinateSearch(x0=x0, f=fn)
    min_coords_2 = coords.calculate()
    n_coords_2 = coords.function_calls

    print("\n\nFunkcija 2 | minimum: (4,2) | x0: (0.1, 0.3)\n")
    print("{:^15} | {:^23} | {}".format("algoritam", "minimum", "broj poziva funkcije"))
    print("-"*64)
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Simplex", min_simp_2[0, 0], min_simp_2[1, 0], n_simp_2))
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Hooke Jeeves", min_hj_2[0, 0], min_hj_2[1, 0], n_hj_2))
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Coordinate", min_coords_2[0, 0], min_coords_2[1, 0],
                                                        n_coords_2))

    x0 = Matrix(5, 1)
    p = Matrix().set_values([[1], [2], [3], [4], [5]])
    fn = F3(p)

    simplex = Simplex(x0=x0, f=fn)
    min_simp_3 = simplex.calculate()
    n_simp_3 = simplex.function_calls

    hj = HookeJeeves(x0=x0, f=fn)
    min_hj_3 = hj.calculate()
    n_hj_3 = hj.function_calls

    coords = CoordinateSearch(x0=x0, f=fn)
    min_coords_3 = coords.calculate()
    n_coords_3 = coords.function_calls

    print("\n\nFunkcija 3 | minimum: (1,2,3,4,5) | x0: (0, 0, 0, 0, 0)\n")
    print("{:^15} | {:^56} | {}".format("algoritam", "minimum", "broj poziva funkcije"))
    print("-"*97)
    print("{:^15} | ({:10.6f},{:10.6f},{:10.6f},{:10.6f},{:10.6f}) | {:^20}".format("Simplex", min_simp_3[0, 0],
                                                                                    min_simp_3[1, 0], min_simp_3[2, 0],
                                                                                    min_simp_3[3, 0], min_simp_3[4, 0],
                                                                                    n_simp_3))
    print("{:^15} | ({:10.6f},{:10.6f},{:10.6f},{:10.6f},{:10.6f}) | {:^20}".format("Hooke Jeeves", min_hj_3[0, 0],
                                                                                    min_hj_3[1, 0], min_hj_3[2, 0],
                                                                                    min_hj_3[3, 0], min_hj_3[4, 0],
                                                                                    n_hj_3))
    print("{:^15} | ({:10.6f},{:10.6f},{:10.6f},{:10.6f},{:10.6f}) | {:^20}".format("Coordinates", min_coords_3[0, 0],
                                                                                    min_coords_3[1, 0],
                                                                                    min_coords_3[2, 0],
                                                                                    min_coords_3[3, 0],
                                                                                    min_coords_3[4, 0], n_coords_3))

    x0 = Matrix().set_values([[5.1], [1.1]])
    fn = F4()

    simplex = Simplex(x0=x0, f=fn)
    min_simp_4 = simplex.calculate()
    n_simp_4 = simplex.function_calls

    hj = HookeJeeves(x0=x0, f=fn)
    min_hj_4 = hj.calculate()
    n_hj_4 = hj.function_calls

    coords = CoordinateSearch(x0=x0, f=fn)
    min_coords_4 = coords.calculate()
    n_coords_4 = coords.function_calls

    print("\n\nFunkcija 4 | minimum: (0,0) | x0: (5.1, 1.1)\n")
    print("{:^15} | {:^23} | {}".format("algoritam", "minimum", "broj poziva funkcije"))
    print("-"*64)
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Simplex", min_simp_4[0, 0], min_simp_4[1, 0], n_simp_4))
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Hooke Jeeves", min_hj_4[0, 0], min_hj_4[1, 0], n_hj_4))
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Coordinates", min_coords_4[0, 0], min_coords_4[1, 0],
                                                         n_coords_4))
    return


def treci_zadatak():
    x0 = Matrix().set_values([[5], [5]])
    fn = F4()

    simplex = Simplex(x0, fn)
    min_simp = simplex.calculate()
    n_simp = simplex.function_calls

    hj = HookeJeeves(x0, fn)
    min_hj = hj.calculate()
    n_hj = hj.function_calls

    print("Funkcija 4 | minimum: (0,0) | x0: (5, 5)\n")
    print("{:^15} | {:^23} | {}".format("algoritam", "minimum", "broj poziva funkcije"))
    print("-"*64)
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Simplex", min_simp[0, 0], min_simp[1, 0], n_simp))
    print("{:^15} | ({:10.6f},{:10.6f}) | {:^20}".format("Hooke Jeeves", min_hj[0, 0], min_hj[1, 0], n_hj))
    return


def cetvrti_zadatak():
    x0 = Matrix().set_values([[0.5], [0.5]])
    fn = F1()
    h_s = [1, 3, 7, 12, 18]
    print("Funkcija 1 | minimum: (1,1) | x0: (0.5, 0.5)")
    print("{:^5} | {:^23} | {}".format("h", "minimum", "broj poziva funkcije"))
    print("-"*54)
    for h in h_s:
        simpleks = Simplex(x0, fn, h=h)
        min_simp = simpleks.calculate()
        n_simp = simpleks.function_calls
        print("{:^5} | ({:10.6f},{:10.6f}) | {:^20}".format(h, min_simp[0, 0], min_simp[1, 0], n_simp))

    x0 = Matrix().set_values([[20], [20]])
    print("\nFunkcija 1 | minimum: (1,1) | x0: (20, 20)")
    print("{:^5} | {:^23} | {}".format("h", "minimum", "broj poziva funkcije"))
    print("-"*54)
    for h in h_s:
        simpleks = Simplex(x0, fn, h=h)
        min_simp = simpleks.calculate()
        n_simp = simpleks.function_calls
        print("{:^5} | ({:10.6f},{:10.6f}) | {:^20}".format(h, min_simp[0, 0], min_simp[1, 0], n_simp))
    return


def peti_zadatak():
    x0_s = [(-14, 0), (3, 49), (-37, 15), (34, -49)]
    print("Funkcija 6 | minimum: (0,0)")
    print("{:^9} | {:^23} | {:^15} | {}".format("x0", "minimum", "f(minimum)", "broj poziva funkcije"))
    print("-"*75)
    tocnost = 0
    for (x1, x2) in x0_s:
        x0 = Matrix().set_values([[x1], [x2]])
        fn = F6()
        simp = Simplex(x0, fn, e=1e-4)
        min_hj = simp.calculate()
        n_hj = simp.function_calls
        f = fn.f(min_hj)
        tocnost += 1 if f < 1e-4 else 0
        print("({:>3},{:>3}) | ({:10.6f},{:10.6f}) | {:^15.6f} | {:^20}".format(x1, x2, min_hj[0, 0], min_hj[1, 0],
                                                                             f, n_hj))
    tocnost /= len(x0_s)
    print("-"*75)
    print("Točnost pronalaska minimuma: {:.4f}".format(tocnost))

    return


def read_from_file(input_file):
    params = dict()
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        elements = line.strip('\n').split(' ')
        params[elements[0]] = elements[1]
    return params


def citanje_golden_ratio(input_file):
    params = read_from_file(input_file)
    f = None
    a, b, x0 = None, None, None
    if "f" in params:
        if params["f"] == "1":
            f = F1()
        elif params["f"] == "2":
            f = F2()
        elif params["f"] == "3":
            if "p" in params:
                elements = params["p"].split(',')
                n = len(elements)
                p = Matrix(n, 1)
                for idx, element in enumerate(elements):
                    p[idx, 0] = float(element)
                f = F3(p)
            else:
                f = F3()
        elif params["f"] == "4":
            f = F4()
        elif params["f"] == "6":
            f = F6()
        else:
            print("Kriva funkcija!")
            return

    if "x0" in params:
        elements = params['x0'].split(' ')
        n = len(elements)
        x0 = Matrix(n, 1)
        for idx, element in enumerate(elements):
            x0[idx, 0] = float(element)
    elif 'a' in params and 'b' in params:
        elements = params['a'].split(' ')
        n = len(elements)
        a = Matrix(n, 1)
        for idx, element in enumerate(elements):
            a[idx, 0] = float(element)
        elements = params['b'].split(' ')
        n = len(elements)
        b = Matrix(n, 1)
        for idx, element in enumerate(elements):
            b[idx, 0] = float(element)
    else:
        print("Početna točka ili početni interval moraju biti zadani!")
        return

    e = float(params.get('e', 1e-6))
    h = float(params.get('h', 1))

    golden_ratio = GoldenRatio(f.f, a=a, b=b, x0=x0, e=e, h=h)
    l, r = golden_ratio.calculate()
    print(l, r)


def citanje_unimodal(input_file):
    params = read_from_file(input_file)
    f = None
    x0 = None
    if "f" in params:
        if params["f"] == "1":
            f = F1()
        elif params["f"] == "2":
            f = F2()
        elif params["f"] == "3":
            if "p" in params:
                elements = params["p"].split(',')
                n = len(elements)
                p = Matrix(n, 1)
                for idx, element in enumerate(elements):
                    p[idx, 0] = float(element)
                f = F3(p)
            else:
                f = F3()
        elif params["f"] == "4":
            f = F4()
        elif params["f"] == "6":
            f = F6()
        else:
            print("Kriva funkcija!")
            return

    if "x0" in params:
        elements = params['x0'].split(' ')
        n = len(elements)
        x0 = Matrix(n, 1)
        for idx, element in enumerate(elements):
            x0[idx, 0] = float(element)
    else:
        print("Greška! x0 mora biti zadan!")
        return
    h = float(params.get('h', 1))

    uni = UnimodalInterval(x0, f.f, h=h)
    l, r = uni.calculate()
    print(l, r)


def citanje_simpleks(input_file):
    params = read_from_file(input_file)
    f = None
    x0 = None
    if "f" in params:
        if params["f"] == "1":
            f = F1()
        elif params["f"] == "2":
            f = F2()
        elif params["f"] == "3":
            if "p" in params:
                elements = params["p"].split(',')
                n = len(elements)
                p = Matrix(n, 1)
                for idx, element in enumerate(elements):
                    p[idx, 0] = float(element)
                f = F3(p)
            else:
                f = F3()
        elif params["f"] == "4":
            f = F4()
        elif params["f"] == "6":
            f = F6()
        else:
            print("Kriva funkcija!")
            return

    if "x0" in params:
        elements = params['x0'].split(' ')
        n = len(elements)
        x0 = Matrix(n, 1)
        for idx, element in enumerate(elements):
            x0[idx, 0] = float(element)
    else:
        print("Početna točka mora biti zadana!")
        return

    e = float(params.get('e', 1e-6))
    h = float(params.get('h', 1))
    a = float(params.get('a', 1))
    b = float(params.get('b', 0.5))
    g = float(params.get('g', 2))
    s = float(params.get('s', 0.5))

    simp = Simplex(x0, f, a, b, g, s, e, h)
    x_min = simp.calculate()
    print(x_min)


def citanje_hj(input_file):
    params = read_from_file(input_file)
    f = None
    x0 = None
    if "f" in params:
        if params["f"] == "1":
            f = F1()
        elif params["f"] == "2":
            f = F2()
        elif params["f"] == "3":
            if "p" in params:
                elements = params["p"].split(',')
                n = len(elements)
                p = Matrix(n, 1)
                for idx, element in enumerate(elements):
                    p[idx, 0] = float(element)
                f = F3(p)
            else:
                f = F3()
        elif params["f"] == "4":
            f = F4()
        elif params["f"] == "6":
            f = F6()
        else:
            print("Kriva funkcija!")
            return

    if "x0" in params:
        elements = params['x0'].split(' ')
        n = len(elements)
        x0 = Matrix(n, 1)
        for idx, element in enumerate(elements):
            x0[idx, 0] = float(element)
    else:
        print("Početna točka mora biti zadana!")
        return
    e = float(params.get('e', 1e-6))
    dx = float(params.get('dx', 0.5))
    hj = HookeJeeves(x0, f, dx, e)
    x_min = hj.calculate()
    print(x_min)


def citaj_koords(input_file):
    params = read_from_file(input_file)
    f = None
    x0 = None
    if "f" in params:
        if params["f"] == "1":
            f = F1()
        elif params["f"] == "2":
            f = F2()
        elif params["f"] == "3":
            if "p" in params:
                elements = params["p"].split(',')
                n = len(elements)
                p = Matrix(n, 1)
                for idx, element in enumerate(elements):
                    p[idx, 0] = float(element)
                f = F3(p)
            else:
                f = F3()
        elif params["f"] == "4":
            f = F4()
        elif params["f"] == "6":
            f = F6()
        else:
            print("Kriva funkcija!")
            return

    if "x0" in params:
        elements = params['x0'].split(' ')
        n = len(elements)
        x0 = Matrix(n, 1)
        for idx, element in enumerate(elements):
            x0[idx, 0] = float(element)
    else:
        print("Početna točka mora biti zadana!")
        return
    e = float(params.get('e', 1e-6))
    h = float(params.get('h', 1))
    cs = CoordinateSearch(x0, f, e, h)
    x_min = cs.calculate()
    print(x_min)


def ispisi_zadatke():
    print()
    print("="*64)
    print("{:^64}".format("1.zadatak"))
    print("=" * 64)
    print()
    prvi_zadatak()
    print()
    print("="*64)
    print("{:^64}".format("2.zadatak"))
    print("=" * 64)
    print()
    drugi_zadatak()
    print()
    print("="*64)
    print("{:^64}".format("3.zadatak"))
    print("=" * 64)
    print()
    treci_zadatak()
    print()
    print("="*54)
    print("{:^54}".format("4.zadatak"))
    print("=" * 54)
    print()
    cetvrti_zadatak()
    print()
    print("="*75)
    print("{:^58}".format("5.zadatak"))
    print("=" * 75)
    print()
    peti_zadatak()


if __name__ == '__main__':
    ispisi_zadatke()

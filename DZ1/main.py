from Matrica import *


def zadatak_sustav(broj_zadatka, input_file1, input_file2):
    print("{}. zadatak:".format(broj_zadatka))
    a = Matrix(input_file=input_file1)
    b = Matrix(input_file=input_file2)
    print("A = ")
    print(a)
    print("b = ")
    print(b)
    print("Rjesavanje s LU dekompozicijom daje: ")
    print(a.solve(b, False))
    print("\nRjesavanje s LUP dekompozicijom daje: ")
    print(a.solve(b, True))


def zadatak_sustav_epsilon(broj_zadatka, input_file1, input_file2, epsilon):
    print("{}. zadatak:".format(broj_zadatka))
    a = Matrix(input_file=input_file1, epsilon=epsilon)
    b = Matrix(input_file=input_file2)
    print("A = ")
    print(a)
    print("b = ")
    print(b)
    print("Rjesavanje s LU dekompozicijom daje: ")
    print(a.solve(b, False))
    print("\nRjesavanje s LUP dekompozicijom daje: ")
    print(a.solve(b, True))


def zadatak_proizvoljni_sustav(broj_zadatka, input_file):
    print("{}. zadatak:".format(broj_zadatka))
    a_3 = Matrix(input_file=input_file)
    b_3 = Matrix(n=3, m=1).set_values([[2], [3], [5]])
    print("A = ")
    print(a_3)
    print("Stvoren je proizvoljni slobodni vektor b =")
    print(b_3)
    print("Rjesavanje s LU dekompozicijom daje: ")
    print(a_3.solve(b_3, False))
    print("\nRjesavanje s LUP dekompozicijom daje: ")
    print(a_3.solve(b_3, True))


def zadatak_inverz(broj_zadatka, input_file):
    print("{}. zadatak:".format(broj_zadatka))
    a = Matrix(input_file=input_file)
    print("A = ")
    print(a)
    print("Racunanje inverza: ")
    print(a.inverse())


def zadatak_determinanta(broj_zadatka, input_file):
    print("{}. zadatak:".format(broj_zadatka))
    a = Matrix(input_file=input_file)
    print("A = ")
    print(a)
    print("Racunanje determinante: ")
    print(a.det())


def prikazi_zadatke():
    zadatak_sustav(2, '2a', '2b')
    print('\n' + "-" * 40)
    print("-" * 40)

    zadatak_proizvoljni_sustav(3, '3')
    print('\n' + "-" * 40)
    print("-" * 40)

    zadatak_sustav(4, '4a', '4b')
    print('\n' + "-" * 40)
    print("-" * 40)

    zadatak_sustav(5, '5a', '5b')
    print('\n' + "-" * 40)
    print("-" * 40)
    zadatak_sustav_epsilon(6, '6a', '6b', 1e-6)
    print('\n' + "-" * 40)
    print("-" * 40)
    zadatak_inverz(7, '7')
    print('\n' + "-" * 40)
    print("-" * 40)
    zadatak_inverz(8, '8')
    print('\n' + "-" * 40)
    print("-" * 40)
    zadatak_determinanta(9, '9')
    print('\n' + "-" * 40)
    print("-" * 40)
    zadatak_determinanta(10, '10')


if __name__ == '__main__':

    a = Matrix().set_values([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = Matrix(input_file='2a')
    print("Učitane matrice:")
    print("A >> ")
    print(a)
    print("B >> ")
    print(b)
    c = a.transpose()
    print("C = A.T >> ")
    print(c)
    c += a * 0.5 * b * (a - 2 * b)
    print("C = A * 0.5 * B * (A - 2 * B) >> ")
    print(c)
    print("C[0, 0] = ", c[0, 0])
    """
    d = Matrix().set_values([[0.1, 0.2, 3.3]])
    d2 = Matrix().set_values([[0.1, 0.2, 3.3]])
    d = d * 5
    d = d * (1/5)
    print()
    print(d)
    print(d2)
    print("\nMatrice su iste >>", d == d2)
    """
    print("\nIspisujem rezultate zadataka >>\n")
    prikazi_zadatke()


##
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss


from typing import Union, List, Tuple

def rectangular_rule(func, a, b, n):
    """
    Metoda prostokątów do przybliżonego rozwiązania całki oznaczonej.

    :param func: Funkcja, której całkę oznaczoną chcemy przybliżyć.
    :param a: Dolna granica całkowania.
    :param b: Górna granica całkowania.
    :param n: Liczba podprzedziałów (większa liczba n daje dokładniejsze przybliżenie).
    :return: Przybliżona wartość całki oznaczonej.
    """
    if isinstance(a, Union[int, float]) and isinstance(b, Union[int, float]) and b > a:
        step = (b - a)/n
        result = 0
        begin = a
        while begin < b:
            result += func(begin)
            begin += step
        return result*step
    return None


def trapezoidal_rule(func, a, b, n):
    """
    Metoda trapezów do przybliżonego rozwiązania całki oznaczonej.

    :param func: Funkcja, której całkę oznaczoną chcemy przybliżyć.
    :param a: Dolna granica całkowania.
    :param b: Górna granica całkowania.
    :param n: Liczba podprzedziałów (większa liczba n daje dokładniejsze przybliżenie).
    :return: Przybliżona wartość całki oznaczonej.
    """
    if isinstance(a, Union[int, float]) and isinstance(b, Union[int, float]) and b > a:
        step = (b - a)/n
        result = 0
        begin = a
        while begin < b:
            result += (func(begin) + func(begin+step))*step/2
            begin += step
        return result
    return None




def custom_integration(func, a, b, order):
    """
    Własna funkcja całkująca, wykorzystująca kwadraturę Gaussa-Legendre'a.

    :param func: Funkcja do zintegrowania.
    :param a: Dolna granica całkowania.
    :param b: Górna granica całkowania.
    :param order: Rząd kwadratury.
    :return: Przybliżona wartość całki.
    """
    # Przeskalowanie przedziału do (a, b)
    # Obliczenie wartości funkcji w przeskalowanych punktach
    # Obliczenie całki przy użyciu kwadratury Gaussa-Legendre'a
    if isinstance(a, Union[int, float]) and isinstance(b, Union[int, float]):
        x, y = leggauss(order)
        scaledX = 0.5 * (b - a) * x + 0.5 * (b + a)
        scaledY = 0.5 * (b - a) * y

        return np.sum(scaledY * func(scaledX))
    return None



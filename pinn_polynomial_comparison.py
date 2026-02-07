import numpy as np
import matplotlib.pyplot as plt

# Define the polynomial activation functions

def genocchi(x):
    return (x**2 - x) / 2

def chebyshev(x):
    return np.cos(np.arccos(x))


def hermite(x):
    return np.exp(-x**2) * np.polynomial.hermite.hermval(x, [1])


def legendre(x):
    return np.polynomial.legendre.legval(x, [1])

# Comparison function

def compare_polynomials(x):
    plt.figure(figsize=(12, 8))
    plt.plot(x, genocchi(x), label='Genocchi')
    plt.plot(x, chebyshev(x), label='Chebyshev')
    plt.plot(x, hermite(x), label='Hermite')
    plt.plot(x, legendre(x), label='Legendre')

    plt.title('Comparison of Polynomial Activation Functions')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()

# Sample input
x = np.linspace(-1, 1, 100)
compare_polynomials(x)
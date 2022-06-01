# Author: Bence Racz
from math import sqrt
from numpy import dot
from numpy.linalg import norm


matrix = []
number_type = []


# Reads the matrices from the given file
def read_file(path, test=0):
    global matrix
    global number_type
    file = open(path, 'r')
    lines = file.readlines()

    for line in lines:
        number = []
        line = line.split('\n')[0]
        line = line.split(',')
        for character in line:
            number.append(int(character))

        # if we are reading the test file
        if test == 1:
            number_type.append(number[len(number) - 1])
            number.pop()

        matrix.append(number)

    file.close()
    return matrix


# This function returns the result of the Kronecker-delta function
def kronecker_delta(x, y):
    if x == y:
        return 1
    else:
        return 0


# The function calculates the euclidean metric
def euclidean_metric(x, z):
    metric = 0
    for i in range(64):
        metric += sqrt((x[i] - z[i])**2)
    return metric


# The function calculates the cos similarity
def cos_similarity(x, z):
    return dot(x, z) / (norm(x) * norm(z))


# The function implements the k-nearest neighbour
def k_nearest_neighbour():
    return


# The function trains the program
def train():
    matrices = read_file('./optdigits.tra', 1)
    return


# Main function of the app
def main():
    train()
    return


if __name__ == '__main__':
    main()

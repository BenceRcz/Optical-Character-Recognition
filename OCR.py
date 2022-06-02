# Author: Bence Racz
from math import sqrt
import numpy
from numpy import dot
from numpy.linalg import norm
import heapq

matrix = []
k = 5
used_metric = 1
used_algorithm = 'gradiens'
used_file = 'test'


# Reads the matrices from the given file
def read_file(path):
    global matrix
    file = open(path, 'r')
    lines = file.readlines()

    for line in lines:
        number = []
        line = line.split('\n')[0]
        line = line.split(',')
        for character in line:
            number.append(int(character))

        matrix.append(number)

    file.close()
    return


# The function calculates the euclidean metric
def euclidean_metric(x, z):
    metric = 0
    for i in range(64):
        metric += sqrt((x[i] - z[i]) ** 2)
    return metric


# The function calculates the cos similarity
def cos_similarity(x, z):
    return dot(x, z) / (norm(x) * norm(z))


# The function returns the most frequent number in the calculated similarities
def get_most_frequent(similarities):
    most_frequent = numpy.zeros(10)
    maximum = 0
    got_number = 0

    if used_metric == 1:
        first_n = heapq.nsmallest(k, similarities)
    else:
        first_n = heapq.nlargest(k, similarities)

    for i in first_n:
        most_frequent[i[1]] += 1

        if most_frequent[i[1]] > maximum:
            maximum = most_frequent[i[1]]
            got_number = i[1]

    return got_number


# The function implements the k-nearest neighbour
def k_nearest_neighbour(test_vector):
    similarities = []
    for number in matrix:
        if used_metric == 1:
            calculated = euclidean_metric(test_vector, number)
        else:
            calculated = cos_similarity(test_vector, number)

        heapq.heappush(similarities, (calculated, number[len(number) - 1]))

    return get_most_frequent(similarities)


# The function creates the "prototypes" for the centroid algorithm
def create_centroid_data():
    c = []
    frequency = numpy.zeros(10)
    sum_vector = []

    for i in range(10):
        sum_vector.append(numpy.zeros(64))
        c.append(numpy.zeros(64))

    for vector in matrix:
        frequency[vector[64]] += 1

        # The last element of the vector is the number the vector represents
        sum_vector[vector[64]] += vector[:64]

    for i in range(10):
        c[i] = sum_vector[i] / frequency[i]

    return c


# The function implements the centroid algorithm
def centroid(test_vector, c):
    minimal = 99999
    maximal = 0
    got_number = 0
    for i in range(10):
        if used_metric == 1:
            current_number = euclidean_metric(test_vector[:64], c[i])
            if minimal > current_number:
                minimal = current_number
                got_number = i
        else:
            current_number = cos_similarity(test_vector[:64], c[i])
            if maximal < current_number:
                maximal = current_number
                got_number = i

    return got_number


# This function creates the data for the gradiens descent algorithm
def create_gradiens_data(a, b):
    data = []
    for vector in matrix:
        if vector[64] == a:
            vector[64] = 1
            data.append(vector)
        if vector[64] == b:
            vector[64] = -1
            data.append(vector)

    return data


# The function implements the gradiens descent algorithm
def gradiens(vectors, gamma, iterations):
    w = numpy.zeros(64)
    x = []
    y = []
    for vector in vectors:
        x.append(vector[:64])
        y.append(vector[64])

    for i in range(iterations):
        w = w - 2 * gamma / len(vectors) * numpy.matmul( numpy.transpose(x),  numpy.matmul(x, w) - y)
    return w


# The function evaluates the gradiens descent algorithms return value
def got_number_gradiens(vector, w):
    return numpy.sign(numpy.matmul(vector, w))


# This function tests and prints a statistic about the gradiens descent algorithm
def test_gradiens_descent():
    data = create_gradiens_data(4, 7)
    w = gradiens(data, 0.0000035, 2000)

    got_number = 0
    got_error = 0

    for line in data:
        if got_number_gradiens(line[:64], w) == line[64]:
            got_number += 1
        else:
            got_error += 1

    print('-----------Accuracy percentage using the gradiens descent: ',
          100 - ((got_error / (got_error + got_number)) * 100), '-----------', sep='')

    return


# The function trains the program
def test_train(version):
    c = []
    if version == 'centroid':
        c = create_centroid_data()
    got_number = numpy.zeros(10)
    got_error = numpy.zeros(10)

    for number in matrix:
        if version == 'knn':
            current_number = k_nearest_neighbour(number)
        else:
            current_number = centroid(number, c)

        actual_number = number[64]

        if current_number == actual_number:
            got_number[actual_number] += 1
        else:
            got_error[actual_number] += 1

    for i in range(10):
        print('-------------------------For the the number: ', i, '-------------------------')
        print('--    The program got it right: ', got_number[i])
        print('--    The program missed it: ', got_error[i])
        print('--    The error percentage: ', (got_error[i] / (got_number[i] + got_error[i])) * 100)

    print('--------------Overall accuracy percentage: ', 100 - ((sum(got_error) / (sum(got_error) + sum(got_number))) * 100),
          '--------------')

    return


# Main function of the app
def main():
    if used_file == 'train':
        read_file('./optdigits.tra')
    if used_file == 'test':
        read_file('./optdigits.tes')

    if used_algorithm == 'knn':
        test_train('knn')
    if used_algorithm == 'centroid':
        test_train('centroid')
    if used_algorithm == 'gradiens':
        test_gradiens_descent()
    return


if __name__ == '__main__':
    main()

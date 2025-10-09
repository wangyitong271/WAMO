#ifndef RF_H
#define RF_H

#include <vector>

std::vector<double> createVector(int size)
{
    std::vector<double> v(size, 0.0);
    return v;
}

void setVectorData(std::vector<double> &vector, int index, double value)
{
    vector[index] = value;
}


double predict_tree(int node, const double features[], const int children_left[], const int children_right[], const int feature[], const double threshold[], const double value[]) {
    while (children_left[node] != -1) {
        if (features[feature[node]] <= threshold[node])
            node = children_left[node];
        else
            node = children_right[node];
    }
    return value[node];
}

#endif // RANDOM_FOREST_H
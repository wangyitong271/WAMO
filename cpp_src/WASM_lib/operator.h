#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <cmath>

inline void leaky_relu(std::vector<float>& x, float alpha) {
    for (auto& xi : x) {
        xi = xi > 0 ? xi : alpha * xi;
    }
}

inline void tanh_activation(std::vector<float>& x) {
    for (auto& xi : x) {
        xi = std::tanh(xi);
    }
}

inline void sigmoid_activation(std::vector<float>& x) {
    for (auto& xi : x) {
        xi = 1 / (1 + std::exp(-xi));
    }
}

inline void softmax_activation(std::vector<float>& x) {
    float sum = 0;
    for (auto& xi : x) {
        sum += std::exp(xi);
    }
    for (auto& xi : x) {
        xi = std::exp(xi) / sum;
    }
}

inline void relu_activation(std::vector<float>& x) {
    for (auto& xi : x) {
        xi = xi > 0 ? xi : 0;
    }
}

inline void linear_activation(std::vector<float>& x) {
    // do nothing
}

inline void abs(std::vector<float>& x) {
    for (auto& xi : x) {
        xi = std::abs(xi);
    }
}

inline void add(std::vector<float>& x, std::vector<float>& y) {
    for (int i = 0; i < x.size(); i++) {
        x[i] += y[i];
    }
}
inline void addN(std::vector<float>& x, const std::vector<std::vector<float>>& vectors) {
    for (const auto& vec : vectors) {
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += vec[i];
        }
    }
}

inline bool all(const std::vector<float>& x) {
    for (const auto& xi : x) {
        if (xi == 0) return false;
    }
    return true;
}

inline bool any(const std::vector<float>& x) {
    for (const auto& xi : x) {
        if (xi != 0) return true;
    }
    return false;
}

inline size_t argMax(const std::vector<float>& x) {
    return std::distance(x.begin(), std::max_element(x.begin(), x.end()));
}

inline size_t argMin(const std::vector<float>& x) {
    return std::distance(x.begin(), std::min_element(x.begin(), x.end()));
}

inline std::vector<float> avgPool(const std::vector<float>& x, size_t window_size) {
    std::vector<float> pooled;
    for (size_t i = 0; i <= x.size() - window_size; i += window_size) {
        float sum = 0;
        for (size_t j = 0; j < window_size; ++j) {
            sum += x[i + j];
        }
        pooled.push_back(sum / window_size);
    }
    return pooled;
}

inline void concat(std::vector<float>& x, const std::vector<float>& y) {
    x.insert(x.end(), y.begin(), y.end());
}


#endif // OPTIMIZER_H

#ifndef STOCK_BABA_H
#define STOCK_BABA_H
#include "dataset.h"

// 反归一化数据
inline double denormalize_data(double normalized_data) {
    return normalized_data * (max_val - min_val) + min_val;
}

// 读取 dataset 中前 x% 的数据
inline std::vector<double> read_data(int x) {
    std::vector<double> predict_data_x;
    //dataset是数组，里面的元素是double类型的数据
    int num_elements = sizeof(dataset) / sizeof(dataset[0]);

    // 读取前 x% 的数据
    for (int i = 0; i < num_elements * x / 100; ++i) {
        predict_data_x.push_back(dataset[i]);
    }

    return predict_data_x;
}

// 可能我们现需要做类似这样的事情
// def create_dataset(dataset, time_step=1):
// dataX = []
// for i in range(len(dataset) - time_step - 1):
// a = dataset[i:(i + time_step), 0]
// dataX.append(a)
// return np.array(dataX)
inline std::vector<std::vector<std::vector<double>>> create_dataset(const std::vector<double> &data, int time_steps) {
    std::vector<std::vector<std::vector<double>>> dataX;
    for (size_t i = 0; i < data.size() - time_steps - 1; ++i) {
        std::vector<std::vector<double>> a;
        for (int j = 0; j < time_steps; ++j) {
            std::vector<double> time_step = { data[i + j] };
            a.push_back(time_step);
        }
        dataX.push_back(a);
    }
    return dataX;
    
}


inline std::vector<std::vector<std::vector<double>>> prepare_Xi(const std::vector<double> &data, int time_steps) {
    std::vector<std::vector<std::vector<double>>> Xi;
    
    // 计算可以分割成多少个完整的批次
    int num_batches = data.size() / time_steps;
    
    for (int i = 0; i < num_batches; ++i) {
        std::vector<std::vector<double>> batch;
        for (int j = 0; j < time_steps; ++j) {
            // 每个时间步的数据维度为1
            std::vector<double> time_step = { data[i * time_steps + j] };
            batch.push_back(time_step);
        }
        Xi.push_back(batch);
    }
    
    return Xi;
}

#endif // ! 
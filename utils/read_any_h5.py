import json
import sys
import numpy as np
from tensorflow.keras.models import load_model

def layer_to_dict(layer, input_shape):
    layer_dict = {}
    layer_type = layer.__class__.__name__

    layer_dict["type"] = layer_type
    layer_dict["input_shape"] = input_shape
    weights = layer.get_weights()
    if weights:
        weights_list = [w.tolist() for w in weights]
        if layer_type == "LSTM":
            layer_dict["kernel"] = weights_list[0]
            layer_dict["recurrent_kernel"] = weights_list[1]
            layer_dict["bias"] = weights_list[2]
        else:
            layer_dict["weights"] = weights_list
    config = layer.get_config()
    for key, value in config.items():
        layer_dict[key] = value
    return layer_dict

def model_to_dict(model):
    model_dict = {}
    model_dict["layers"] = []
    input_shape = model.input_shape[1:]  # 获取模型的输入形状
    for layer in model.layers:
        layer_info = layer_to_dict(layer, input_shape)
        model_dict["layers"].append(layer_info)
        input_shape = layer.output_shape[1:]  # 更新输入形状为当前层的输出形状
    return model_dict

def save_to_header(input_model,output_header="model_weights.h"):
    model = load_model(input_model)
    model_dict = model_to_dict(model)
    print(model_dict)
    json_str = json.dumps(model_dict, indent=4)
    cpp_string = f'''#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <string>

const std::string model_config_json = R"json(
{json_str}
)json";

#endif // MODEL_WEIGHTS_H
'''

    # 写入到输出文件
    with open(output_header, "w", encoding="utf-8") as f:
        f.write(cpp_string)

if __name__ == "__main__":
    # save_to_header("../train/mnist/mnist_cnn_model.h5", "model_weights.h")
    # save_to_header("../train/cat_dog/best_cat_dog_model.h5", "model_weights.h")
    save_to_header("../train/stock/lstm_stock_model.h5", "model_weights.h")
    # save_to_header("mobilenetv2.h5", "model_weights.h")
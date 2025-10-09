import json
import sys
import numpy as np
from tensorflow.keras.models import load_model

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def layer_to_dict(layer, input_shape):
    layer_dict = {}
    layer_type = layer.__class__.__name__

    layer_dict["type"] = layer_type
    layer_dict["input_shape"] = convert_to_serializable(input_shape)
    weights = layer.get_weights()
    if weights:
        weights_list = [convert_to_serializable(w) for w in weights]
        if layer_type == "LSTM":
            layer_dict["kernel"] = weights_list[0]
            layer_dict["recurrent_kernel"] = weights_list[1]
            layer_dict["bias"] = weights_list[2]
        else:
            layer_dict["weights"] = weights_list
    config = layer.get_config()
    layer_dict.update(convert_to_serializable(config))
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

def save_to_header(input_model, output_header="model_weights.h", var_name="model_config_json"):
    model = load_model(input_model)
    model_dict = model_to_dict(model)
    json_str = json.dumps(model_dict, indent=None, separators=(",", ":"))  # 压缩JSON，避免超大文本

    # 将 JSON 内容转为字节数组
    byte_array = json_str.encode("utf-8")
    total_len = len(byte_array)

    # 优化性能：用列表收集字符串
    cpp_lines = [f"const char {var_name}[] = {{\n"]
    line = "    "
    for i, b in enumerate(byte_array):
        line += f"0x{b:02x}, "
        if i % 16 == 15:
            cpp_lines.append(line + "\n")
            line = "    "
        # 进度展示，每1000字节打印一次
        if i % 1000 == 0 or i == total_len - 1:
            percent = (i + 1) * 100 // total_len
            print(f"转换进度: {i + 1}/{total_len} ({percent}%)")
    if line.strip():
        cpp_lines.append(line + "\n")
    cpp_lines.append("\n};\n")
    cpp_lines.append(f"const size_t {var_name}_len = sizeof({var_name});\n")

    header_guard = output_header.replace(".", "_").upper()

    cpp_string = f'''#ifndef {header_guard}
#define {header_guard}

#include <cstddef>

{''.join(cpp_lines)}
#endif // {header_guard}
'''

    with open(output_header, "w", encoding="utf-8") as f:
        f.write(cpp_string)

    # 写入到输出文件
    with open(output_header, "w", encoding="utf-8") as f:
        f.write(cpp_string)

if __name__ == "__main__":
    # save_to_header("../train/mnist/mnist_cnn_model.h5", "model_weights.h")
    # save_to_header("../train/cat_dog/best_cat_dog_model.h5", "model_weights.h")
    # save_to_header("../train/stock/lstm_stock_model.h5", "model_weights.h")
    save_to_header("mobilenetv2.h5", "model_weights_v3.h")
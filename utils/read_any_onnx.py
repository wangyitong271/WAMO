import json
import sys
import numpy as np
import onnx
from onnx import numpy_helper


def tensor_to_dict(tensor):
    """将ONNX张量转换为字典格式"""
    tensor_dict = {}
    tensor_dict["name"] = tensor.name
    tensor_dict["shape"] = list(tensor.dims)
    tensor_dict["data_type"] = tensor.data_type
    
    # 将张量数据转换为列表
    tensor_data = numpy_helper.to_array(tensor)
    tensor_dict["data"] = tensor_data.tolist()
    
    return tensor_dict


def node_to_dict(node):
    """将ONNX节点转换为字典格式"""
    node_dict = {}
    node_dict["name"] = node.name
    node_dict["op_type"] = node.op_type
    node_dict["input"] = list(node.input)
    node_dict["output"] = list(node.output)
    
    # 处理属性
    if node.attribute:
        node_dict["attributes"] = {}
        for attr in node.attribute:
            attr_value = None
            if attr.type == onnx.AttributeProto.INT:
                attr_value = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attr_value = attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                attr_value = attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.INTS:
                attr_value = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attr_value = list(attr.floats)
            elif attr.type == onnx.AttributeProto.STRINGS:
                attr_value = [s.decode('utf-8') for s in attr.strings]
            else:
                attr_value = str(attr)
            
            node_dict["attributes"][attr.name] = attr_value
    
    return node_dict


def model_to_dict(model):
    """将ONNX模型转换为字典格式"""
    model_dict = {}
    
    # 基本信息
    model_dict["ir_version"] = model.ir_version
    model_dict["producer_name"] = model.producer_name
    model_dict["producer_version"] = model.producer_version
    model_dict["domain"] = model.domain
    model_dict["model_version"] = model.model_version
    model_dict["doc_string"] = model.doc_string
    
    # 图信息
    graph = model.graph
    model_dict["graph"] = {}
    
    # 输入信息
    model_dict["graph"]["input"] = []
    for input_info in graph.input:
        input_dict = {
            "name": input_info.name,
            "type": str(input_info.type),
            "shape": [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        }
        model_dict["graph"]["input"].append(input_dict)
    
    # 输出信息
    model_dict["graph"]["output"] = []
    for output_info in graph.output:
        output_dict = {
            "name": output_info.name,
            "type": str(output_info.type),
            "shape": [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
        }
        model_dict["graph"]["output"].append(output_dict)
    
    # 节点信息
    model_dict["graph"]["nodes"] = []
    for node in graph.node:
        node_dict = node_to_dict(node)
        model_dict["graph"]["nodes"].append(node_dict)
    
    # 权重信息（初始化的张量）
    model_dict["graph"]["initializers"] = []
    for initializer in graph.initializer:
        tensor_dict = tensor_to_dict(initializer)
        model_dict["graph"]["initializers"].append(tensor_dict)
    
    return model_dict


def save_to_header(input_model, output_header="model_weights.h"):
    """将ONNX模型保存为C++头文件"""
    try:
        # 加载ONNX模型
        model = onnx.load(input_model)
        model_dict = model_to_dict(model)
        
        print("ONNX模型信息:")
        print(f"生产者: {model_dict['producer_name']} {model_dict['producer_version']}")
        print(f"输入数量: {len(model_dict['graph']['input'])}")
        print(f"输出数量: {len(model_dict['graph']['output'])}")
        print(f"节点数量: {len(model_dict['graph']['nodes'])}")
        print(f"权重数量: {len(model_dict['graph']['initializers'])}")
        
        # 转换为JSON字符串
        json_str = json.dumps(model_dict, indent=4, ensure_ascii=False)
        
        # 生成C++头文件内容
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
        
        print(f"成功生成头文件: {output_header}")
        return True
        
    except Exception as e:
        print(f"处理ONNX模型时出错: {str(e)}")
        return False


def extract_weights_only(input_model, output_header="model_weights.h"):
    """仅提取权重信息，生成更简洁的头文件"""
    try:
        # 加载ONNX模型
        model = onnx.load(input_model)
        graph = model.graph
        
        weights_dict = {}
        
        # 提取所有权重
        for initializer in graph.initializer:
            tensor_name = initializer.name
            tensor_data = numpy_helper.to_array(initializer)
            weights_dict[tensor_name] = {
                "shape": list(tensor_data.shape),
                "data": tensor_data.tolist()
            }
        
        # 转换为JSON字符串
        json_str = json.dumps(weights_dict, indent=4, ensure_ascii=False)
        
        # 生成C++头文件内容
        cpp_string = f'''#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <string>

const std::string model_weights_json = R"json(
{json_str}
)json";

#endif // MODEL_WEIGHTS_H
'''
        
        # 写入到输出文件
        with open(output_header, "w", encoding="utf-8") as f:
            f.write(cpp_string)
        
        print(f"成功提取权重并生成头文件: {output_header}")
        print(f"提取了 {len(weights_dict)} 个权重张量")
        return True
        
    except Exception as e:
        print(f"提取ONNX权重时出错: {str(e)}")
        return False


if __name__ == "__main__":
    save_to_header("../model/onnx_model/yolov8n.onnx", "model_weights.h")
    
 
    # 测试文件（如果存在）
    # test_files = [
    #     "../model/onnx_model/yolov8n.onnx",
    #     "../model/onnx_model/nms-yolov8.onnx"
    # ]
    
    # for test_file in test_files:
    #     try:
    #         import os
    #         if os.path.exists(test_file):
    #             print(f"\n处理文件: {test_file}")
    #             save_to_header(test_file, f"weights_{os.path.basename(test_file).replace('.onnx', '')}.h")
    #             break
    #     except Exception as e:
    #         print(f"处理 {test_file} 时出错: {e}")
    
    # print("\n如果没有找到测试文件，请手动指定ONNX文件路径")

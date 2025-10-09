import json
import sys
import numpy as np
import torch


def tensor_to_dict(tensor, name):
    """将PyTorch张量转换为字典格式"""
    tensor_dict = {}
    tensor_dict["name"] = name
    tensor_dict["shape"] = list(tensor.shape)
    tensor_dict["dtype"] = str(tensor.dtype)
    tensor_dict["device"] = str(tensor.device)
    
    # 将张量数据转换为列表
    tensor_dict["data"] = tensor.cpu().numpy().tolist()
    
    return tensor_dict


def state_dict_to_dict(state_dict):
    """将PyTorch state_dict转换为字典格式"""
    state_dict_info = {}
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensor_dict = tensor_to_dict(value, key)
            state_dict_info[key] = tensor_dict
        else:
            # 处理非张量值（如标量）
            state_dict_info[key] = {
                "name": key,
                "value": value,
                "type": str(type(value).__name__)
            }
    
    return state_dict_info


def model_to_dict(model_path):
    """将PyTorch模型转换为字典格式"""
    model_dict = {}
    
    try:
        # 尝试加载模型
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 检查checkpoint的类型
        if isinstance(checkpoint, dict):
            # 如果是字典，可能包含state_dict和其他信息
            if 'state_dict' in checkpoint:
                # 包含state_dict的情况
                model_dict["state_dict"] = state_dict_to_dict(checkpoint['state_dict'])
                
                # 提取其他信息
                for key, value in checkpoint.items():
                    if key != 'state_dict':
                        if isinstance(value, torch.Tensor):
                            model_dict[key] = tensor_to_dict(value, key)
                        else:
                            model_dict[key] = {
                                "name": key,
                                "value": value,
                                "type": str(type(value).__name__)
                            }
            else:
                # 直接是state_dict的情况
                model_dict["state_dict"] = state_dict_to_dict(checkpoint)
        else:
            # 如果是直接的模型对象
            if hasattr(checkpoint, 'state_dict'):
                model_dict["state_dict"] = state_dict_to_dict(checkpoint.state_dict())
            else:
                # 尝试作为state_dict处理
                model_dict["state_dict"] = state_dict_to_dict(checkpoint)
        
        # 添加模型基本信息
        model_dict["model_info"] = {
            "file_path": model_path,
            "total_parameters": len(model_dict.get("state_dict", {})),
            "parameter_types": {}
        }
        
        # 统计参数类型
        if "state_dict" in model_dict:
            for key, param_info in model_dict["state_dict"].items():
                if isinstance(param_info, dict) and "shape" in param_info:
                    shape_str = str(param_info["shape"])
                    if shape_str not in model_dict["model_info"]["parameter_types"]:
                        model_dict["model_info"]["parameter_types"][shape_str] = 0
                    model_dict["model_info"]["parameter_types"][shape_str] += 1
        
        return model_dict
        
    except Exception as e:
        print(f"加载PyTorch模型时出错: {str(e)}")
        return None


def save_to_header(input_model, output_header="model_weights.h"):
    """将PyTorch模型保存为C++头文件"""
    try:
        # 加载PyTorch模型
        model_dict = model_to_dict(input_model)
        if model_dict is None:
            return False
        
        print("PyTorch模型信息:")
        print(f"总参数数量: {model_dict['model_info']['total_parameters']}")
        print(f"参数类型分布: {model_dict['model_info']['parameter_types']}")
        
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
        print(f"处理PyTorch模型时出错: {str(e)}")
        return False


def extract_weights_only(input_model, output_header="model_weights.h"):
    """仅提取权重信息，生成更简洁的头文件"""
    try:
        # 加载PyTorch模型
        checkpoint = torch.load(input_model, map_location=torch.device('cpu'))
        
        weights_dict = {}
        
        # 提取state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
            else:
                state_dict = checkpoint
        
        # 提取所有权重
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                weights_dict[key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "data": tensor.cpu().numpy().tolist()
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
        print(f"提取了 {len(weights_dict)} 个权重参数")
        return True
        
    except Exception as e:
        print(f"提取PyTorch权重时出错: {str(e)}")
        return False


def analyze_model_structure(input_model):
    """分析PyTorch模型结构"""
    try:
        checkpoint = torch.load(input_model, map_location=torch.device('cpu'))
        
        print(f"分析模型文件: {input_model}")
        print("=" * 50)
        
        if isinstance(checkpoint, dict):
            print("模型类型: 字典格式")
            print(f"顶层键: {list(checkpoint.keys())}")
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"State dict参数数量: {len(state_dict)}")
                
                # 显示参数名称和形状
                print("\n参数详情:")
                for key, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        print(f"  {key}: {list(tensor.shape)} ({tensor.dtype})")
                    else:
                        print(f"  {key}: {type(tensor)}")
            
            # 显示其他信息
            for key, value in checkpoint.items():
                if key != 'state_dict':
                    print(f"{key}: {type(value)}")
        else:
            print("模型类型: 直接模型对象")
            if hasattr(checkpoint, 'state_dict'):
                state_dict = checkpoint.state_dict()
                print(f"State dict参数数量: {len(state_dict)}")
                
                print("\n参数详情:")
                for key, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        print(f"  {key}: {list(tensor.shape)} ({tensor.dtype})")
                    else:
                        print(f"  {key}: {type(tensor)}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"分析模型结构时出错: {str(e)}")


if __name__ == "__main__":
    # 示例用法
    # 完整模型信息导出
    # save_to_header("model.pth", "model_weights.h")
    
    # 仅权重导出
    # extract_weights_only("model.pth", "model_weights.h")
    
    # 分析模型结构
    # analyze_model_structure("model.pth")
    
    print("PyTorch模型处理工具")
    print("使用方法:")
    print("1. save_to_header(model_path, output_header) - 导出完整模型信息")
    print("2. extract_weights_only(model_path, output_header) - 仅导出权重")
    print("3. analyze_model_structure(model_path) - 分析模型结构")
    print("\n如果没有找到测试文件，请手动指定PyTorch文件路径")

"""
模型处理模块
负责处理不同模型文件的导出和WASM生成逻辑
"""

import os
import subprocess
import threading
import shutil
from exports.export_cat_dog_params import export_cat_dog
from exports.export_lstm_stock_params import export_lstm_stock
from exports.export_mnist_params import export_mnist
from exports.export_transformer_params import export_transformer
from exports.export_weather_params import export_weather
from exports.export_resnet_params import export_resnet
from exports.export_gan_params import export_gan
from utils.read_any_h5 import save_to_header as save_h5_to_header
from utils.read_any_onnx import save_to_header as save_onnx_to_header
from utils.read_any_pt import save_to_header as save_pt_to_header


class ModelHandler:
    """模型处理器类，负责处理各种模型的导出和WASM生成"""
    
    def __init__(self, ui_callback=None):
        """
        初始化模型处理器
        Args:
            ui_callback: UI回调函数，用于更新UI状态
        """
        self.ui_callback = ui_callback
        self.supported_models = {
            "mnist_cnn_model.h5": {
                "framework": "Keras",
                "export_func": export_mnist,
                "wasm_dir": "model/mnist",
                "description": "mnist手写数字识别"
            },
            "weather_rnn.h5": {
                "framework": "Keras", 
                "export_func": export_weather,
                "wasm_dir": "model/weather",
                "description": "RNN天气预测"
            },
            "best_cat_dog_model.h5": {
                "framework": "Keras",
                "export_func": export_cat_dog,
                "wasm_dir": "model/cat_dog", 
                "description": "cat dog 识别"
            },
            "lstm_stock_model.h5": {
                "framework": "Keras",
                "export_func": export_lstm_stock,
                "wasm_dir": "model/stock",
                "description": "LSTM股票预测"
            },
            "occupancy_model_transformer.h5": {
                "framework": "Keras",
                "export_func": export_transformer,
                "wasm_dir": "model/transformer",
                "description": "transformer"
            },
            "letter_recognition_model.h5": {
                "framework": "Keras",
                "export_func": export_resnet,
                "wasm_dir": "model/resnet",
                "description": "ResNet英文字符识别"
            },
            "wave_generator_model.h5": {
                "framework": "Keras",
                "export_func": export_gan,
                "wasm_dir": "model/gan",
                "description": "GAN波形生成"
            }
        }
    
    def detect_model_type(self, filename):
        """检测模型文件类型并返回对应的框架"""
        filename_lower = filename.lower()
        if filename_lower.endswith('.h5'):
            return 'Keras'
        elif filename_lower.endswith('.joblib'):
            return 'Scikit-learn'
        elif filename_lower.endswith('.pth') or filename_lower.endswith('.pt'):
            return 'PyTorch'
        elif filename_lower.endswith('.onnx'):
            return 'ONNX'
        else:
            return 'Unknown'
    
    def is_model_supported(self, model_file):
        """检查模型是否被支持"""
        return model_file in self.supported_models
    
    def get_model_info(self, model_file):
        """获取模型信息"""
        return self.supported_models.get(model_file, None)
    
    def get_supported_models(self):
        """获取所有支持的模型列表"""
        return list(self.supported_models.keys())
    
    def export_model_cpp(self, model_file, framework):
        """
        导出模型为CPP代码
        Args:
            model_file: 模型文件名
            framework: 框架类型
        Returns:
            bool: 是否成功
        """
        if not self.is_model_supported(model_file):
            return False, f"不支持的模型: {model_file}"
        
        model_info = self.get_model_info(model_file)
        if model_info["framework"] != framework:
            return False, f"模型类型应该选择{model_info['framework']}"
        
        try:
            # 禁用UI按钮
            if self.ui_callback:
                self.ui_callback('disable_buttons')
            
            # 执行导出
            model_info["export_func"]()
            
            # 启用UI按钮
            if self.ui_callback:
                self.ui_callback('enable_buttons')
            
            return True, f"成功导出模型: {model_file}"
            
        except Exception as e:
            # 确保按钮被重新启用
            if self.ui_callback:
                self.ui_callback('enable_buttons')
            return False, f"导出失败: {str(e)}"
    
    def export_model_wasm(self, model_file):
        """
        导出模型为WASM
        Args:
            model_file: 模型文件名
        Returns:
            bool: 是否成功
        """
        if not self.is_model_supported(model_file):
            return False, f"不支持的模型: {model_file}"
        
        model_info = self.get_model_info(model_file)
        
        try:
            # 禁用UI按钮
            if self.ui_callback:
                self.ui_callback('disable_buttons')
            
            # 执行WASM导出
            success = self._export_wasm_thread(model_info["wasm_dir"], model_info["description"])
            
            # 启用UI按钮
            if self.ui_callback:
                self.ui_callback('enable_buttons')
            
            if success:
                return True, f"成功导出WASM: {model_file}"
            else:
                return False, f"WASM导出失败: {model_file}"
                
        except Exception as e:
            # 确保按钮被重新启用
            if self.ui_callback:
                self.ui_callback('enable_buttons')
            return False, f"WASM导出出错: {str(e)}"
    
    def _export_wasm_thread(self, model_dir, model_name):
        """
        内部方法：执行WASM导出
        Args:
            model_dir: 模型目录
            model_name: 模型名称
        Returns:
            bool: 是否成功
        """
        try:
            print("当前工作目录:", os.getcwd())
            os.chdir(model_dir)
            print("当前工作目录:", os.getcwd())
            
            # 检查compile.bat文件是否存在
            if not os.path.exists("compile.bat"):
                print(f"错误: 在目录 {model_dir} 中找不到 compile.bat 文件")
                os.chdir("../../")
                return False
            
            print(f"开始执行 compile.bat...")
            process = subprocess.run(
                ["cmd.exe", "/c", "compile.bat"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            # 检查返回码
            # if process.returncode == 0:
            #     print(f"compile.bat 执行成功")
            #     print(f"标准输出: {process.stdout}")
            # else:
            #     print(f"compile.bat 执行失败，返回码: {process.returncode}")
            #     print(f"标准输出: {process.stdout}")
            #     print(f"错误输出: {process.stderr}")
            #     os.chdir("../../")
            #     return False
            
            os.chdir("../../")
            print(f"导出为 WASM {model_name}")
            return True
            
        except Exception as e:
            print(f"WASM导出出错: {str(e)}")
            # 确保回到原始目录
            try:
                os.chdir("../../")
            except:
                pass
            return False
    
    def export_generic_model_cpp(self, model_path, detected_framework):
        """
        导出通用模型为CPP代码（粗糙版本，不保证能运行）
        Args:
            model_path: 模型文件的完整路径
            detected_framework: 检测到的框架
        Returns:
            tuple: (success, message)
        """
        try:
            # 禁用UI按钮
            if self.ui_callback:
                self.ui_callback('disable_buttons')
            
            # 检查文件是否存在
            if not os.path.exists(model_path):
                if self.ui_callback:
                    self.ui_callback('enable_buttons')
                return False, f"模型文件不存在: {model_path}"
            
            # 创建模型目录
            model_filename = os.path.basename(model_path)
            model_name = os.path.splitext(model_filename)[0]  # 去掉扩展名
            model_dir = f"model/{model_name}"
            os.makedirs(model_dir, exist_ok=True)
            
            # 根据框架类型选择相应的处理函数
            if detected_framework == "Keras":
                # 处理H5文件
                success = save_h5_to_header(model_path, f"{model_dir}/model_weights.h")
                if not success:
                    if self.ui_callback:
                        self.ui_callback('enable_buttons')
                    return False, f"H5模型处理失败: {model_filename}"
                    
            elif detected_framework == "ONNX":
                # 处理ONNX文件
                success = save_onnx_to_header(model_path, f"{model_dir}/model_weights.h")
                if not success:
                    if self.ui_callback:
                        self.ui_callback('enable_buttons')
                    return False, f"ONNX模型处理失败: {model_filename}"
                    
            elif detected_framework == "PyTorch":
                # 处理PyTorch文件
                success = save_pt_to_header(model_path, f"{model_dir}/model_weights.h")
                if not success:
                    if self.ui_callback:
                        self.ui_callback('enable_buttons')
                    return False, f"PyTorch模型处理失败: {model_filename}"
            else:
                if self.ui_callback:
                    self.ui_callback('enable_buttons')
                return False, f"不支持的框架类型: {detected_framework}"
            
            # 复制必要的头文件和源文件
            self._copy_generic_cpp_files(model_dir, model_name)
            
            # 启用UI按钮
            if self.ui_callback:
                self.ui_callback('enable_buttons')
            
            return True, f"成功导出通用模型: {model_filename} (框架: {detected_framework})"
            
        except Exception as e:
            # 确保按钮被重新启用
            if self.ui_callback:
                self.ui_callback('enable_buttons')
            return False, f"通用模型导出失败: {str(e)}"
    
    def _copy_generic_cpp_files(self, model_dir, model_name):
        """
        复制通用的CPP文件到模型目录
        Args:
            model_dir: 模型目录路径
            model_name: 模型名称
        """
        try:
            # 复制基础头文件
            base_files = [
                ("cpp_src/WASM_lib/network.h", f"{model_dir}/network.h"),
                ("cpp_src/WASM_lib/read_model.h", f"{model_dir}/read_model.h"),
                ("cpp_src/third_party_lib/json.hpp", f"{model_dir}/json.hpp")
            ]
            
            for src, dst in base_files:
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"复制文件: {src} -> {dst}")
            
            # 复制加密相关文件
            crypto_files = [
                ("cpp_src/crypt/rsa.h", f"{model_dir}/rsa.h"),
                ("cpp_src/crypt/encode_rsa.cpp", f"{model_dir}/encode_rsa.cpp"),
                ("cpp_src/crypt/model_weights_loader.h", f"{model_dir}/model_weights_loader.h"),
                ("cpp_src/crypt/generate_headers.cpp", f"{model_dir}/generate_headers.cpp"),
                ("cpp_src/crypt/feistel_crypto.cpp", f"{model_dir}/feistel_crypto.cpp"),
                ("cpp_src/crypt/feistel_crypto.h", f"{model_dir}/feistel_crypto.h")
            ]
            
            for src, dst in crypto_files:
                if os.path.exists(src):
                    shutil.copy(src, dst)
                    print(f"复制加密文件: {src} -> {dst}")
            
            # 复制一个通用的CPP源文件并重命名
            generic_cpp = "cpp_src/model_predict/image_classify_emcc.cpp"
            if os.path.exists(generic_cpp):
                shutil.copy(generic_cpp, f"{model_dir}/{model_name}.cpp")
                print(f"复制通用CPP文件: {generic_cpp} -> {model_dir}/{model_name}.cpp")
            
            print(f"成功复制所有文件到: {model_dir}")
            
        except Exception as e:
            print(f"复制文件时出错: {str(e)}")
            raise

    def handle_unsupported_model(self, model_file, detected_framework):
        """
        处理不支持的模型
        Args:
            model_file: 模型文件名
            detected_framework: 检测到的框架
        Returns:
            tuple: (is_handled, message)
        """
        if detected_framework in ["PyTorch", "ONNX"]:
            message = (f"检测到{detected_framework}模型文件: {model_file}\n"
                      f"当前版本支持读取和解析该文件格式，但暂不支持导出为WASM。\n"
                      f"请使用支持的Keras模型文件进行WASM导出。")
            return True, message
        else:
            message = (f"暂时不支持该模型的导出: {model_file}\n"
                      f"检测到的框架类型: {detected_framework}")
            return True, message

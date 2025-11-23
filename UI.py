import os
import subprocess
import sys
import threading
import shutil
from datetime import datetime

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QPushButton, QGridLayout, QLabel, QFileDialog, QLineEdit, \
    QMessageBox, QRadioButton, QVBoxLayout, QGroupBox, QHBoxLayout, QCheckBox


# 导入模型处理器
from model_handler import ModelHandler

# 导入控制流混淆模块
from confuse.operator_execution_path_perturbation import operator_execution_path_perturbation
# 导入二进制混淆相关模块
from confuse.wasm_obfuscator.dummy_operator_injection import dummy_operator_injection
from confuse.wasm_obfuscator.function_obfuscator import obfuscate_wasm_functions
from confuse.wasm_obfuscator.metadata_obfuscation import metadata_obfuscation
# 导入新增的混淆模块
from confuse.wasm_obfuscator.simple_indirect_call_obfuscator import obfuscate_wasm_simple_indirect_call
from confuse.wasm_obfuscator.operator_dispatch_indexing import operator_dispatch_indexing
from confuse.wasm_obfuscator.intermediate_computation_obfuscation import intermediate_computation_obfuscation
from confuse.wasm_obfuscator.operator_decision_perturbation import operator_decision_perturbation
# 导入数据段混淆模块
from confuse.wasm_obfuscator.dummy_weight_injection_obfuscator import add_string_to_wasm_data
# 导入假解密混淆模块
from confuse.wasm_obfuscator.fake_decryption_obfuscation import add_simple_start_section


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.model_handler = ModelHandler(ui_callback=self.ui_callback)
        self.current_model_path = None
        self.initUI()
        self.resize(520, 620)

    def ui_callback(self, action):
        if action == 'disable_buttons':
            self.disable_button()
        elif action == 'enable_buttons':
            self.enable_button()

    def initUI(self):
        font = QFont()
        font.setPointSize(16)

        # === 模型部分 ===
        self.labelSelectModel = QLabel('选择模型', self)
        self.labelSelectModel.setFont(font)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setReadOnly(True)
        self.lineEdit.mousePressEvent = self.open_model_file_dialog
        self.lineEdit.setFont(font)

        self.labelFramework = QLabel('模型框架', self)
        self.labelFramework.setFont(font)

        self.comboBoxFramework = QComboBox(self)
        self.comboBoxFramework.addItems(["Keras", "TensorFlow", "PyTorch", "Scikit-learn"])
        self.comboBoxFramework.setFont(font)

        # === WASM文件部分 ===
        self.labelSelectWasm = QLabel('WASM文件', self)
        self.labelSelectWasm.setFont(font)

        self.wasmFileEdit = QLineEdit(self)
        self.wasmFileEdit.setReadOnly(True)
        self.wasmFileEdit.setFont(font)

        self.buttonBrowseWasm = QPushButton('浏览...', self)
        self.buttonBrowseWasm.setFont(font)
        self.buttonBrowseWasm.clicked.connect(self.open_wasm_file_dialog)

        # === 二进制混淆选项部分（修改为多选框） ===
        self.obfuscationGroup = QGroupBox("二进制混淆选项")
        self.obfuscationGroup.setFont(font)
        vbox = QVBoxLayout()

        self.radioTypeObfuscation = QCheckBox("虚假算子注入", self)
        self.radioTypeObfuscation.setFont(font)

        self.radioFunctionObfuscation = QCheckBox("虚假权重注入", self)
        self.radioFunctionObfuscation.setFont(font)

        self.radioCustomSectionObfuscation = QCheckBox("元数据混淆", self)
        self.radioCustomSectionObfuscation.setFont(font)

        self.radioControlFlowObfuscation = QCheckBox("算子执行路径扰动", self)
        self.radioControlFlowObfuscation.setFont(font)

        self.radioIndirectCallObfuscation = QCheckBox("算子分派索引", self)
        self.radioIndirectCallObfuscation.setFont(font)

        self.radioStack1Obfuscation = QCheckBox("中间计算混淆", self)
        self.radioStack1Obfuscation.setFont(font)

        self.radioStack2Obfuscation = QCheckBox("算子决策扰动", self)
        self.radioStack2Obfuscation.setFont(font)

        self.radioFakeDecryptionObfuscation = QCheckBox("伪装解密混淆", self)
        self.radioFakeDecryptionObfuscation.setFont(font)

        self.radioDataSectionObfuscation = QCheckBox("虚假权重注入", self)
        self.radioDataSectionObfuscation.setFont(font)
        self.radioDataSectionObfuscation.setVisible(False)

        # 加入布局
        vbox.addWidget(self.radioTypeObfuscation)
        vbox.addWidget(self.radioFunctionObfuscation)
        vbox.addWidget(self.radioCustomSectionObfuscation)
        vbox.addWidget(self.radioControlFlowObfuscation)
        vbox.addWidget(self.radioIndirectCallObfuscation)
        vbox.addWidget(self.radioStack1Obfuscation)
        vbox.addWidget(self.radioStack2Obfuscation)
        vbox.addWidget(self.radioFakeDecryptionObfuscation)
        vbox.addWidget(self.radioDataSectionObfuscation)

        self.obfuscationGroup.setLayout(vbox)

        # === 按钮区域 ===
        self.buttonCpp = QPushButton('生成 CPP', self)
        self.buttonWasm = QPushButton('导出为 WASM', self)
        self.buttonBinaryObfuscate = QPushButton('二进制混淆', self)

        self.buttonCpp.setFont(font)
        self.buttonWasm.setFont(font)
        self.buttonBinaryObfuscate.setFont(font)

        self.buttonCpp.clicked.connect(self.generate_cpp)
        self.buttonWasm.clicked.connect(self.export_wasm)
        self.buttonBinaryObfuscate.clicked.connect(self.binary_obfuscate_wasm)

        grid = QGridLayout()
        grid.addWidget(self.labelSelectModel, 0, 0)
        grid.addWidget(self.lineEdit, 0, 1, 1, 2)

        grid.addWidget(self.labelFramework, 1, 0)
        grid.addWidget(self.comboBoxFramework, 1, 1, 1, 2)

        grid.addWidget(self.labelSelectWasm, 2, 0)
        grid.addWidget(self.wasmFileEdit, 2, 1)
        grid.addWidget(self.buttonBrowseWasm, 2, 2)

        grid.addWidget(self.obfuscationGroup, 3, 0, 1, 3)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.buttonCpp)
        buttonLayout.addWidget(self.buttonWasm)
        buttonLayout.addWidget(self.buttonBinaryObfuscate)

        grid.addLayout(buttonLayout, 4, 0, 1, 3)

        self.setLayout(grid)
        self.setWindowTitle('WASM 生成工具')
        self.show()

    def open_model_file_dialog(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开模型文件", "",
                                                   "All Supported Models (*.h5 *.joblib *.pt *.onnx);;H5 files (*.h5);;Joblib files (*.joblib);;PyTorch files (*.pt);;ONNX files (*.onnx);;All files (*.*)")
        if file_path:
            filename = os.path.basename(file_path)
            self.lineEdit.setText(filename)
            # 保存完整的文件路径
            self.current_model_path = file_path
            # 自动检测并设置框架
            detected_framework = self.model_handler.detect_model_type(filename)
            if detected_framework != 'Unknown':
                # 查找对应的框架索引
                framework_index = self.comboBoxFramework.findText(detected_framework)
                if framework_index >= 0:
                    self.comboBoxFramework.setCurrentIndex(framework_index)
                    print(f"自动检测到模型类型: {detected_framework}")
                else:
                    print(f"检测到模型类型 {detected_framework}，但框架选择框中没有对应选项")
            else:
                print(f"无法识别的模型文件类型: {filename}")
            
    def open_wasm_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择WASM文件", "",
                                                   "WASM files (*.wasm)")
        if file_path:
            self.wasmFileEdit.setText(file_path)

    def disable_button(self):
        self.buttonCpp.setEnabled(False)
        self.buttonWasm.setEnabled(False)
        self.buttonBinaryObfuscate.setEnabled(False)
        self.buttonBrowseWasm.setEnabled(False)

    def enable_button(self):
        self.buttonCpp.setEnabled(True)
        self.buttonWasm.setEnabled(True)
        self.buttonBinaryObfuscate.setEnabled(True)
        self.buttonBrowseWasm.setEnabled(True)



    def generate_cpp(self):
        """生成CPP代码"""
        model_file = self.lineEdit.text()
        framework = self.comboBoxFramework.currentText()
        detected_framework = self.model_handler.detect_model_type(model_file)
        
        # 检查是否为支持的模型
        if self.model_handler.is_model_supported(model_file):
            try:
                thread = threading.Thread(
                    target=self._export_cpp_thread, 
                    args=(model_file, framework)
                )
                thread.start()
            except Exception as e:
                self.enable_button()
                print(f"导出CPP出错: {e}")
        else:
            # 对于不支持的模型，尝试通用导出
            if detected_framework in ["Keras", "ONNX", "PyTorch"]:
                # 使用完整的文件路径
                model_path = getattr(self, 'current_model_path', model_file)
                try:
                    thread = threading.Thread(
                        target=self._export_generic_cpp_thread, 
                        args=(model_path, detected_framework)
                    )
                    thread.start()
                except Exception as e:
                    self.enable_button()
                    print(f"通用导出CPP出错: {e}")
            else:
                # 处理完全不支持的模型
                is_handled, message = self.model_handler.handle_unsupported_model(model_file, detected_framework)
                # if is_handled:
                #     QMessageBox.information(self, "提示", message, QMessageBox.Yes)
                #     print(message)
    
    def _export_cpp_thread(self, model_file, framework):
        """内部方法：执行CPP导出"""
        success, message = self.model_handler.export_model_cpp(model_file, framework)
        if not success:
            QMessageBox.warning(self, "错误", message, QMessageBox.Yes)
        print(message)
    
    def _export_generic_cpp_thread(self, model_file, detected_framework):
        """内部方法：执行通用CPP导出"""
        success, message = self.model_handler.export_generic_model_cpp(model_file, detected_framework)
        # if not success:
        #     QMessageBox.warning(self, "错误", message, QMessageBox.Yes)
        print(message)


    def export_wasm(self):
        """导出为WASM"""
        model_file = self.lineEdit.text()
        detected_framework = self.model_handler.detect_model_type(model_file)
        
        # 检查是否为支持的模型
        if self.model_handler.is_model_supported(model_file):
            try:
                thread = threading.Thread(
                    target=self._export_wasm_thread, 
                    args=(model_file,)
                )
                thread.start()
            except Exception as e:
                self.enable_button()
                print(f"导出WASM出错: {e}")
        else:
            # 处理不支持的模型
            is_handled, message = self.model_handler.handle_unsupported_model(model_file, detected_framework)
            if is_handled:
                QMessageBox.information(self, "提示", message, QMessageBox.Yes)
                print(message)
    
    def _export_wasm_thread(self, model_file):
        """内部方法：执行WASM导出"""
        success, message = self.model_handler.export_model_wasm(model_file)
        if not success:
            QMessageBox.warning(self, "错误", message, QMessageBox.Yes)
        print(message)
    
    def binary_obfuscate_wasm_thread(self, input_file):
        """对指定的WASM文件执行二进制混淆操作，生成单一指定类型的混淆"""
        try:
            self.disable_button()
            print(f"开始二进制混淆: {input_file}")
                
            # 获取文件信息
            file_path = os.path.abspath(input_file)
            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            file_base, file_ext = os.path.splitext(file_name)
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"错误: 找不到输入文件: {file_path}")
                self.enable_button()
                return
            
            # 确定混淆类型
            obfuscation_type = ""
            obfuscation_description = ""
            
            if self.radioTypeObfuscation.isChecked():
                obfuscation_type = "type"
                obfuscation_description = "Type Obfuscation"
            elif self.radioFunctionObfuscation.isChecked():
                obfuscation_type = "function"
                obfuscation_description = "Function Obfuscation"
            elif self.radioCustomSectionObfuscation.isChecked():
                obfuscation_type = "custom"
                obfuscation_description = "Custom Section Obfuscation"
            elif self.radioControlFlowObfuscation.isChecked():
                obfuscation_type = "control_flow"
                obfuscation_description = "Control Flow Obfuscation"
            elif self.radioIndirectCallObfuscation.isChecked():
                obfuscation_type = "indirect_call"
                obfuscation_description = "Indirect Call Obfuscation"
            elif self.radioStack1Obfuscation.isChecked():
                obfuscation_type = "stack1"
                obfuscation_description = "Stack Obfuscation (Type 1)"
            elif self.radioStack2Obfuscation.isChecked():
                obfuscation_type = "stack2"
                obfuscation_description = "Stack Obfuscation (Type 2)"
            elif self.radioFakeDecryptionObfuscation.isChecked():
                obfuscation_type = "fake_decryption"
                obfuscation_description = "Fake Decryption Obfuscation"
            elif self.radioDataSectionObfuscation.isChecked():
                obfuscation_type = "data_section"
                obfuscation_description = "Data Section Obfuscation"
            else:
                print("错误: 未选择混淆类型")
                self.enable_button()
                return
            
            # 创建输出文件夹名称，使用英文
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ob_dir = os.path.join(file_dir, f"ob_{obfuscation_type}_{timestamp}")
            os.makedirs(ob_dir, exist_ok=True)
            print(f"创建输出文件夹: {ob_dir}")
            
            # 复制原始文件到final文件夹中
            copy_success = False
            try:
                output_wasm = os.path.join(ob_dir, file_name)
                shutil.copy2(file_path, output_wasm)
                copy_success = True
                print(f"已复制原始文件到: {output_wasm}")
            except Exception as e:
                print(f"复制文件时出错: {str(e)}")
                self.enable_button()
                return
            
            if not copy_success:
                print("错误: 无法复制原始文件")
                self.enable_button()
                return
            
            # 设置混淆参数
            num_types = 25
            num_funcs = 1  # 为栈混淆设置较小的函数数量
            complexity = 1
            compiler = "rustc"
            
            # 执行选择的混淆
            success = False
            
            if obfuscation_type == "type":
                # 类型混淆
                print("执行类型混淆...")
                try:
                    success = dummy_operator_injection(output_wasm, output_wasm, num_types)
                    if success:
                        print(f"类型混淆成功: {output_wasm}")
                    else:
                        print("类型混淆失败")
                except Exception as e:
                    print(f"类型混淆出错: {str(e)}")
            
            elif obfuscation_type == "function":
                # 函数混淆
                print("执行虚假权重注入...")
                try:
                    success = obfuscate_wasm_functions(output_wasm, output_wasm, num_funcs, complexity)
                    if success:
                        print(f"虚假权重注入成功: {output_wasm}")
                    else:
                        print("虚假权重注入失败")
                except Exception as e:
                    print(f"虚假权重注入出错: {str(e)}")
            
            elif obfuscation_type == "custom":
                # 自定义段混淆
                print("执行自定义段混淆...")
                try:
                    success = metadata_obfuscation(output_wasm, output_wasm, compiler)
                    if success:
                        print(f"自定义段混淆成功: {output_wasm}")
                    else:
                        print("自定义段混淆失败")
                except Exception as e:
                    print(f"自定义段混淆出错: {str(e)}")
            
            elif obfuscation_type == "control_flow":
                # 控制流展平
                print("执行控制流展平...")
                try:
                    temp_output = os.path.join(ob_dir, f"temp_{file_name}")
                    success = operator_execution_path_perturbation(output_wasm, temp_output)
                    if success and os.path.exists(temp_output):
                        shutil.copy2(temp_output, output_wasm)
                        os.remove(temp_output)
                        print(f"控制流展平成功: {output_wasm}")
                    else:
                        print("控制流展平失败")
                except Exception as e:
                    print(f"控制流展平出错: {str(e)}")
            
            elif obfuscation_type == "indirect_call":
                # 间接调用混淆
                print("执行间接调用混淆...")
                try:
                    # success = obfuscate_wasm_simple_indirect_call(output_wasm, output_wasm)
                    success = operator_dispatch_indexing(output_wasm, output_wasm)
                    if success:
                        print(f"间接调用混淆成功: {output_wasm}")
                    else:
                        print("间接调用混淆失败")
                except Exception as e:
                    print(f"间接调用混淆出错: {str(e)}")
            
            elif obfuscation_type == "stack1":
                # 栈混淆1
                print("执行栈混淆(类型1)...")
                try:
                    success = intermediate_computation_obfuscation(output_wasm, output_wasm, num_funcs)
                    if success:
                        print(f"栈混淆(类型1)成功: {output_wasm}")
                    else:
                        print("栈混淆(类型1)失败")
                except Exception as e:
                    print(f"栈混淆(类型1)出错: {str(e)}")
            
            elif obfuscation_type == "stack2":
                # 栈混淆2
                print("执行栈混淆(类型2)...")
                try:
                    success = operator_decision_perturbation(output_wasm, output_wasm, num_funcs)
                    if success:
                        print(f"栈混淆(类型2)成功: {output_wasm}")
                    else:
                        print("栈混淆(类型2)失败")
                except Exception as e:
                    print(f"栈混淆(类型2)出错: {str(e)}")
            
            elif obfuscation_type == "fake_decryption":
                # 假解密混淆
                print("执行假解密混淆...")
                try:
                    success = add_simple_start_section(output_wasm, output_wasm)
                    if success:
                        print(f"假解密混淆成功: {output_wasm}")
                    else:
                        print("假解密混淆失败")
                except Exception as e:
                    print(f"假解密混淆出错: {str(e)}")
            
            elif obfuscation_type == "data_section":
                # 数据段混淆
                print("执行数据段混淆...")
                try:
                    # 添加测试字符串，使用独立文件模式避免WASM文件损坏
                    test_string = "WasmCrypt_DataObfuscation_" + timestamp
                    success = add_string_to_wasm_data(output_wasm, output_wasm, test_string, use_separate_file=False)
                    if success:
                        print(f"数据段混淆成功: {output_wasm}")
                        print(f"添加的数据: {test_string}")
                    else:
                        print("数据段混淆失败")
                except Exception as e:
                    print(f"数据段混淆出错: {str(e)}")
            
        except Exception as e:
            print(f"二进制混淆过程中发生错误: {str(e)}")
        
        finally:
            # 确保按钮被重新启用
            self.enable_button()
            print("二进制混淆处理完成")

    def binary_obfuscate_wasm(self):
        """执行二进制混淆操作，使用选择的WASM文件"""
        wasm_file = self.wasmFileEdit.text()
        if not wasm_file:
            QMessageBox.warning(self, "警告", "请先选择WASM文件", QMessageBox.Yes)
            return
            
        try:
            # 使用 QApplication.processEvents() 确保UI更新
            QApplication.processEvents()
            
            thread = threading.Thread(target=self.binary_obfuscate_wasm_thread, args=(wasm_file,))
            thread.daemon = True  # 将线程设为守护线程，确保主程序退出时线程也会退出
            thread.start()
        except Exception as e:
            self.enable_button()
            print(f"二进制混淆出错: {e}")
            QMessageBox.warning(self, "错误", f"二进制混淆出错: {str(e)}", QMessageBox.Yes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

README

1.我们将在本章中展示如何设置 Emscripten，您将需要在您的系统上安装以下内容：

Python >= 3.6

Node.js > 12.18

注意

emsdk 自带一个兼容的 Node.js 版本。

2. 使用emsdk安装Emscripten

3. 克隆emsdk仓库并进入emsdk文件夹：

$ git clone https://github.com/emscripten-core/emsdk

$ cd emsdk 

$ emsdk install latest

前述命令可能需要一些时间来运行；它将构建并设置整个工具链。

接下来，我们将激活最新版本的emsdk。激活将更新本地shell，添加所需的环境引用，并使当前shell中的用户使用最新的SDK。它会将Emscripten工具链的路径和其他必要信息写入用户主目录下名为.emscripten的文件：

4. 要激活已安装的emsdk，请运行以下命令：
   $ emsdk activate latest
   $ emsdk_env.bat

5.Emscripten工具链已安装！

$ emsdk update

6.现在我们已经完成了Emscripten[编译器](https://so.csdn.net/so/search?q=编译器&spm=1001.2101.3001.7020)的安装，让我们继续使用它。

7..我们可以通过使用以下命令来检查emcc的安装是否成功：

 $ emcc --version

使用Emscripten生成asm.js

我们将使用Emscripten将C/C++程序移植到asm.js或WebAssembly二进制格式，然后在JavaScript引擎中运行它们。



.在浏览器中运行推理
在浏览器中运行推理（Wasm 调用示例）

--------------------

在本项目中，浏览器通过 **Emscripten 编译后的 Wasm 模块**（`Module`）进行推理。前端调用流程如下：

1. **加载 Wasm 模块**：Emscripten 会在网页加载时初始化 `Module`，并提供导出的接口函数（如 `createImage`、`setImageData`、`predict`、`freeImage`）。

2. **准备输入数据**：将图像缩放到固定大小（例如 `200x200`），读取像素值并归一化到 `[0, 1]`。

3. **调用推理接口**：通过 `Module.predict` 将输入数据传入模型，获取预测结果。

4. **释放内存**：使用 `Module.freeImage` 释放临时分配的内存。

```javascript
<script>
    var Module = {
        onRuntimeInitialized: function () {
            console.log("Wasm 模块已加载，可以调用推理函数");
        }
    };

    async function runInference(imageData) {
        // 1. 创建输入数据容器 (200x200x3)
        var data = Module['createImage'](200, 200, 3);

        // 2. 填充像素数据 (归一化到 [0, 1])
        for (var i = 0; i < 200; i++) {
            for (var j = 0; j < 200; j++) {
                var idx = (i * 200 + j) * 4;
                for (var k = 0; k < 3; k++) {
                    Module['setImageData'](data, i, j, k, imageData.data[idx + k] / 255.0);
                }
            }
        }

        // 3. 调用推理函数
        var result = Module['predict'](data);

        // 4. 解析输出结果
        if (result == 0) {
            console.log("预测结果: 猫");
        } else {
            console.log("预测结果: 狗");
        }

        // 5. 释放内存
        Module['freeImage'](data);
    }
</script>

```











1. 项目概述

-------

WasmCrypt 提供一套从模型到受保护 Wasm 的完整流程：包含模型示例、模型转换模块、混淆算法实现与前端运行示例。目标是把初始模型生成 Wasm/JS，随后应用混淆和保护策略，最后在浏览器端通过静态服务器运行验证。

* * *

2. 项目文件夹（只列文件夹层级）

-----------------

* `confuse/`  
  混淆算法相关代码与资源（即“混淆”实现）。

* `cpp_sre/`  
  C++ 算子实现集合 —— 包含约 140+ 个算子，是模型转换后在本地/wasm 中依赖的核心算子实现。

* `model/`  
  示例与案例模型目录：包含原始模型（各种格式）、模型转换模块产生的 Wasm 与 JS 文件（示例）。

* `web/`  
  前端示例目录：每个示例模型在 `web` 下有对应的文件夹（将 `model/` 中生成的 wasm/js 复制到对应 `web/<model-name>/` 下即可通过静态服务器运行）。

* `ui/` 或 `gui/`（若存在）  
  项目的图形界面（Python UI）或启动脚本，用于交互式选择并输出混淆后的 wasm。

* `dist/` 或 `release/`（若存在）  
  打包后的可执行文件（`.exe`）或其他二进制发布物。

* `scripts/`（若存在）  
  包含编译、打包或部署相关脚本（例如用于调用 Emscripten 的编译脚本）。

* 其他配置文件：`requirements.txt`、`README.md`（本文件）、若干说明与示例脚本。

> 注：仓库里具体文件名可能略有不同，上述为文件夹层级说明，便于快速定位功能模块。

* * *

3. 环境与版本要求

----------

* **Python**：推荐 **3.8**（已在此版本下测试）。

* **依赖**：使用项目根目录下的 `requirements.txt` 安装 Python 依赖。

* **Emscripten (EMSDK)**：用于将 C/C++ 编译为 Wasm。需要在系统中安装并激活 EMSDK（确保 `emcc` 可用）。

* **Node.js / npm**（可选）：用于本地静态服务（`http-server`）或前端打包工具。

* **C/C++ 编译器**：若需要本地编译或构建 `cpp_sre` 的本地组件。

* * *

4. 安装依赖（命令示例）

-------------

在项目根目录下运行：

1. 创建并激活虚拟环境（推荐）
   
   * Linux / macOS:
     `python3.8 -m venv venv source venv/bin/activate`
   
   * Windows (PowerShell):
     `python -m venv venv.\venv\Scripts\Activate.ps1`

2. 使用 `requirements.txt` 安装 Python 依赖：
   `pip install -r requirements.txt`

3. （可选）安装本地静态服务器（推荐用于前端快速调试）：
   `npm install -g http-server`

4. 安装并激活 Emscripten（若需重新编译 C++ -> Wasm）：  
   请按照 Emscripten 官方说明安装 EMSDK，并确保激活后 `emcc` 命令可用。项目中有用于调用 Emscripten 的脚本（见 `scripts/` 或项目根目录），在运行这些脚本前请先启用 emsdk 环境。

* * *

5. 编译（概要）

---------

项目使用 Emscripten 提供的脚本将 C++（`cpp_sre` 等）编译为 Wasm。通常步骤为：

1. 确保 EMSDK 已安装并激活（`source ./emsdk_env.sh` / Windows 对应命令）。

2. 运行仓库中的 Emscripten 编译脚本（示例）：
   `# 假设项目提供了脚本 scripts/build_wasm.sh 或类似脚本 bash scripts/build_wasm.sh`
   
   > 如果项目没有命名一致的脚本，请查看 `scripts/` 或 README 中的具体编译脚本文件并执行。编译脚本会调用 `emcc` 并输出 `.wasm` 与配套 `.js`。

* * *

6. 运行流程（基础 / 最小步骤）

------------------

下面给出常见的使用流程，按你提供的描述抽象为三步（安装环境 → 启动界面/可执行 → 输出混淆后的 Wasm → 在 web 下运行）：

1. **安装环境**  
   创建虚拟环境并运行 `pip install -r requirements.txt`，安装并激活 EMSDK（如果需要重新编译）。

2. **运行 UI 或 可执行文件（exe）**
   
   * 使用 Python UI（若项目有 `ui.py` 或等效入口）:
     `python ui.py`
   
   * 或者使用打包好的可执行文件（Windows `.exe`）: 双击或从命令行启动 `dist/your_app.exe`。

3. **在界面中选择“输出混洗后的 Wasm”**  
   在 UI/EXE 中选择对应按钮或选项（你描述的流程是：先按一个键输出混洗后的 wasm），程序会把混淆处理后的 Wasm/JS 输出到某个目录（例如 `model/` 下或 `output/` 下，具体以项目实际输出路径为准）。

4. **部署到前端并运行**
   
   * 将生成的 `.wasm` 与配套 `.js`（在 `model/` 中或输出目录）**复制到** `web/<对应模型文件夹>/` 下。
     在该 `web/<model-folder>/` 目录启动静态服务器：

```bash
 # 使用 npm http-server（推荐）
http-server ./web/<model-folder> -p 8080

# 或者使用 Python 内建服务器（简单）
cd web/<model-folder>
python -m http.server 8080

```

* 打开浏览器访问 `http://localhost:8080` 并验证模型在浏览器中的运行与推理结果。

* * *

7. 示例（快速演示）

-----------

假设你有模型名为 `mnist_example`，且混淆输出到 `model/output/mnist_example/`：

```python
# 激活虚拟环境并安装依赖
python3.8 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 启动 UI（或运行 exe）
python ui.py
# 在 UI 中点击 “Output Obfuscated Wasm” （或相应按钮）

# 将生成文件复制到 web 目录下（手动或脚本）
cp -r model/output/mnist_example/* web/mnist_example/

# 启动静态服务器并访问
http-server ./web/mnist_example -p 8080
# 访问 http://localhost:8080

```

在浏览器中运行推理在浏览器中运行推理（Wasm 调用示例）

* * *

8.在本项目中，浏览器通过 **Emscripten 编译后的 Wasm 模块**（`Module`）进行推理。前端调用流程如下：

1. **加载 Wasm 模块**：Emscripten 会在网页加载时初始化 `Module`，并提供导出的接口函数（如 `createImage`、`setImageData`、`predict`、`freeImage`）。

2. **准备输入数据**：将图像缩放到固定大小（例如 `200x200`），读取像素值并归一化到 `[0, 1]`。

3. **调用推理接口**：通过 `Module.predict` 将输入数据传入模型，获取预测结果。

4. **释放内存**：使用 `Module.freeImage` 释放临时分配的内存。

    <script>
        var Module = {
            onRuntimeInitialized: function () {
                console.log("Wasm 模块已加载，可以调用推理函数");
            }
        };
    
        async function runInference(imageData) {
            // 1. 创建输入数据容器 (200x200x3)
            var data = Module['createImage'](200, 200, 3);
    
            // 2. 填充像素数据 (归一化到 [0, 1])
            for (var i = 0; i < 200; i++) {
                for (var j = 0; j < 200; j++) {
                    var idx = (i * 200 + j) * 4;
                    for (var k = 0; k < 3; k++) {
                        Module['setImageData'](data, i, j, k, imageData.data[idx + k] / 255.0);
                    }
                }
            }
    
            // 3. 调用推理函数
            var result = Module['predict'](data);
    
            // 4. 解析输出结果
            if (result == 0) {
                console.log("预测结果: 猫");
            } else {
                console.log("预测结果: 狗");
            }
    
            // 5. 释放内存
            Module['freeImage'](data);
        }
    </script>



* * *

9.常见问题与排查建议

------------

* **找不到 `emcc` / 编译出错**：确认 EMSDK 已正确安装并激活；在终端执行 `emcc -v` 应能显示版本信息。

* **UI 启动失败**：确认已安装 `requirements.txt` 中列出的所有依赖，并使用 Python 3.8。查看终端/日志中的错误堆栈以定位缺失模块。

* **浏览器加载 `.wasm` 报错 CORS / 404**：请确保通过静态服务器提供文件，而不是直接用 `file://` 打开页面；检查 `web/<model-folder>/` 下是否有正确的 `.wasm` 与 `.js` 文件名匹配。

* **混淆后模型行为异常**：确认混淆流程完成且无中间步骤失败；检查混淆日志（如果有）和原始/混淆前后的输出比较，定位差异。

* * *

10.其他说明

-------

* `confuse/` 实现为项目的混淆算法核心，`cpp_sre/` 则包含算子实现（约 140+ 个）。若需扩展算子或修改混淆策略，请在相应目录修改并重新运行编译脚本与混淆流程。

* `model/` 中同时包含原始示例模型与模型转换后生成的 Wasm/JS，可作为部署前的参考。

* 将生成文件复制到 `web` 对应的模型文件夹下后，通过任意静态服务器（如 `http-server` 或 `python -m http.server`）启动即可进行网页端验证。

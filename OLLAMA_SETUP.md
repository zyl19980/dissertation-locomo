# Ollama 本地模型集成说明

本项目已添加对 Ollama 本地大模型的支持,可以使用 Qwen3-8B 等本地模型替代 OpenAI API。

## 1. 安装 Ollama

### Windows 系统:
1. 访问 [Ollama 官网](https://ollama.com/download)
2. 下载 Windows 安装包
3. 运行安装程序

### Linux/Mac 系统:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 2. 下载 Qwen3-8B 模型

安装 Ollama 后,运行以下命令下载模型:

```bash
ollama pull qwen3-8b
```

其他可用的 Qwen 模型:
- `ollama pull qwen2.5:3b` (更小,速度更快)
- `ollama pull qwen2.5:7b` (平衡性能)
- `ollama pull qwen3-8b` (推荐,性能好)

## 3. 启动 Ollama 服务

Ollama 默认会在后台自动运行,默认端口为 `11434`。

如果需要手动启动:
```bash
ollama serve
```

验证服务是否运行:
```bash
curl http://localhost:11434/api/tags
```

## 4. 测试 Ollama 连接

运行测试脚本:
```bash
cd e:\1-ntu\dissertation\论文代码\locomo\locomo
python test_ollama.py
```

## 5. 在代码中使用 Ollama

### 方法 1: 直接调用 run_ollama 函数

```python
from global_methods import run_ollama

# 调用 Ollama 模型
response = run_ollama(
    query="你的问题",
    num_tokens_request=1000,
    model='qwen3-8b',  # 或其他模型名称
    temperature=0.7,
    wait_time=2
)
print(response)
```

### 方法 2: 在 gpt_utils.py 中使用

在运行评估脚本时,指定模型名称为 Ollama 支持的模型:

```bash
python task_eval/xxx.py --model qwen3-8b --batch_size 1
```

或者使用通用的 'ollama' 名称(默认使用 qwen3-8b):
```bash
python task_eval/xxx.py --model ollama --batch_size 1
```

## 6. 支持的模型列表

代码中已添加以下模型的支持:

| 模型名称 | 上下文长度 | 说明 |
|---------|-----------|------|
| `ollama` | 8192 | 通用名称,默认使用 qwen3-8b |
| `qwen2.5:3b` | 32768 | Qwen2.5-3B,最快 |
| `qwen2.5:7b` | 32768 | Qwen2.5-7B,平衡 |
| `qwen3-8b` | 32768 | Qwen3-8B,推荐 |

## 7. 代码修改说明

### 修改的文件:

1. **global_methods.py**
   - 添加了 `run_ollama()` 函数
   - 使用 Ollama HTTP API 进行调用
   - 支持重试机制和错误处理

2. **task_eval/gpt_utils.py**
   - 导入 `run_ollama` 函数
   - 在 `MAX_LENGTH` 字典中添加 Ollama 模型配置
   - 在 `get_gpt_answers()` 函数中添加 Ollama 调用逻辑
   - 自动识别模型名称并选择相应的调用方式

## 8. 注意事项

1. **首次使用速度较慢**: 模型首次加载需要时间,之后会保持在内存中
2. **内存要求**: Qwen3-8B 需要至少 8GB 可用内存
3. **默认端口**: Ollama 默认使用 `localhost:11434`,如需修改可在 `run_ollama()` 函数中指定 `host` 参数
4. **batch_size 限制**: 建议使用 `batch_size=1` 以获得更好的稳定性

## 9. 常见问题

### Q: 连接失败
**A**: 检查 Ollama 服务是否运行:
```bash
ollama list  # 查看已下载的模型
ollama serve # 启动服务
```

### Q: 模型响应慢
**A**:
- 首次加载模型较慢,请耐心等待
- 考虑使用更小的模型 (如 qwen2.5:3b)
- 确保有足够的 RAM 和 CPU 资源

### Q: 如何切换其他模型
**A**: 在调用时修改 `model` 参数:
```python
response = run_ollama(query="...", model='qwen2.5:7b')
```

## 10. 性能对比

与 OpenAI API 相比:
- ✅ **优势**: 完全本地运行,无需 API 密钥,无调用费用,数据隐私
- ⚠️ **劣势**: 需要本地算力,速度取决于硬件配置

建议用途:
- 开发测试阶段使用本地模型
- 生产环境根据需求选择 API 或本地模型

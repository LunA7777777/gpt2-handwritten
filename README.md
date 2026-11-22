# GPT-2 手写前向

用纯 Python `list` 和显式 `for` 循环实现 GPT-2 (124M) 的完整前向，
不用批量张量运算，再与 HuggingFace 官方实现逐元素对拍。

这不是一个能用的 GPT-2 库，是一份**把每一步数值摊开来看**的教学实现。

## 结果

在本机（RTX 3070 Ti Laptop，Python 3.12，PyTorch 2.10.0+cu128，
Transformers 5.22.2）实际运行 `对拍.py` 得到的输出：

```
A. 分词器
  'The capital of France is'             一致
  'Hello, my name is'                    一致
  'In 1969, humans first walked on the'  一致
  'def quicksort(arr):'                  一致
  'a'                                    一致

B. logits 逐元素（每句最后一个位置）
  输入                                      最大绝对误差     top-1
  'The capital of France is'                 9.095e-05      一致
  'Hello, my name is'                        6.053e-05      一致
  'In 1969, humans first walked on the'      5.987e-05      一致
  'def quicksort(arr):'                      6.726e-05      一致
  'a'                                        1.085e-04      一致

D. 参数量
  我按 safetensors 头统计 = 124439808
  torch model.parameters() = 124439808
  差值 = 0
  lm_head 与 wte 共享同一块内存 = True
```

`C. 逐层中间量` 打印每一层 hidden state 与官方实现的最大绝对误差，
用来定位误差从哪一层开始放大——12 层累计到 1e-4 量级属正常 float32 舍入，
1e-1 以上才是算错了。

## 实现里值得看的三个点

- **「12 个头」不是张量维度**：就是 768 个数切成 12 段、每段 64 个，
  第 h 个头只碰 `q[t][h*64 : (h+1)*64]`。
- **「因果掩码」不是矩阵**：内层循环上界写 `range(t + 1)`，第 t 个词
  只和第 0..t 个词打分，右边根本没进循环。
- **「权重共享」**：出口又把入口查过的 `wte` 用了一遍。

`最简版.py --消融` 会关掉部件重跑，观察模型怎么坏。

## 运行

需要 GPT-2 权重（`model.safetensors`、`vocab.json`、`merges.txt`、
`config.json`），放在与脚本同级的 `gpt2权重/` 目录下；可从
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2) 获取。

```powershell
python -m pip install numpy regex
python 最简版.py --层 1     # 单层，约 3 秒
python 最简版.py            # 完整前向，约 20 秒
python 最简版.py --消融     # 消融实验，约 1 分钟

python -m pip install torch transformers
python 对拍.py              # 对拍，需要上述权重
```

## 文件

| 文件 | 内容 |
|---|---|
| `最简版.py` | GPT-2 前向，三种数据结构：`float`、`list[float]`、`list[list[float]]` |
| `对拍.py` | 与 HuggingFace `GPT2LMHeadModel` 逐元素比较 |

原始版本在私有学习仓库中维护；这里发布的是可独立运行的快照。

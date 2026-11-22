# -*- coding: utf-8 -*-
"""把 最简版.py 和 HuggingFace 官方实现逐元素对拍。

**不用读这个文件。** 它只回答一个问题：最简版.py 算的到底是不是 GPT-2。
「输出看起来通顺」不算验证 —— 这是唯一的正确性依据。

需要 torch + transformers（最简版.py 本身不依赖它们）。
"""
import os
import sys

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import 最简版 as G  # noqa: E402

CASES = ["The capital of France is", "Hello, my name is",
         "In 1969, humans first walked on the", "def quicksort(arr):", "a"]

W = G.Weights(os.path.join(G.WDIR, "model.safetensors"))
TOK = G.Tokenizer(G.WDIR)
参考分词 = GPT2TokenizerFast.from_pretrained(G.WDIR)
模型 = GPT2LMHeadModel.from_pretrained(G.WDIR).eval()

print("\nA. 分词器 ───────────────────────────────────────────────")
for s in CASES:
    我的, 官方 = TOK.encode(s), 参考分词.encode(s)
    print("  %-38r %s" % (s, "一致 %s" % 我的 if 我的 == 官方
                          else "不一致！我的=%s 官方=%s" % (我的, 官方)))

print("\nB. logits 逐元素（每句最后一个位置）─────────────────────")
print("  %-38s %13s %10s" % ("输入", "最大绝对误差", "top-1"))
for s in CASES:
    ids = TOK.encode(s)
    我的 = G.forward(ids, W)
    with torch.no_grad():
        官方 = 模型(torch.tensor([ids])).logits[0, -1].numpy()
    误差 = max([abs(我的[j] - float(官方[j])) for j in range(len(我的))])
    一致 = 我的.index(max(我的)) == int(官方.argmax())
    print("  %-38r %13.3e %10s" % (s, 误差, "一致" if 一致 else "不一致！"))

print("\nC. 逐层中间量（定位误差从哪层开始放大）──────────────────")
ids = TOK.encode(CASES[0])
S = {}
G.forward(ids, W, S)
with torch.no_grad():
    hs = 模型(torch.tensor([ids]), output_hidden_states=True).hidden_states
# HF 的 hidden_states 是每个 block **之前** append 的，共 13 项 =
# [嵌入后, 过完 block0, ..., 过完 block10, 过完 block11 且过完 ln_f]。
# 所以只能比到 x11，最后一项那边多过了一次 ln_f，没有可比项。
for i in range(G.N_LAYER):
    我的, 官方 = S["x%02d" % i], hs[i][0].numpy()
    print("  x%02d  %.3e" % (i, max([abs(我的[t][c] - float(官方[t][c]))
                                     for t in range(len(我的))
                                     for c in range(G.N_EMBD)])))

print("\nD. 参数量 ───────────────────────────────────────────────")
# h.N.attn.bias 是常量因果掩码，不是参数，要排除。不能写成
# endswith("attn.bias") —— h.N.attn.c_attn.bias 也以它结尾，会被一起
# 误杀 12×2304=27648 个。这个 bug 当年就是被这一节抓出来的。
def 是掩码(k):
    p = k.split(".")
    return len(p) >= 2 and p[-2] == "attn" and p[-1] == "bias"


我的n = sum([int(np.prod(W.header[k]["shape"]))
             for k in W.header if not 是掩码(k)])
官方n = sum([p.numel() for p in 模型.parameters()])
print("  我按 safetensors 头统计 = %d" % 我的n)
print("  torch model.parameters() = %d" % 官方n)
print("  差值 = %d" % (官方n - 我的n))
print("  lm_head 与 wte 共享同一块内存 = %s"
      % (模型.lm_head.weight.data_ptr()
         == 模型.transformer.wte.weight.data_ptr()))
print()
print("  logits 本身在 -130 ~ -100，float32 有效位约 7 位，12 层累计到")
print("  1e-3 量级属于正常浮点误差，1e-1 以上才说明算错了。最简版用")
print("  Python 的 float（float64），官方用 float32 —— 误差里有一部分")
print("  其实是官方那边的舍入，这边反而更准。")

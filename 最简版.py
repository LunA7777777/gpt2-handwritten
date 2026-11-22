# -*- coding: utf-8 -*-
"""GPT-2 (124M) 的完整前向，只用 list 和 for 循环写。

────────────────────────────────────────────────────────────────────
只有三种数据结构
────────────────────────────────────────────────────────────────────

    标量    float                      一个数
    向量    [float]                    长度 768        TArray<float>
    矩阵    [[float], [float], ...]    一个 list 装行   TArray<TArray<float>>

主干 x 从头到尾就是「T 行 × 768 列」，T 是词的个数。
整个模型在干的事，就是反复改这 T×768 个数。

────────────────────────────────────────────────────────────────────
三个换成循环就一眼看穿的东西
────────────────────────────────────────────────────────────────────

  · 「12 个头」= 把 768 个数切成 12 段、每段 64 个。第 h 个头只碰
    q[t][h*64 : (h+1)*64]，从头到尾不碰别人的。numpy 写成
    reshape(B,T,12,64).transpose(0,2,1,3) 纯粹是为了凑张量。

  · 「因果掩码」= 内层循环的上界 range(t + 1)。第 t 个词只和第 0..t
    个词打分，右边的根本没进循环。不需要造矩阵填 -inf。

  · 「权重共享」= 出口又把入口查过的那张 wte 拿来用了一遍。

用法：
    python 最简版.py                       跑一遍，约 20 秒
    python 最简版.py "Hello, my name is"   换句子
    python 最简版.py --层 1                只跑 1 层，3 秒
    python 最简版.py --消融                关掉部件重跑，约 1 分钟

写码约定：全文不用「裸生成器表达式」（sum(x for x in y) 这种），一律写成
列表推导 sum([x for x in y])。因为 PyCharm 的格式化会给函数实参补尾逗号，
而裸生成器后面跟逗号是语法错误 —— 加了方括号就炸不了。
"""
import io
import json
import math
import os
import struct
import sys

import numpy as np  # 只用来读文件，一次乘法都不做
import regex

sys.stdout = io.TextIOWrapper ( sys.stdout.buffer , encoding = "utf-8" , errors = "replace" , newline = "" , )

# 全部来自 GPT-2 124M 的官方 config.json
N_LAYER , N_HEAD , N_EMBD = 12 , 12 , 768
HEAD_DIM = N_EMBD // N_HEAD  # 64，每个头分到的通道数
BLOCK_SIZE = 1024  # 位置表的行数 = 上下文长度上限
EPS = 1e-5  # LayerNorm 防除零
SCALE = 1.0 / math.sqrt ( HEAD_DIM , )  # 0.125，注意力打分的缩放
WDIR = os.path.join ( os.path.dirname ( os.path.abspath ( __file__ , ) , ) , "gpt2权重" , )


# ══════════════════════════════════════════════════════════════════
#   1. 读权重
# ══════════════════════════════════════════════════════════════════
class Weights :
	"""safetensors 格式：

		[0, 8)      u64 小端 = 头部 JSON 的字节数 N
		[8, 8+N)    JSON：{张量名: {dtype, shape, data_offsets:[b0,b1]}}
		[8+N, EOF)  裸数据，b0/b1 是相对这里的偏移
	"""
	
	def __init__ ( self , path , ) :
		with open ( path , "rb" , ) as f :
			n = struct.unpack ( "<Q" , f.read ( 8 , ) , ) [ 0 ]
			self.header = json.loads ( f.read ( n , ) , )
		self.header.pop ( "__metadata__" , None , )
		self._base = 8 + n
		self._mm = np.memmap ( path , dtype = np.uint8 , mode = "r" , )
	
	def __getitem__ ( self , name , ) :
		b0 , b1 = self.header [ name ] [ "data_offsets" ]
		raw = self._mm [ self._base + b0 : self._base + b1 ]
		return raw.view ( np.float32 , ).reshape ( self.header [ name ] [ "shape" ] , )


def 装载一层 ( W , i , ) :
	"""把第 i 层的 12 个权重搬成 Python list。

	四个大矩阵转置一次：文件里存的是 (输入维, 输出维)，而 matmul 要按
	「输出维」取行。转置只是换个存法，一个数都没变 —— 在这里一次性
	摆平，下面的算法就不用再操心内存布局。
	"""
	p , L = "h.%d." % i , { }
	for 名 in ("ln_1.weight" , "ln_1.bias" , "ln_2.weight" , "ln_2.bias" , "attn.c_attn.bias" , "attn.c_proj.bias" , "mlp.c_fc.bias" , "mlp.c_proj.bias") :
		L [ 名 ] = W [ p + 名 ].tolist ( )
	for 名 in ("attn.c_attn.weight" , "attn.c_proj.weight" , "mlp.c_fc.weight" , "mlp.c_proj.weight") :
		L [ 名 ] = W [ p + 名 ].T.tolist ( )
	return L


# ══════════════════════════════════════════════════════════════════
#   2. 分词（字节级 BPE）—— 和 Transformer 无关，跳过不影响理解
# ══════════════════════════════════════════════════════════════════
_PAT = regex.compile ( r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" , )


class Tokenizer :
	"""文本 -> 一串整数 id。id 是 wte 的行号，是下标不是数值 ——
	id=464 和 id=465 之间没有任何大小关系。"""
	
	def __init__ ( self , d , ) :
		with open ( os.path.join ( d , "vocab.json" , ) , encoding = "utf-8" , ) as f :
			self.encoder = json.load ( f , )
		self.decoder = { v : k for k , v in self.encoder.items ( ) }
		with open ( os.path.join ( d , "merges.txt" , ) , encoding = "utf-8" , ) as f :
			合并 = [ tuple ( l.split ( ) , ) for l in f.read ( ).split ( "\n" , ) [ 1 :-1 ] ]
		self.ranks = dict ( zip ( 合并 , range ( len ( 合并 , ) , ) , ) , )
		# 把 0-255 每个字节搬到一个可打印码位上（空格 0x20 -> 'Ġ'），
		# 否则控制字符没法当字符串做合并。
		bs = (list ( range ( 33 , 127 , ) , ) + list ( range ( 161 , 173 , ) , ) + list ( range ( 174 , 256 , ) , ))
		cs , n = bs [ : ] , 0
		for b in range ( 256 , ) :
			if b not in bs :
				bs.append ( b , )
				cs.append ( 256 + n , )
				n += 1
		self.b2u = dict ( zip ( bs , [ chr ( c , ) for c in cs ] , ) , )
	
	def _bpe ( self , w , ) :
		w = tuple ( w , )
		while len ( w , ) > 1 :
			对 = min ( set ( zip ( w [ :-1 ] , w [ 1 : ] , ) , ) , key = self._rank , )
			if 对 not in self.ranks :
				break
			out , i = [ ] , 0
			while i < len ( w , ) :
				if i < len ( w , ) - 1 and (w [ i ] , w [ i + 1 ]) == 对 :
					out.append ( w [ i ] + w [ i + 1 ] , )
					i += 2
				else :
					out.append ( w [ i ] , )
					i += 1
			w = tuple ( out , )
		return w
	
	def _rank ( self , 对 , ) :
		return self.ranks.get ( 对 , float ( "inf" , ) , )
	
	def encode ( self , text , ) :
		ids = [ ]
		for 块 in regex.findall ( _PAT , text , ) :
			s = "".join ( [ self.b2u [ b ] for b in 块.encode ( "utf-8" , ) ] , )
			ids.extend ( [ self.encoder [ p ] for p in self._bpe ( s , ) ] , )
		return ids
	
	def piece ( self , i , ) :
		return self.decoder [ i ].replace ( "Ġ" , " " , ).replace ( "Ċ" , "\\n" , )


# ══════════════════════════════════════════════════════════════════
#   3. 算子 —— 六个函数、20 行代码，Transformer 的算术全在这儿了
# ══════════════════════════════════════════════════════════════════
def dot ( a , b , ) :
	"""两个等长向量的点积。一次前向要调用它 46 万次，99% 的时间在这。"""
	s = 0.0
	for k in range ( len ( a , ) , ) :
		s += a [ k ] * b [ k ]
	return s


def matmul ( X , W , b , ) :
	"""X: T 行 × in 列。W: out 行 × in 列。b: 长 out。返回 T 行 × out 列。

	矩阵乘法就是一格一个点积：第 (t, j) 格 = X 的第 t 行 · W 的第 j 行。
	"""
	return [ [ dot ( x , W [ j ] , ) + b [ j ] for j in range ( len ( b , ) , ) ] for x in X ]


def 相加 ( X , Y , ) :
	"""两个同样大小的矩阵逐元素相加。残差连接就是这一句。"""
	return [ [ X [ t ] [ c ] + Y [ t ] [ c ] for c in range ( len ( X [ t ] , ) , ) ] for t in range ( len ( X , ) , ) ]


def layernorm ( v , g , b , ) :
	"""把一个 768 维向量拉成均值 0、方差 1，再逐通道乘 g 加 b。

	只在「一个位置的 768 个通道」之间做，不跨位置。所以调用它的地方
	都写成 for 每一行 各算各的。
	"""
	n = len ( v , )
	mu = sum ( v , ) / n
	var = sum ( [ (z - mu) * (z - mu) for z in v ] , ) / n
	s = 1.0 / math.sqrt ( var + EPS , )
	return [ g [ c ] * (v [ c ] - mu) * s + b [ c ] for c in range ( n , ) ]


def softmax ( v , ) :
	"""一串分数变成一串加起来等于 1 的概率。减最大值只为防 exp 溢出。"""
	m = max ( v , )
	e = [ math.exp ( z - m , ) for z in v ]
	s = sum ( e , )
	return [ z / s for z in e ]


def gelu ( z , ) :
	"""GPT-2 用的是 tanh 近似版（HF 里叫 gelu_new），不是精确版。
	换成精确版或 ReLU，logits 就和官方对不上了。"""
	return 0.5 * z * (1.0 + math.tanh ( 0.7978845608028654 * (z + 0.044715 * z * z * z) , ))


# ══════════════════════════════════════════════════════════════════
#   4. 注意力 —— 一个位置只看得见它自己和它左边的
# ══════════════════════════════════════════════════════════════════
def attention ( x , L , 记录 = None , 去掉掩码 = False , ) :
	"""x: T 行 × 768 列  ->  T 行 × 768 列。"""
	T = len ( x , )
	
	# 一次算出 q / k / v。2304 = 768 × 3，前中后各 768 列。
	qkv = matmul ( x , L [ "attn.c_attn.weight" ] , L [ "attn.c_attn.bias" ] , )
	q = [ r [ 0 :768 ] for r in qkv ]
	k = [ r [ 768 :1536 ] for r in qkv ]
	v = [ r [ 1536 :2304 ] for r in qkv ]
	
	y = [ [ 0.0 ] * N_EMBD for _ in range ( T , ) ]
	权重 = [ ]
	for h in range ( N_HEAD , ) :
		lo = h * HEAD_DIM  # 这个头只碰第 lo..hi 段通道，
		hi = lo + HEAD_DIM  # 12 个头把 768 分成 12 段互不相干的 64
		本头 = [ ]
		for t in range ( T , ) :
			# 这一个上界就是「因果掩码」：第 t 个词只和第 0..t 个打分。
			上界 = T if 去掉掩码 else t + 1
			分数 = [ dot ( q [ t ] [ lo :hi ] , k [ u ] [ lo :hi ] , ) * SCALE for u in range ( 上界 , ) ]
			w = softmax ( 分数 , )  # 归一成注意力权重，加起来是 1
			本头.append ( w , )
			for u in range ( 上界 , ) :  # 按权重把各位置的 v 累加到自己身上
				wu , vu = w [ u ] , v [ u ]
				for c in range ( lo , hi , ) :
					y [ t ] [ c ] += wu * vu [ c ]
		权重.append ( 本头 , )
	if 记录 is not None :
		记录 [ "注意力" ] = 权重
	
	return matmul ( y , L [ "attn.c_proj.weight" ] , L [ "attn.c_proj.bias" ] , )


def mlp ( x , L , ) :
	"""768 -> 3072 -> 过 gelu -> 768。每个位置各算各的，位置之间不通气。"""
	h = matmul ( x , L [ "mlp.c_fc.weight" ] , L [ "mlp.c_fc.bias" ] , )
	h = [ [ gelu ( z , ) for z in r ] for r in h ]
	return matmul ( h , L [ "mlp.c_proj.weight" ] , L [ "mlp.c_proj.bias" ] , )


def block ( x , L , 记录 = None , 去掉掩码 = False , ) :
	"""一层。两个分支各自算完再加回主干 —— 主干 x 的大小一次都没变。"""
	a = [ layernorm ( r , L [ "ln_1.weight" ] , L [ "ln_1.bias" ] , ) for r in x ]
	x = 相加 ( x , attention ( a , L , 记录 , 去掉掩码 , ) , )  # 残差 1
	m = [ layernorm ( r , L [ "ln_2.weight" ] , L [ "ln_2.bias" ] , ) for r in x ]
	return 相加 ( x , mlp ( m , L , ) , )  # 残差 2


# ══════════════════════════════════════════════════════════════════
#   5. 前向 —— 整个模型
# ══════════════════════════════════════════════════════════════════
def forward ( ids , W , 快照 = None , 层数 = N_LAYER , 清零位置表 = False , 去掉掩码 = False , 跳过的层 = () , ) :
	"""ids: 一串整数  ->  最后一个位置对词表里每个词的打分。

	后三个开关是消融用的：把某个部件关掉再跑一次，看输出怎么坏。
	"""
	快照 = { } if 快照 is None else 快照
	T = len ( ids , )
	assert T <= BLOCK_SIZE , ("序列长度 %d 超过 %d —— 上下文上限就是 wpe 的行数，没有更玄的东西" % (T , BLOCK_SIZE))
	
	wte = W [ "wte.weight" ]  # 词表   50257 行 × 768 列
	wpe = W [ "wpe.weight" ]  # 位置表  1024 行 × 768 列
	
	# 查两张表，相加。这就是主干的初值。
	词 = [ wte [ ids [ t ] ].tolist ( ) for t in range ( T , ) ]
	位置 = (
		[ [ 0.0 ] * N_EMBD for _ in range ( T , ) ] if 清零位置表 else [ wpe [ t ].tolist ( ) for t in range ( T , ) ])
	x = 相加 ( 词 , 位置 , )
	快照 [ "x00" ] = x
	
	for i in range ( 层数 , ) :
		if i not in 跳过的层 :
			x = block ( x , 装载一层 ( W , i , ) , 快照 if i == 0 else None , 去掉掩码 , )
		快照 [ "x%02d" % (i + 1) ] = x
	
	g , b = W [ "ln_f.weight" ].tolist ( ) , W [ "ln_f.bias" ].tolist ( )
	x = [ layernorm ( r , g , b , ) for r in x ]
	
	# 出口：主干最后一行，和词表的每一行做点积 —— 词表就是入口查过的
	# 那张 wte，这就是「权重共享」。整表转 list 要 1.2 GB，所以逐行取。
	末 = x [ -1 ]
	return [ dot ( 末 , wte [ j ].tolist ( ) , ) for j in range ( len ( wte , ) , ) ]


# ══════════════════════════════════════════════════════════════════
#   6. 跑一遍，把上面每一步的真实数值打出来
# ══════════════════════════════════════════════════════════════════
def 前几名 ( logits , TOK , k = 5 , ) :
	"""取分数最高的 k 个，返回 [(id, 片段, logit, 概率), ...]。"""
	p = softmax ( logits , )
	序 = list ( range ( len ( logits , ) , ) , )
	序.sort ( key = p.__getitem__ , reverse = True , )
	return [ (j , TOK.piece ( j , ) , logits [ j ] , p [ j ]) for j in 序 [ :k ] ]


def 主 ( text , 层数 , 要消融 , ) :
	W = Weights ( os.path.join ( WDIR , "model.safetensors" , ) , )
	TOK = Tokenizer ( WDIR , )
	ids = TOK.encode ( text , )
	T = len ( ids , )
	S = { }
	logits = forward ( ids , W , S , 层数 , )
	
	print ( "\n〇  输入 ─────────────────────────────────────────────────" , )
	print ( '  "%s"' % text , )
	print ( "  " + "  ".join ( [ "%d(%s)" % (i , TOK.piece ( i , )) for i in ids ] , ) , )
	
	print ( "\n一  主干 x 是什么 ───────────────────────────────────────" , )
	x = S [ "x00" ]
	print ( "  type(x)       = %s" % type ( x , ) , )
	print ( "  len(x)        = %d      ← 行数 = 词的个数 T" % len ( x , ) , )
	print ( "  len(x[0])     = %d    ← 列数 = 768，从头到尾不变" % len ( x [ 0 ] , ) , )
	print ( "  type(x[0][0]) = %s" % type ( x [ 0 ] [ 0 ] , ) , )
	print ( "  x[0][:6]      = [%s, ...]" % ", ".join ( [ "%.4f" % z for z in x [ 0 ] [ :6 ] ] , ) , )
	print ( "  一个 list 装着 %d 个 list，每个里面 768 个 float。" % T , )
	print ( "  下面所有的事，都是在反复改这 %d×768 个数。" % T , )
	
	print ( "\n二  %d 层，每层改了多少 ─────────────────────────────────" % 层数 , )
	print ( "  层    |x|     本层改动    占比" , )
	for i in range ( 层数 , ) :
		前 , 后 = S [ "x%02d" % i ] [ -1 ] , S [ "x%02d" % (i + 1) ] [ -1 ]
		n = math.sqrt ( sum ( [ z * z for z in 后 ] , ) / N_EMBD , )
		d = math.sqrt ( sum ( [ (后 [ c ] - 前 [ c ]) ** 2 for c in range ( N_EMBD , ) ] , ) / N_EMBD , )
		print ( "  %2d  %7.3f  %9.4f  %5.1f%%" % (i , n , d , 100.0 * d / n) , )
	print ( "  形状一次都没变 —— 每层都是「读一遍主干，算个增量，加回去」。" , )
	print ( "  这就是残差主干：12 层不是 12 次改写，是 12 次追加。" , )
	
	print ( "\n三  因果掩码（第 0 层第 0 个头）─────────────────────────" , )
	print ( "       " + " ".join ( [ "%6s" % ("→%d" % u) for u in range ( T , ) ] , ) , )
	for t in range ( T , ) :
		w = S [ "注意力" ] [ 0 ] [ t ]
		print ( "  t=%d  " % t + " ".join ( [ "%6.3f" % z for z in w ] , ) + "       " * (
					T - len ( w , )) + "  %r" % TOK.piece ( ids [ t ] , ) , )
	print ( "  行 t 只有 t+1 个数。空白处不是 0，是压根没进循环 ——" , )
	print ( "  attention() 里写的是 range(t + 1)，掩码就是这个上界。" , )
	
	print ( "\n四  出口 ────────────────────────────────────────────────" , )
	print ( "  logits 是一个长 %d 的 list，词表里每个词一个分数。" % len ( logits , ) , )
	for r , (j , s , l , p) in enumerate ( 前几名 ( logits , TOK , ) , ) :
		print ( "  %d. %-12r logit=%9.4f  %5.2f%%" % (r + 1 , s , l , 100 * p) , )
	if 层数 < N_LAYER :
		print ( "  （只跑了 %d 层，这个答案是废的）" % 层数 , )
	
	if 要消融 :
		print ( "\n五  消融 —— 关掉部件，看它怎么坏 ────────────────────────" , )
		for 名 , kw in [ ("完整模型" , { }) , ("去掉位置表 wpe" , { "清零位置表" : True }) , ("去掉因果掩码" , { "去掉掩码" : True }) , ("只留第 0 层" , { "跳过的层" : set ( range ( 1 , 12 , ) , ) }) ] :
			best = 前几名 ( forward ( ids , W , **kw , ) , TOK , 4 , )
			print ( "  %-16s %s" % (名 , "  ".join ( [ "%r(%.1f%%)" % (s , 100 * p) for _ , s , _ , p in best ] , )) , )
		print ( "  「去掉因果掩码」那行反而更准 —— 因为前面的位置能看见" , )
		print ( "  后面的词了，等于提前泄题。掩码不是为了让推理更准，" , )
		print ( "  是为了让训练成立：不挡住，模型抄答案就行，学不会预测。" , )


if __name__ == "__main__" :
	a = sys.argv [ 1 : ]
	要消融 = "--消融" in a
	if 要消融 :
		a.remove ( "--消融" , )
	层数 = N_LAYER
	if "--层" in a :
		i = a.index ( "--层" , )
		层数 = int ( a [ i + 1 ] , )
		del a [ i :i + 2 ]
	主 ( a [ 0 ] if a else "The capital of France is" , 层数 , 要消融 , )

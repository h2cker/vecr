# vecr-compress

[English](README.md) | 中文

**可审计、确定性的 LLM 上下文压缩库。** 订单 ID、URL、日期、引用编号、代码片段等结构化数据，会被一份你可以查看、扩展、审计的正则白名单在压缩过程中**保证留下来**。每一次固定（pin）和每一次丢弃（drop）都有记录：每次调用都会返回 `retained_matches` 和 `dropped_segments` 两份列表。

[![PyPI version](https://img.shields.io/pypi/v/vecr-compress)](https://pypi.org/project/vecr-compress/)
[![Python versions](https://img.shields.io/pypi/pyversions/vecr-compress)](https://pypi.org/project/vecr-compress/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#可复现的-benchmark)
[![Downloads](https://img.shields.io/pypi/dm/vecr-compress)](https://pypi.org/project/vecr-compress/)

## 为什么需要这个

2026 年 Factory.ai 的一项生产环境研究发现：在所有被测压缩方案里，"工件类信息"（artifact tracking，指 ID、文件路径、错误码）是表现最差的一类——5 分制只拿到 2.19–2.45 分，甚至不如 OpenAI 的原生 compaction（3.43/5.0）。目前没有任何公开发布的库提供**确定性的留存原语（retention primitive）**：主流方案都依赖 LLM 判断或学习得到的打分器，结果就是你的客户 ID、交易金额、合规引用可能被悄悄丢掉。vecr-compress 正好补上这个空缺。它**不追求最高压缩率**——那是 Compresr 的赛道。它提供的是一个**可审计、可扩展、从头到尾讲得清楚**的白名单式压缩器。

## 30 秒示例

```python
from vecr_compress import compress

messages = [
    {"role": "system", "content": "You are a refund analyst."},
    {"role": "user", "content":
        "Hi there! Thanks so much for reaching out today. "
        "Hope you are having a wonderful morning. "
        "We really appreciate you writing in. "
        "Happy to take a look at this for you. "
        "Totally understand how important this is. "
        "Sure thing, let me pull up the record. "
        "Absolutely, this is our top priority. "
        "The refund request references order ORD-99172 placed on 2026-03-15. "
        "The customer email is buyer@example.com. "
        "The total charge was $1,499.00 on card ending 4242. "
        "Thanks again for your patience and have a great day!"},
    {"role": "user", "content": "What is the order ID and refund amount?"},
]

result = compress(messages, budget_tokens=80, protect_tail=1)

for m in result.messages:
    print(m["role"], "->", m["content"])

print(f"\n{result.original_tokens} -> {result.compressed_tokens} tokens "
      f"({result.ratio:.2%}); pinned {len(result.retained_matches)} facts, "
      f"dropped {len(result.dropped_segments)} segments")
```

输入里所有结构化事实——`ORD-99172`、`2026-03-15`、`buyer@example.com`、`$1,499.00`——都会留下来，因为它们在 token 预算打包之前就已被留存白名单**固定**住。而像 "We really appreciate you writing in" 和 "Sure thing, let me pull up the record" 这类套话则会被**丢弃**。此示例的实际效果：131 → 78 tokens（约 60%），固定 3 个事实，丢弃 6 段填充内容。

> 为什么加 `protect_tail=1`？默认情况下最后两条消息会被原样保护（保证"用户最新提问"始终在场）。这里倒数第二条恰好是我们想压缩的主体，所以把末尾保护范围降到 1。`protect_tail` / `protect_system` 的完整语义见 [RETENTION.md](RETENTION.md)。

## 留存契约（Retention Contract）

vecr-compress 内置 13 条规则。任何命中规则的片段都会被**固定（pin）**——无论 token 预算多紧都会保留。如果被固定的内容总量超出预算，压缩器会原样返回所有被固定片段并记录一条 warning，**绝不悄悄丢事实**。

| 模式 | 示例匹配 | 用途 |
|---|---|---|
| `uuid` | `3f6e4b1a-23cd-4e5f-9012-abcdef012345` | 追踪 ID、会话 ID、关联键 |
| `date` | `2026-03-15`, `2026-03-15T09:30:00` | 截止日期、时间戳、审计日志 |
| `code-id` | `ORD-99172`, `INV_2024_A`, `CUST#42` | 订单号、发票号、客户编号 |
| `email` | `buyer@example.com` | 联系人记录、PII 审计 |
| `url` | `https://api.example.com/v2/orders` | 接口地址、证据链接、来源 |
| `path` | `/var/log/app/error.log`, `C:/data/report.csv` | 文件引用、错误位置 |
| `code-span` | `` `raise ValueError(msg)` `` | 散文里的内联代码 |
| `fn-call` | `process_refund(order_id, amount)`, `obj.method(a, b)`（仅限代码风格标识符） | code review 里的函数引用 |
| `citation` | `[12]`, `[Smith 2023]` | 学术和法律引用 |
| `json-kv` | `"status": "pending_review"` | 结构化 payload 字段 |
| `hash` | `9f3ab2c4`（≥ 8 位十六进制且包含 ≥ 2 位数字） | Git SHA、内容摘要 |
| `number` | `$1,499.00`, `12.4%`, `v3.2.1` | 金额、比率、版本号 |
| `integer` | `9172`, `99172`, `2026`（≥ 4 位数字） | ID、参考号、年份 |

用你自己的规则扩展契约：

```python
import re
from vecr_compress import compress, RetentionRule, DEFAULT_RULES

custom_rules = DEFAULT_RULES.with_extra([
    RetentionRule(name="invoice", pattern=re.compile(r"INV-\d{6}")),
])
result = compress(messages, budget_tokens=2000, retention_rules=custom_rules)
```

规则测试和扩展的详情见 [RETENTION.md](RETENTION.md)。

## 可复现的 Benchmark

大海捞针（NeedleInHaystack）存活率：11 个针 × 3 个位置 × 6 种压缩比 × 3 种配置 = 594 次试验（`bench/needle.py`）。

**结构化针（7 个）—— baseline vs. L2 留存**

| ratio | baseline | + L2 retention |
|---:|:---:|:---:|
| 1.00 | 100% | 100% |
| 0.50 | 100% | 100% |
| 0.30 | 100% | 100% |
| 0.15 | 100% | 100% |
| 0.08 | 100% | 100% |
| 0.04 | 100% | 100% |

在这份合成数据上，baseline 启发式打分器保住了所有结构化 token。L2 把这个**观察到的现象**升级为**确定性契约**——不管工作负载、打分器或数据分布怎么变，这个 100% 都成立，不只是本测试样本里成立。只要 `ORD-\d+` 出现在输入中，它就会出现在输出中。

**隐蔽针（4 个，纯散文）—— 权衡在这里体现**

| ratio | baseline | + L2 retention |
|---:|:---:|:---:|
| 1.00 | 100% | 100% |
| 0.50 | 100% | 100% |
| 0.30 | 83% | 83% |
| 0.15 | 75% | 67% |
| 0.08 | 75% | 0% |
| 0.04 | 75% | 0% |

L2 的代价：必须保留的结构化内容会把预算占满，激进压缩比下基本没地方留给纯散文里的隐蔽针（目标 0.15 → 实际 0.16，因为白名单会覆盖预算设置）。在自然语言问答（HotpotQA probe，N=100）上，混合了 question 感知的打分器在压缩比 0.5 时比单纯 L2 额外提升 **+9.9 个百分点**的关键支撑事实存活率——通过 `compress(..., use_question_relevance=True)` 启用（v0.1.3+）。默认关闭，让确定性契约保持醒目；当上下文偏长散文、且有一个明确问题时，再打开它。详情见 [docs/BENCHMARK.md](docs/BENCHMARK.md)。

注：当必须保留的内容很多时，实际压缩率可能超过目标值——这是刻意设计的预期行为，不是 bug。

复现：

```bash
pip install -e .
python -m bench.needle
```

## 安装

```bash
pip install vecr-compress                  # 核心（依赖 tiktoken）
pip install vecr-compress[langchain]       # 含 LangChain 适配器
pip install vecr-compress[llamaindex]      # 含 LlamaIndex 适配器
```

需要 Python 3.10+。

## LangChain / LlamaIndex

框架适配器通过 extras（`[langchain]`、`[llamaindex]`）按需安装。核心压缩逻辑**不依赖**任何框架。

**LangChain** —— 在把聊天历史送给 chat model 前先压缩：

```python
from langchain.messages import HumanMessage, SystemMessage
from vecr_compress.adapters.langchain import VecrContextCompressor

compressor = VecrContextCompressor(budget_tokens=2000)
compressed = compressor.compress_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Long conversation history..."),
    HumanMessage(content="The actual question."),
])

# 自然语言问答场景（问题明确、上下文偏散文）可开启 question 感知混合打分：
#   VecrContextCompressor(budget_tokens=2000, use_question_relevance=True)
```

**LlamaIndex** —— 在最终答案合成（synthesis）前对检索到的 node 做后处理：

```python
from llama_index.core.schema import NodeWithScore, TextNode
from vecr_compress.adapters.llamaindex import VecrNodePostprocessor

processor = VecrNodePostprocessor(budget_tokens=1500)
kept = processor.postprocess_nodes(nodes, query_str="用户的问题")
```

## 工作原理（30 秒版）

按顺序应用两层：

1. **留存白名单**：命中任意内置规则的片段会被**固定**，完全跳过预算打包。
2. **启发式打包**：剩余片段按熵和结构信号（数字、括号、大小写）打分；`Hi!`、`Thanks!`、`As an AI…` 这类套话得 0 分，在任何预算计算之前就被直接丢弃；其余片段按 token 预算贪心打包。

question 感知的混合打分通过 `compress(..., use_question_relevance=True)` 开启（v0.1.3+），它在启发式分数（权重 0.6）之上叠加一层 Jaccard 重叠分数（权重 0.4）。进阶用户也可以传入自定义 `ScorerFn`——`blended_score`、`heuristic_score`、`question_relevance` 都已从 `vecr_compress` 顶层导出。详情见 [RETENTION.md](RETENTION.md)。

## 对比

| | 方法 | 开源 | 留存契约 |
|---|---|---|---|
| **Compresr (YC W26)** | LLM 摘要，托管模型 | 否 | 无 —— JSON 原子级处理在规划中 |
| **LLMLingua-2** | 概率式 token 分类器 | 是 | 无 |
| **LangChain DeepAgents compact** | 智能体自主触发 | 是（LangChain） | 无 |
| **服务商原生 compaction**（OpenAI / Google） | 不透明、绑定单一服务商 | 否 | 无 |
| **vecr-compress** | 正则白名单 + 启发式背包（knapsack） | 是 | **确定性、可审计** |

追求最高压缩率选 Compresr。需要纯 Python 研究工具选 LLMLingua-2。想要一个**可审计、可扩展、端到端能讲清楚**的白名单压缩器，并且可以接受 v0.1 的限制（仅 Python、句子级粒度、不支持流式）——选 vecr-compress。

## 明确不做的事

- **不支持流式输出**。`compress()` 是同步、一次性调用。
- **不改写工具调用**。`tool_use` / `tool_result` 块原样透传——安全，但那些轮次的压缩增益为 0。
- **只做句子级粒度**。不做 token 级裁剪、不做基于学习模型的改写。
- **针对英文调优**。停用词表和正则都是英文优先，多语言效果**未经测试**。
- **不带 embedding 打分器**。Jaccard 是词法重叠。语义相关性打分放在参考网关（reference gateway）里。

## 贡献 / 许可 / 链接

Apache 2.0。欢迎通过主仓库贡献。

- 主仓库：[https://github.com/h2cker/vecr](https://github.com/h2cker/vecr)
- Issues：[https://github.com/h2cker/vecr/issues](https://github.com/h2cker/vecr/issues)
- 留存契约详情：[RETENTION.md](RETENTION.md)
- 更新日志：[CHANGELOG.md](CHANGELOG.md)

# LabMate `paper_rag` 面试知识库

这份文档是个人面试复习材料，用来帮助你讲清楚 LabMate / `paper_rag` 的开发过程、RAG 框架是怎么搭起来的、每个阶段解决了什么问题，以及当前系统如何为后续实验管理系统服务。

注意：这是开发和面试准备文档，不是公开 README。

## 1. 一分钟项目介绍

LabMate 是一个面向深度学习科研流程的轻量级 AI lab assistant。当前已经完成主体框架的是 `paper_rag`，它是一个本地优先的科研论文 RAG 系统。

`paper_rag` 的目标是帮助研究人员管理本地论文库，完成论文检索、证据问答、结构化 metadata 构建、常见知识缓存、多论文对比和总结。后续它会作为文献知识服务，被实验管理系统 `experiment_agent` 调用。

面试时可以这样说：

> 我做了一个轻量级科研论文 RAG 系统。它可以从本地 PDF 论文库中提取文本，切成带来源信息的 chunks，构建 FAISS 向量索引，支持语义检索和带引用的问答。后续我又加入了结构化 paper cards、metadata search、query cache、topic cache、多论文对比、LLM-assisted comparison，以及面向实验管理系统调用的公共 API。整个实现没有使用 LangChain，而是用显式的 Python 模块和 CLI，方便调试、维护和后续 agent 集成。

## 2. 当前系统整体架构

当前 `paper_rag` 可以理解为四层：

1. 数据与产物层
   - 原始 PDF 放在 `data/raw_papers/`。
   - 运行时产物放在 `paper_rag/storage/`。
   - chunks、vector store、paper cards、cache 都是本地生成文件，不提交 Git。

2. 核心逻辑层
   - 代码在 `paper_rag/src/paper_rag/`。
   - 包括 ingestion、indexing、search、QA、paper cards、metadata search、router、cache、comparison、pipeline、public API。

3. CLI 层
   - 脚本在 `paper_rag/scripts/`。
   - 当前推荐运行方式是：

```bash
python paper_rag/scripts/xxx.py ...
```

4. 对外集成层
   - Stage 8B 新增 `paper_rag.api`。
   - 未来实验管理系统可以直接 import Python API，而不是 shell 调 CLI。
   - `TOOL_CAPABILITIES` 可以告诉调用方某个工具是否会调用 LLM、是否依赖 FAISS、是否写入 storage。

## 3. RAG Pipeline 是怎么组建的

基础 RAG 主流程是：

```text
PDF 文件
-> 扫描 inventory
-> PDF 文本抽取
-> page/chunk 切分
-> 保存 chunk metadata
-> embedding
-> FAISS 向量索引
-> query embedding
-> top-k chunk 检索
-> 构造 evidence context
-> LLM 基于证据回答
-> 输出答案和来源引用
```

为了让系统更像科研助理，后续又扩展了 metadata、cache 和 comparison：

```text
paper inventory / chunks
-> paper cards
-> enriched paper cards
-> cleaned paper cards
-> metadata search
-> query router
-> query cache / topic cache
-> structured comparison
-> LLM-assisted comparison
-> evidence-grounded comparison
-> public API for experiment_agent
```

核心原则：

```text
能用稳定 metadata 解决的问题，优先不调用 LLM。
metadata 不够时，再用 vector retrieval。
只有需要自然语言生成、总结、解释、证据综合时，才调用 LLM。
```

## 4. 每个 Stage 在做什么

### Stage 1: PDF Inventory and Chunk Ingestion

作用：
- 扫描本地论文目录。
- 给每篇论文分配稳定 `paper_id`，例如 `p000001`。
- 抽取 PDF 文本。
- 按页和长度切成 chunks。
- 保存 source file、page number、chunk id 等 provenance 信息。

主要文件：
- `paper_rag/scripts/ingest_pdfs.py`
- `paper_rag/src/paper_rag/ingestion.py`

输出：
- `paper_rag/storage/paper_inventory.csv`
- `paper_rag/storage/chunks.jsonl`

面试表达：

> RAG 的第一步不是直接问 LLM，而是把原始 PDF 变成可检索、可追溯的数据。我给每篇论文分配稳定 paper_id，并且每个 chunk 都保留 page number 和 source file，这样后续回答可以追溯来源。

### Stage 2: FAISS Vector Index Building

作用：
- 读取 `chunks.jsonl`。
- 用 sentence-transformers 对 chunk 文本做 embedding。
- 用 FAISS 建立本地向量索引。
- 保存和 vector id 对齐的 metadata。

主要文件：
- `paper_rag/scripts/build_index.py`
- `paper_rag/src/paper_rag/indexing.py`

输出：
- `paper_rag/storage/vector_store/index.faiss`
- `paper_rag/storage/vector_store/metadata.jsonl`
- `paper_rag/storage/vector_store/manifest.json`

面试表达：

> 我选择 FAISS 是因为当前论文库是本地单用户场景，FAISS 足够轻量、速度快，也不需要额外服务。metadata 和 FAISS vector id 对齐，检索结果可以还原到具体论文、页面和 chunk。

### Stage 3: Vector Search

作用：
- 输入 query。
- 使用同一个 embedding model 编码 query。
- 在 FAISS 中检索 top-k chunks。
- 返回 score 和来源信息。

主要文件：
- `paper_rag/scripts/search_papers.py`
- `paper_rag/src/paper_rag/search.py`

面试表达：

> 我把 search 和 QA 分开实现。这样可以先单独评估 retrieval 是否靠谱，再决定是否调用 LLM。检索结果本身也能被其他模块复用。

### Stage 4: Evidence-Based QA

作用：
- 先检索相关 chunks。
- 把 chunks 压缩成 evidence context。
- 调用 OpenAI-compatible LLM。
- 要求 LLM 只基于证据回答。
- 输出答案和 sources。

主要文件：
- `paper_rag/scripts/ask_papers.py`
- `paper_rag/src/paper_rag/qa.py`
- `paper_rag/src/paper_rag/llm_client.py`

面试表达：

> QA 阶段的重点是 evidence-grounded。Prompt 里明确要求模型只能使用给定 chunks，如果证据不足就说 evidence is insufficient。这样能降低 hallucination，并且答案后面会附来源。

### Stage 5A: Heuristic Paper Cards and Metadata Search

作用：
- 从 inventory 生成结构化 paper cards。
- 初始版本主要基于文件名和目录做 heuristic metadata。
- 支持 metadata search，不需要向量检索或 LLM。

主要文件：
- `paper_rag/scripts/generate_paper_cards.py`
- `paper_rag/scripts/metadata_search.py`
- `paper_rag/src/paper_rag/paper_cards.py`
- `paper_rag/src/paper_rag/metadata_search.py`

核心字段：
- `paper_id`
- `title`
- `year`
- `venue`
- `task`
- `method_keywords`
- `datasets`
- `metrics`
- `baselines`
- `summary`
- `limitations`

面试表达：

> 不是所有问题都适合走向量检索。比如“哪些论文用了某个 dataset 或 metric”，metadata search 更便宜、更稳定。所以我引入 paper cards，作为结构化文献 metadata 层。

### Stage 5B: LLM-Assisted Paper Card Enrichment

作用：
- 用 LLM 补全 paper card 中缺失的结构化字段。
- 只发送有限 chunks，不把整篇论文都塞给 LLM。
- 提取 task、method keywords、datasets、metrics、baselines、summary、limitations。
- 每张 card 记录 enrichment 状态。

主要文件：
- `paper_rag/scripts/enrich_paper_cards.py`
- `paper_rag/src/paper_rag/paper_card_enricher.py`

面试表达：

> 我没有让 LLM 每次查询都重新理解论文，而是用 LLM 离线补全 paper cards。这样昂贵的论文理解结果可以复用，后续 metadata search 和 comparison 都能利用。

### Stage 5C: Paper Card Metadata Cleanup

作用：
- 清理弱标题 metadata。
- 检测 arXiv id、raw PDF filename、article-text filename 这类 `title/title_guess`。
- 能从已有字段或文件名得到更好标题时自动更新。
- 不能恢复时标记 `needs_review`。
- 支持 manual title override。

主要文件：
- `paper_rag/scripts/cleanup_paper_cards.py`
- `paper_rag/src/paper_rag/paper_card_cleanup.py`

已知问题：
- 有些论文标题仍然可能是 `2412.08197v1`、`2504.05224v1` 这种文件名。
- 这会影响 compare 输出可读性。
- 这是 metadata 质量问题，不是 comparison 逻辑问题。

面试表达：

> RAG 系统的质量不只取决于 embedding 和 LLM，metadata 质量也很关键。因为后续 metadata search、comparison summary 都依赖 paper cards，所以我单独做了 metadata cleanup 阶段。

### Stage 6A: Rule-Based Router and Exact Query Cache

作用：
- 提供统一入口 `paper_query`。
- 根据 query 选择 metadata、search 或 answer 模式。
- 增加 exact query cache。
- 避免重复 query 反复检索或调用 LLM。

主要文件：
- `paper_rag/scripts/paper_query.py`
- `paper_rag/src/paper_rag/router.py`
- `paper_rag/src/paper_rag/query_cache.py`

面试表达：

> 我这里没有用 LLM router，而是用了 rule-based router。因为当前查询类型比较简单，用规则更便宜、更可控，也更容易 debug。

### Stage 6B: Topic Cache

作用：
- 缓存稳定的 topic-level 知识总结。
- 常见主题不用每次重新检索和调用 LLM。
- 支持 `--force_refresh`。

主要文件：
- `paper_rag/scripts/topic_cache.py`
- `paper_rag/src/paper_rag/topic_cache.py`
- `paper_rag/src/paper_rag/topic_cache_store.py`

面试表达：

> query cache 缓存具体用户问题，topic cache 缓存稳定领域知识。比如 common metrics、common datasets、frequency-domain features，这些内容变化不频繁，可以缓存下来减少 token 成本并提高回答稳定性。

### Stage 7A: Metadata-Based Multi-Paper Comparison

作用：
- 基于 paper cards 做多论文结构化对比。
- 支持 keyword、dataset、metric、year、venue、paper_id 等过滤。
- 支持 Markdown 和 JSON 输出。
- 不调用 LLM，不加载 embedding，不加载 FAISS。

主要文件：
- `paper_rag/scripts/compare_papers.py`
- `paper_rag/src/paper_rag/compare_papers.py`

面试表达：

> 我先实现 deterministic comparison，而不是直接让 LLM 总结。这样可以给后续实验管理系统提供稳定的结构化输出，比如哪些 papers 用了哪些 datasets、metrics、baselines。

### Stage 7B: LLM-Assisted Multi-Paper Comparison Summary

作用：
- 复用 Stage 7A 的筛选逻辑。
- 只把 compact paper-card fields 发给 LLM。
- 生成自然语言多论文对比总结。
- 第一版不做 chunk-level citations，只引用 `paper_id`。

主要文件：
- `paper_rag/scripts/compare_papers_llm.py`
- `paper_rag/src/paper_rag/compare_papers_llm.py`

边界：
- 不读 PDF。
- 不读 `chunks.jsonl`。
- 不加载 FAISS。
- 不加载 embedding。
- 不提供 chunk-level citation。

面试表达：

> Stage 7B 是基于 paper cards 的 synthesis，不是完整 evidence-grounded comparison。这个边界很重要，因为模型没有看到 chunks，就不能假装给出 chunk citation。

### Stage 7C: Lightweight Evidence-Grounded Multi-Paper Synthesis

作用：
- 复用 Stage 7A 过滤逻辑选择论文。
- 从 chunk metadata 中为每篇论文选择少量 balanced evidence chunks。
- 调用 LLM 生成带 evidence id 的多论文对比。
- 明确加入 comparability and protocol caveats。

主要文件：
- `paper_rag/scripts/compare_papers_evidence.py`
- `paper_rag/src/paper_rag/compare_papers_evidence.py`

边界：
- 这是轻量版 evidence-grounded comparison。
- 不做严格公平性判定。
- 不自动排名论文。
- 不做深度 protocol normalization。

面试表达：

> 多论文对比在科研中很容易因为数据集、训练测试协议、指标和 baseline 不一致而产生误导。Stage 7C 的目标不是自动判断谁更强，而是给出有证据的对比，并提醒协议差异。

### Stage 8A: Unified Workspace Build Pipeline

作用：
- 把已有 ingest、index、cards、cleanup、enrich 串起来。
- `--all` 只运行非 LLM 阶段。
- LLM enrichment 必须显式 `--run_enrich`，避免误消耗 token。
- 每个底层阶段仍然保持独立，方便后续单独调整。

主要文件：
- `paper_rag/scripts/build_workspace.py`
- `paper_rag/src/paper_rag/pipeline.py`

面试表达：

> Stage 8A 不是重写 pipeline，而是做 orchestration。底层每个阶段仍然分离，所以如果后面要改 chunking、indexing 或 card cleanup，只需要改对应模块。

### Stage 8B: Public API and Artifact Defaults

作用：
- 为后续实验管理系统准备稳定 Python API。
- 统一 downstream artifact 路径解析。
- 默认优先使用更高质量的 paper cards。
- 提供 `TOOL_CAPABILITIES`，让调用方知道哪些工具会调用 LLM、依赖 FAISS、写入 storage。

主要文件：
- `paper_rag/src/paper_rag/paths.py`
- `paper_rag/src/paper_rag/api.py`
- `paper_rag/src/paper_rag/__init__.py`

默认 paper card 顺序：

```text
paper_cards_cleaned.jsonl
-> paper_cards_enriched.jsonl
-> paper_cards.jsonl
```

面试表达：

> Stage 8B 是为了让 `paper_rag` 不只是 CLI 工具，而是一个可以被实验管理系统稳定调用的文献知识服务。实验系统可以先看 `TOOL_CAPABILITIES`，避免误调用会消耗 token 或依赖 FAISS 的工具。

## 5. 为什么这是一个 RAG 项目

它符合 RAG 的三个关键部分：

1. Retrieval
   - chunk embedding
   - FAISS 检索
   - metadata search

2. Augmentation
   - 把检索到的 chunks、筛选出的 paper cards 或 evidence snippets 作为上下文。

3. Generation
   - LLM 根据 evidence 或 metadata 生成答案、topic summary、comparison summary。

它还包含实际 RAG 系统常见工程组件：
- chunking
- vector index
- provenance
- metadata layer
- cache layer
- router
- structured outputs
- comparison workflows
- public API

## 6. 为什么不使用 LangChain

本项目刻意没有使用 LangChain。

原因：
- 当前 pipeline 不复杂，用显式 Python 函数更清楚。
- 每个阶段输入输出明确，方便 debug。
- storage path、cache、prompt 都可控。
- 避免框架隐藏行为。
- 后续 `experiment_agent` 可以直接调用这些函数。

面试表达：

> 我没有使用 LangChain，是因为这个项目第一版更重视透明性和可维护性。每个阶段都是普通 Python 函数，输入输出明确，方便定位 retrieval、cache 或 prompt 的问题。

## 7. 关键工程取舍

### 本地优先

数据和运行产物都保存在本地：
- PDFs
- chunks
- vector stores
- model cache
- query cache
- topic cache
- API key

这些都不提交 Git。

### 先 metadata，后 retrieval，最后 LLM

优先级：

```text
topic cache
-> paper cards / metadata search
-> vector retrieval
-> LLM generation
```

这样能降低成本，提高稳定性。

### CLI 和核心逻辑分离

CLI 只是薄包装，核心逻辑在 `paper_rag/src/paper_rag/`。

好处：
- 人可以用 CLI。
- agent 可以直接 import Python API。

### Prompt 保守

Prompt 要求：
- 只使用提供的 evidence 或 metadata。
- 不伪造 datasets、metrics、baselines。
- 没有信息就说未提供。
- 只有真的提供 chunks 时才做 chunk citation。

## 8. 当前重要 Public APIs

当前应该重点记住：

- `build_workspace(...)`
- `search_papers(...)`
- `ask_papers(...)`
- `cleanup_paper_cards(...)`
- `paper_query(...)`
- `get_topic_summary(...)`
- `compare_papers(...)`
- `compare_papers_with_llm(...)`
- `compare_papers_with_evidence(...)`
- `resolve_cards_path(...)`
- `resolve_chunk_metadata_path(...)`
- `TOOL_CAPABILITIES`

面试表达：

> 这些函数相当于 `paper_rag` 的服务边界。未来实验管理系统可以直接调用它们，而不是通过 shell 调 CLI。

## 9. 当前系统可以怎样服务实验管理系统

假设实验管理系统在分析实验时遇到不确定问题，例如：

- “我的实验应该和哪些论文对比？”
- “哪些论文用了 CASIA、FaceForensics++ 或 cross-dataset evaluation？”
- “这些论文的评价指标是否一致？”
- “某个 baseline 是否常见？”
- “两篇论文能不能直接比较？”

它可以调用：

```python
from paper_rag.api import (
    compare_papers,
    compare_papers_with_evidence,
    search_papers,
    ask_papers,
    get_topic_summary,
    TOOL_CAPABILITIES,
)
```

推荐调用策略：

```text
先查 TOOL_CAPABILITIES
-> 如果是结构化问题，调用 compare_papers / metadata search
-> 如果是常见知识，调用 topic_cache
-> 如果需要原文证据，调用 search_papers / ask_papers
-> 如果是多论文实验协议问题，调用 compare_papers_with_evidence
```

关键边界：
- `paper_rag` 提供文献证据。
- `experiment_agent` 管理实验记录、日志、指标和报告。
- `paper_rag` 不负责自动决定实验结论。

## 10. 当前限制

1. paper-card metadata 质量还不完全稳定。
   - 部分标题仍来自文件名。
   - 部分 datasets、metrics、baselines 缺失。

2. Stage 7B 没有 chunk-level citations。
   - 它只基于 paper cards。
   - 需要证据引用时应使用 Stage 7C。

3. Stage 7C 只是轻量版 evidence-grounded comparison。
   - 它能提示 protocol caveats。
   - 但不能自动做严格公平性判定。

4. 还没有 incremental indexing。
   - 当前索引构建更偏 batch。

5. retrieval evaluation 还没有系统化。
   - 后续需要 query set 和 expected hits。

6. `experiment_agent` 还没有正式实现。
   - 当前只是让 `paper_rag` 先成为可调用的 literature knowledge service。

## 11. 面试常见问答

### Q1：你这个 RAG pipeline 是怎么搭的？

可以答：

> 我先做 PDF ingestion，把论文转成带 page 和 source metadata 的 chunks。然后用 sentence-transformers 做 embedding，用 FAISS 建本地向量索引。检索阶段返回 top-k chunks 和 provenance。QA 阶段把 chunks 作为 evidence context 发给 OpenAI-compatible LLM，并要求模型只基于证据回答。后面我又加了 paper cards、metadata search、router、query cache、topic cache、多论文 comparison 和 public API，让系统更适合科研工作流。

### Q2：你怎么降低幻觉？

可以答：

> 首先，QA prompt 明确要求只使用 retrieved chunks。其次，如果证据不足，要求模型说 evidence is insufficient。对于 comparison summary，我只传 compact paper-card fields，并要求模型不要编造 datasets、metrics、baselines 或 citations。最后，我区分 metadata-only 功能和 evidence-grounded 功能，不让模型假装引用没有看到的 chunks。

### Q3：为什么有向量检索还要 paper cards？

可以答：

> 向量检索适合语义相关性，但科研里很多问题是结构化 metadata 问题，比如某篇论文用了什么 dataset、metric、baseline，哪一年、哪个 venue。paper cards 可以让这些查询更便宜、更稳定，也方便多论文对比。

### Q4：query cache 和 topic cache 有什么区别？

可以答：

> query cache 缓存的是用户的 exact query，比如同一句问题下次直接返回。topic cache 缓存的是稳定领域知识，比如 common metrics、common datasets、frequency-domain features。前者是交互级缓存，后者是知识级缓存。

### Q5：为什么不用 LLM router？

可以答：

> 当前 query 类型比较明确，metadata/search/answer 三类用规则就能区分。rule-based router 更便宜、更稳定、更容易 debug。等以后 query intent 复杂到规则难以覆盖时，再考虑 LLM router。

### Q6：这个系统怎么服务后续实验管理 agent？

可以答：

> `paper_rag` 会作为 literature knowledge service。未来 `experiment_agent` 可以调用它查询相关方法、数据集、指标、baseline、局限性和文献证据。实验日志、结果和报告放在 experiment_agent 侧，paper_rag 只负责论文知识和来源证据。

### Q7：多论文对比怎么保证公平？

可以答：

> 当前轻量版不会直接判断公平或排名。它会收集每篇论文的 balanced evidence，并在输出里提醒 comparability and protocol caveats。真正严格的公平比较需要进一步抽取训练集、测试集、split、指标、baseline、preprocessing、cross-dataset protocol 等，这会作为后续 protocol normalization 阶段处理。

### Q8：当前主体框架算完成了吗？

可以答：

> 基础框架已经基本完成。现在已经有从 PDF 到 chunks、FAISS、search、QA、paper cards、metadata search、cache、multi-paper comparison、unified pipeline 和 public API 的完整链路。后续主要是质量升级，比如 metadata cleanup、retrieval evaluation、incremental indexing 和 protocol-aware comparison。

## 12. 项目开发时间线怎么讲

可以按这个顺序讲：

```text
1. 先把 PDF 变成可检索 chunks。
2. 再用 embedding + FAISS 做本地向量索引。
3. 加 search API，先验证 retrieval。
4. 加 QA，用 retrieved chunks 约束 LLM 回答。
5. 加 paper cards，支持 metadata search。
6. 用 LLM 离线 enrich paper cards。
7. 做 metadata cleanup，提高 card 质量。
8. 加 router 和 query cache，减少重复成本。
9. 加 topic cache，缓存稳定领域知识。
10. 加多论文 comparison，先 deterministic，再 LLM-assisted，再 evidence-grounded。
11. 加 unified pipeline，把原始论文库到主要产物串起来。
12. 加 public API 和 artifact defaults，为 experiment_agent 做准备。
```

## 13. 面试时可以反复强调的点

- 我把 retrieval 和 generation 分开，方便单独调试检索质量。
- 从一开始就保留 chunk-level provenance。
- 能用 metadata search 解决的问题，不轻易调用 LLM。
- query cache 和 topic cache 用来减少重复 LLM 成本。
- prompt 是 conservative / evidence-bounded 的。
- 不使用 LangChain，是为了透明、轻量、可维护。
- `paper_rag` 已经具备被后续实验管理系统调用的服务边界。
- 多论文对比中，公平性不是简单总结能解决的问题，需要 protocol-aware evidence。

# `paper_rag` 后续升级点总结

这份文档用于开发规划和面试准备，重点总结当前 RAG 框架已经完成什么，以及后续可以从哪些方向升级。

## 1. 当前基础框架状态

当前 `paper_rag` 的主体框架已经基本完成，已经形成一条从原始论文库到可调用文献知识服务的链路：

```text
PDF 论文库
-> ingestion
-> chunks
-> FAISS index
-> vector search
-> evidence QA
-> paper cards
-> metadata search
-> paper card enrichment / cleanup
-> query cache / topic cache
-> multi-paper comparison
-> unified build pipeline
-> public API for experiment_agent
```

现在它已经可以支持后续实验管理系统调用，例如：

- 查找某个实验应该参考哪些论文。
- 查询某类方法常用哪些数据集、指标、baseline。
- 对比多篇论文的方法设计、数据集、指标和局限。
- 在论文证据不足时提醒不要做过度结论。
- 为实验报告提供文献背景和证据来源。

但是它还不是“完全成熟的科研级 RAG 系统”。后续升级重点应该放在质量、可评估性、增量维护和实验系统集成上。

## 2. Metadata 质量升级

这是当前最重要的升级方向之一。

### 当前问题

部分 paper cards 的标题、数据集、指标、baseline 仍然不完整。例如有些标题可能还是：

```text
2412.08197v1
2504.05224v1
raw PDF filename
```

这会影响：

- metadata search 的召回。
- compare_papers 的可读性。
- LLM comparison summary 的质量。
- 后续实验系统选择 baseline 或相关论文时的可靠性。

### 可升级点

1. 第一页标题抽取
   - 从 PDF 第一页抽取候选标题。
   - 结合字体大小、位置、行数、arXiv pattern 过滤。

2. LLM-assisted title repair
   - 只对弱标题或 `needs_review` 的 cards 调用 LLM。
   - 控制 token 成本。

3. Manual override
   - 支持人工维护小型 title override 文件。
   - 适合少量问题论文。

4. Metadata confidence
   - 给 title、dataset、metric、baseline 等字段增加 confidence 或 source。
   - 区分 heuristic、LLM-enriched、manual override、PDF-extracted。

面试表达：

> RAG 不只是 embedding 检索，metadata 质量会直接影响系统上层能力。后续我会优先增强 paper card 质量，尤其是真实标题、数据集、指标和 baseline 的可靠抽取。

## 3. Retrieval 质量评估

### 当前问题

现在已经能检索，但还没有系统化评估 retrieval quality。

### 可升级点

1. 构建 evaluation query set
   - 例如：frequency-domain features、CASIA、boundary metrics、cross-dataset evaluation。
   - 为每个 query 标注 expected papers 或 expected chunks。

2. 评估指标
   - Recall@k
   - MRR
   - Hit@k
   - citation correctness

3. 对比不同配置
   - chunk size
   - chunk overlap
   - embedding model
   - top_k
   - query rewrite

4. 错误分析
   - query 没召回是因为 chunking、embedding、metadata 缺失，还是 query 表达问题。

面试表达：

> 一个 RAG 系统不能只看生成答案是否流畅，还要评估 retrieval 本身。后续我会用固定 query set 评估 Recall@k、MRR 和 citation correctness，定位问题到底出在检索还是生成。

## 4. Incremental Indexing

### 当前问题

当前更偏 batch build：新增论文后，通常重新 ingest/index/cards。

### 可升级点

1. PDF fingerprint
   - 根据路径、大小、mtime 或 hash 判断论文是否变化。

2. Incremental ingestion
   - 只处理新增或变化的 PDF。

3. Incremental FAISS update
   - 为新增 chunks 添加向量。
   - 维护 chunk id / vector id 对齐。

4. Deletion / rebuild policy
   - 删除论文时如何处理旧向量。
   - 可以先采用“软删除 + 定期全量 rebuild”的简单策略。

面试表达：

> 第一版 batch build 更简单可靠。等论文库变大后，可以做 incremental indexing，只处理新增或变化的论文，减少重建成本。

## 5. Evidence Selection 升级

### 当前问题

Stage 7C 已经能为每篇论文选择 balanced evidence chunks，但选择策略仍比较轻量。

### 可升级点

1. Section-aware evidence
   - 优先选 method、experiment、dataset、metric、limitation 相关 sections。

2. Query-aware reranking
   - 在初始召回后用 cross-encoder 或 LLM reranker 重排。

3. Coverage diagnostics
   - 告诉用户每篇论文是否找到了 dataset、metric、baseline、protocol evidence。

4. Evidence budget control
   - 根据论文数动态控制每篇 chunks 数量。

面试表达：

> 多论文比较时，证据选择不能只看相似度，还要保证每篇论文都有相近的信息覆盖，比如方法、数据集、指标、实验协议。后续可以加入 section-aware selection 和 coverage diagnostics。

## 6. Protocol-Aware Comparison

这是科研场景中非常重要的升级方向。

### 当前问题

现在 Stage 7C 会提醒 protocol caveats，但还不能自动判断对比是否严格公平。

论文之间可能存在差异：

- 训练数据不同。
- 测试数据不同。
- train/test split 不同。
- 指标定义不同。
- baseline 不同。
- preprocessing 不同。
- threshold setting 不同。
- 是否 cross-dataset evaluation 不同。

### 可升级点

1. 抽取 protocol fields
   - train datasets
   - test datasets
   - split
   - metrics
   - baselines
   - preprocessing
   - robustness tests
   - cross-dataset setting

2. Protocol compatibility matrix
   - 显示哪些论文可以直接比较，哪些只能定性比较。

3. Fairness warning
   - 如果协议不一致，明确提示不能直接排名。

4. Experiment-agent integration
   - 实验管理系统可以把自己的实验配置和论文 protocol 对齐。
   - 例如提醒：“你的测试集和某论文不同，因此不能直接引用其结果作为公平对比。”

面试表达：

> 多论文对比最容易误导的地方是协议不一致。后续我会把 comparison 从普通 summary 升级为 protocol-aware comparison，抽取训练集、测试集、指标、baseline 和 split，再判断是否可比。

## 7. Better Query Routing

### 当前状态

当前是 rule-based router，优点是便宜、稳定、可解释。

### 可升级点

1. 保持 rule-based 作为第一层。
2. 对不确定 query 才使用 LLM router。
3. 让 router 输出结构化 intent：
   - metadata
   - search
   - answer
   - topic_cache
   - compare
   - evidence_compare
4. 记录 router decision，用于调试。

面试表达：

> 我不会一开始就用 LLM router，因为成本和可控性不划算。更合理的是 rule-based first，只有 intent 不确定时再调用 LLM router。

## 8. Cache 升级

### 当前状态

已经有：

- exact query cache
- exact topic cache

### 可升级点

1. Cache invalidation
   - 当 paper_cards、chunks 或 index 更新后，相关 cache 应标记过期。

2. Cache metadata
   - 记录基于哪个 cards/index manifest 生成。

3. Topic cache review
   - 对常见领域知识建立人工可审查的 topic list。

4. 谨慎考虑 semantic cache
   - 当前没有实现 semantic cache 是合理的。
   - 后续如果实现，需要避免错误复用语义相近但实际不同的问题。

面试表达：

> Cache 能降低成本，但也会带来过期和错误复用风险。当前我先做 exact cache，后续如果做 semantic cache，需要配套 cache invalidation 和置信度机制。

## 9. Structured Outputs and Reports

### 当前状态

比较工具已经支持 Markdown 和 JSON。

### 可升级点

1. 统一返回 schema
   - 方便 experiment_agent 消费。

2. Comparison report object
   - selected papers
   - extracted evidence
   - protocol caveats
   - generated summary
   - sources

3. 可保存为实验报告素材
   - Markdown
   - JSON
   - later: HTML / notebook / report section

面试表达：

> 为了让 RAG 结果能进入实验管理系统，输出不能只是自然语言，还要有结构化 JSON。这样 agent 可以继续使用 paper_id、metrics、datasets、citations。

## 10. Integration with `experiment_agent`

### 当前准备

Stage 8B 已经准备了：

- `paper_rag.api`
- `TOOL_CAPABILITIES`
- stable public functions
- shared artifact defaults

### 后续集成方式

实验管理系统可以这样调用：

```text
读取实验配置/结果
-> 判断需要文献支持的问题
-> 查看 TOOL_CAPABILITIES
-> 调用 metadata / search / QA / comparison 工具
-> 把文献证据写入实验分析记录
-> 生成实验报告
```

典型场景：

- 实验结果下降，询问是否有论文提到相同 limitation。
- 新模型要做 baseline selection，查询相关论文常用 baselines。
- 写报告时，自动补充相关工作对比。
- 检查当前实验是否缺少 cross-dataset evaluation。

边界：

- `paper_rag` 不管理实验生命周期。
- `paper_rag` 不保存实验日志。
- `paper_rag` 不自动决定模型设计。
- `experiment_agent` 负责实验记录、分析和报告。

## 11. UI / Visualization 是否需要现在做

当前不建议在 `paper_rag` 中单独做复杂可视化。

原因：

- 后续大的实验管理系统会有自己的可视化和报告界面。
- `paper_rag` 更适合作为知识服务和 API 层。
- 如果现在做 UI，容易和未来系统重复。

合理选择：

- 保留 CLI。
- 保留 Markdown/JSON 输出。
- 等 `experiment_agent` 接入后，在实验系统中统一展示。

## 12. 推荐升级优先级

如果只考虑主体框架之后的实用升级，建议顺序：

1. Metadata cleanup / title repair
2. Retrieval evaluation
3. Evidence selection diagnostics
4. Protocol-aware comparison
5. Incremental indexing
6. Cache invalidation
7. Experiment-agent integration
8. Optional UI inside experiment system

## 13. 面试总结句

可以这样总结：

> 当前 `paper_rag` 的主体框架已经完成，具备从 PDF 到检索、问答、结构化 metadata、多论文对比、缓存和公共 API 的完整链路。后续升级重点不是简单增加功能，而是提高 metadata 质量、检索可评估性、证据选择质量和多论文比较的协议公平性，并把它作为文献知识服务接入实验管理系统。

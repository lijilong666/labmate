# RAG 轻量记忆系统规模测试报告

日期：2026-08-26

## 测试目的

验证当前轻量记忆系统在较大合成工作负载下的正确性，包括：

- SQLite 持久化和版本状态；
- 多会话作用域隔离；
- 显式纠正和归档；
- 重复 episode consolidation；
- FTS5 召回；
- stale memory 排除；
- 数据库健康审计；
- memory-aware query cache 的跨会话隔离。

规模测试使用临时数据库，不读取或修改用户的生产记忆数据库，也不调用 LLM。

## 自动回归场景

日常测试套件新增以下规模场景：

- 8 个会话、64 个召回 case 的完整 scale workload；
- 12 个会话、每会话 20 次写入，共 240 次并发 SQLite 写入；
- 单一事实连续纠正 59 次，形成 60 层版本链；
- 20 个会话使用相同 query，验证首次各自 miss、第二次各自 hit，且每个会话只写一个 episode；
- 已存在的数据库文件拒绝覆盖；
- 非法规模参数拒绝执行。

## 大规模独立运行配置

```text
sessions                              100
facts_per_session                       50
episodes_per_session                    15
duplicate_episode_pairs_per_session      3
global_fact_count                      100
retrieval_cases                       1000
top_k                                    6
seed                              20260826
```

最终生成：

- 7,220 个记忆版本；
- 6,880 个 active memory；
- 300 个重复 episode 归档候选；
- 20 个执行纠正与归档生命周期操作的会话；
- 约 6.0 MiB SQLite 数据库。

## 正确性结果

```text
Memory Recall@6                 1.0000
MRR                             1.0000
nDCG@6                          1.0000
Precision@6                     0.1667
stale-memory error rate         0.0000
stale-case rate                 0.0000
session isolation violations         0
post-consolidation redundancy   0.0000
audit healthy                     true
overall passed                    true
```

每个 case 只有一个标注相关记忆，因此在固定返回上限 `K=6` 的定义下，完美命中对应的
`Precision@6` 为 `1/6 = 0.1667`；它不表示召回失败。所有标注记忆均排在第一位，因此 MRR 和 nDCG 均为 1。

## 性能结果

本次本地运行的阶段耗时：

```text
populate              78.23 s
lifecycle              0.76 s
consolidation           7.69 s
1000-case retrieval    13.61 s
audit                   0.64 s
total                 101.54 s
approx throughput      80.96 operations/s
```

这些数字用于定位工程瓶颈，不是跨机器性能基准。当前最明显的瓶颈是 `MemoryStore` 每次写入独立打开连接并提交事务；
对于在线低频写入这是可接受的，但如果未来需要批量导入或迁移数万条记忆，应增加事务级 bulk write API，并分别测量
单条在线写延迟和批量吞吐。

## 结论边界

本轮结果支持以下结论：

- 当前实现能在 100 个会话和 7,220 个记忆版本的合成规模下保持作用域与生命周期正确性；
- 当前确定性 consolidation 能清除本工作负载中的精确重复 episode；
- 当前测试集中没有出现 stale recall 或跨会话污染；
- 当前写入路径存在清晰的批量吞吐优化空间。

本轮结果不支持“真实用户答案质量已经提升”的结论。该结论仍需要人工标注的真实查询集、固定论文库、memory on/off
配对运行，以及回答忠实度和引用准确率评审。

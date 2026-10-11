# 天数 trace 离线源码映射复核

日期：2026-10-11。此次没有启动 GPU、修改官方测试参数或实现优化 kernel。

## 已完成

- 对归档 trace 只读分析：62,204 个独立 kernel 事件、2,736 个折叠 Graph、20 种 kernel 名称。
- 与先前汇总逐项复核：事件次数和耗时一致，原始 trace 哈希不变。
- 14 种名称找到同名源码定义或生成器候选；另 6 种是待导出编译产物的 Inductor 融合 kernel。候选映射仍需安装源码、实际 callable 和模型层标记确认。
- 查到锁定天数 FlagGems 的 linear 已用 M/N/K 的 align32 调优分组；mm 已有候选裁剪、性能模型及 top-15 筛选。它们应作为科研的现有方法对照。

天数插件提交：`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`；FlagGems 提交：`4df52d9168ac55c1c1061cefb4638c570292d896`。vLLM 0.24.0 官方 tag 仅作源码候选参考，未核验安装 wheel 源码一致性。

## 可见热点与限制

| kernel | 占可见独立 kernel 时间 |
|---|---:|
| linear_kernel | 44.19% |
| kernel_unified_attention | 38.61% |
| mm_kernel | 9.67% |
| argmax_kernel_inner | 2.67% |

这些比例不含折叠 Graph 与拷贝，不能作为全模型热点排名或墙钟时间占比。Graph 内部没有独立 kernel 记录；Tensor Shape、dtype、模型层和 NVTX/调用栈未采到。当前 trace 是重复 prompt、并发 8 的缓存命中诊断，不等同于官方 4K/16K 工作负载。

逻辑 backend `default.flagos` 不证明 attention kernel 属于 FlagGems。锁定插件默认走 vLLM TRITON_ATTN 分支；实际开关、backend class、源码归属尚需核验。

## 下一次诊断要补

1. 安装 vLLM 的源文件哈希，关键 callable/backend，JIT/Inductor 编译产物。
2. 安装版 ixsys 是否支持 Graph 节点展开；确认后使用真实支持的选项。
3. linear/mm 的 M/N/K、dtype/stride、模型层、实际配置与调优成本。
4. attention 的 Q/K/V 与 KV cache 布局、序列长度、阶段及 Graph padding。
5. 对应 4K/16K 场景的短 prefill/decode 片段和 Shape 频率。

完整数据、源码 URL/行号/哈希、离线分析脚本、采集剧本与空模板在 [独立科研仓库](https://github.com/sitraliqui/sitraliqui-kernel-autotuning-research/tree/main/docs/trace-analysis-20261011)。本次仅发布元数据和分析结果，不发布原始 trace、模型权重、凭据或环境秘密。

动态 Kernel dispatch 的赛事许可仍待评委回复；当前没有将其集成到比赛路径。正式正确性/吞吐测试继续保留原版 serve 参数和 benchmark 脚本，诊断修改与正式测试隔离。

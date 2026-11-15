# 官网基准与当前天数版本核验（2026-11-08）

## 已核实事实

当前安装 FlagGems 的版本字符串是 5.3.5，实际仓库提交为 `4df52d9168ac55c1c1061cefb4638c570292d896`。2026年11月8日读取官方 GitHub 公共引用，`v5.3.5` 指向 `a7620cc191a0b42e040194622c5758b22a7a25dc`。双向 compare API 均成功：安装提交是标签提交的祖先，标签在其之后 **7 个提交**，没有分叉。这是当前公开引用的核验，不证明该标签在历史任意日期都未移动。

来源：[官方标签](https://github.com/flagos-ai/FlagGems/releases/tag/v5.3.5)、[精确提交比较](https://github.com/flagos-ai/FlagGems/compare/4df52d9168ac55c1c1061cefb4638c570292d896...a7620cc191a0b42e040194622c5758b22a7a25dc)。API 原始响应、读取时间和 SHA256 保存在本目录 `version-provenance`，不是根据版本号或提交日期推测祖先关系。

| 提交 | 实际变化 | 与当前路径的关系 |
|---|---|---|
| `9c02feb7` | 增加天数 `repeat_interleave.self_int` 专用实现及导出 | 属于天数后端，但未证明当前比赛工作负载命中它；不能仅凭厂商相同认定有收益 |
| `06c18e1a` | 增加天数 `matmul_bias_activation` 融合实现及导出 | 代码实际是矩阵乘＋bias＋ReLU；不是当前普通 Linear，也不是 SiLU×up。锁定插件源码没有找到对该名称的直接调用 |
| `999fe10f` | FP8 W8A8 BMM 实现、导出与测试 | 本轮锁定 BF16 模型路径，尚无该 FP8 算子命中证据 |
| `d1c970e0` | `backends.yaml` 的天数 Python 声明从 3.10 改为 3.12 | 构建/环境配置声明，不会由这一个文件自动改变正在运行的 Python 或 Kernel |
| `11ee8d5c` | PR 规则检查工作流、配置和检查工具 | CI 改动，未见模型执行代码改动 |
| `f49e2fb2` | 寒武纪配置调整；公共 `libentry.py` 在候选计时发生 `RuntimeError` 时记录失败并返回无穷大 | **会触及各后端共享的调优入口**，不能说完全与天数无关；但尚无证据表明本轮存在该异常、它改变了选中配置或造成官网性能差距 |
| `a7620cc1` | 寒武纪 paged KV load 的标量广播修复 | 不属于当前已记录的天数 vLLM Triton Attention 源码 |

## 当前命中路径是否被这七个提交改了

依据已封存的 stock/common/long 环境指纹、锁定插件完整源码以及上述 **22 个差异文件**：

- **天数 MM**：`flag_gems/runtime/backend/_iluvatar/ops/mm.py` 与对应 `tune_configs.yaml` 不在差异中。公共 `libentry.py` 有异常处理变化，须与 MM 数学 Kernel 本体区别看待。
- **Attention**：当前记录的是 `vllm.v1.attention.ops.triton_unified_attention`，以及 `triton_reshape_and_cache_flash`；它们属于当前 vLLM 源码，不在这组 FlagGems 提交中。不能把寒武纪 KV 修复当成天数 Attention 修复。
- **RoPE**：当前锁定的 FlagGems `modules/rotary_embedding.py`、`fused/rotary_embedding.py` 未被这七个提交修改；插件侧 rotary/dispatch 文件也不是此 FlagGems 仓库比较的改动。
- **dispatch**：天数后端新增两个算子的导出，公共算子表新增 FP8 BMM；当前插件 dispatch 源码没有因此被改写。静态字符串检索只能说明未找到直接调用，不能替代全部运行期调用证明。

因此，已发现真实版本身份差异，但**没有发现“这七个提交直接替换当前 MM/Attention/RoPE 核心实现，从而保证补齐性能差距”的证据**。也不能反过来保证版本变化对性能完全没有影响。

## 官网到底锁定了什么

2026年11月8日重新读取[完整官方赛道页面](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)：指定 `vllm-plugin-FL` 的 `flagos-2026-s2` 分支、FlagGems 的 `v5.3.5` 标签、MiniCPM5-2B、天数 BI-V150、沐曦 C500（显存64GB），提供原始 serve 命令、两种负载及基准表；声明评测脚本和依赖在镜像内置。

页面没有给出基准测试所用的插件精确 commit、不可变镜像 digest、天数显存容量、主机 CPU/RAM、具体驱动/PyTorch/Triton/EvalScope 版本、PCI、运行频率或功率。FlagGems 标签可解析到当前精确提交，但这不等于官网基准图片已经披露了当时实际安装的 commit。页面的 EvalScope v1.5.1 链接是数据集说明链接，不能当作基准镜像安装版本。

当前实际机器为 BI-V150 32GB、主机20GiB；驱动4.5.0；torch `2.10.0+corex.4.5.0`；vLLM `0.24.0+empty`；插件提交 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`；FlagTree `0.6.2a3+iluvatar3.6`；EvalScope `1.11.1`。原版与候选在同一环境、同一 boot/PCI 下测量并封存，环境指纹一致。**尚不能证明这与官网基准的完整环境身份一致。**

## 对现有结果意味着什么

已测的本机对照不会因为发现版本差异而消失：完整4K中，融合组合相对本机原版为 **+3.7784% total tokens/s**，相对共同修复版为 **+4.0401%**；但每臂只有一次独立服务，仍缺反序复验，不能写成稳定增益。

官网4K表的吞吐为2028.01 tokens/s、Mean TTFT为11573.53 ms。按页面99%吞吐/101%TTFT允许范围计算，候选当前1926.49 tokens/s、12344.70 ms，距离允许边界还需要约 **+4.22%吞吐、−5.31% Mean TTFT**。这里是距离允许边界，不是距离官网原始均值。融合相对共同修复版的 Mean TTFT 几乎不变，现有窄范围 Decode 融合没有解决这项差距。

没有证据承诺更新7个提交就能达标；也没有依据把现有对照判成全部白费。版本与实现差异必须披露，正式交付的基准来源还需要解决。此前数值差异、提交入口和主选择逻辑的风险继续保留。

补充：官网未明确最终评分分母究竟固定用基线表，还是评委同机重跑原版。上述+4.22%/−5.31%只作为官网表参考筛查，不代表已获得最终资格判定；不能从“统一复现”推定必然重跑stock。原文核对与未发送的问题草稿见OFFICIAL-SCORING-DENOMINATOR.md。

## 接下来可验证的短期路线

1. 完成正在运行的锁定版本16K stock→long对照、归档和独立复算；本轮不升级、不插入 profiling、不改服务或评测脚本。
2. 另开独立服务做4K反序复验，判断本机净收益是否重复；旧顺序结果原样保留，不能只挑最快轮。
3. 长测结束后，把上述版本差异与官网未披露项作为具体核对清单。若需要以官方标签重新建基准，应另建隔离源码/环境、单独记录构建和正确性，不能用换版后的成绩覆盖旧结果，亦不能把上游改动算成本队贡献。
4. TTFT 诊断优先核对真实 prefill/排队/Graph执行条件，再针对实测开销实施修改。Attention＋Linear在历史窗口占比高，只能支持选题优先级，不能直接换算模型提升。

当前没有执行升级、安装或外部提交。详细机器可读结果为 `version-provenance/source-environment-audit.json`，22个文件与7个完整提交ID均已列入。

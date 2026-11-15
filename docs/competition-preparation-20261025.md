# 比赛准备交付：prefix B验收与天数2D诊断

2026-10-25。仅离线准备；未SSH、未开关机、未预约、未获取GPU锁、未改科研侧文件。

## 科研侧最新状态与分工

本次只读检查科研会话最近进度与本地报告，发现它已完成新的prefix短模型试验，并转向天数分段主案例。这改变了此前比赛计划的前置条件：prefix B已有性能信号，但长生成/并发的一致性采用门没有通过，原版A/A也存在波动。研究试验关闭prefix caching，不是比赛官方结果。

| 工作 | 科研侧 | 比赛侧本次交付 |
|---|---|---|
| prefix B | 保留已有实现、研究短测及失败证据 | 固定补丁交接合同和分级验收；不重复实现B |
| 天数Attention | 独立算子的S值、2D强对照和机制实验 | 固定S16的真实服务2D/mixed覆盖与Shape诊断；不做分段搜索 |
| GPU | 科研正在准备其后续上机实验 | 本次完全离线；以后独占共享锁并核实实际任务，不占用科研实例 |

尚未向其他会话发送消息。分工与协调记录保存在原工作区`outputs/session-split-20261025/competition-preparation.json`，不改写它的GPU状态记录。

## 沐曦prefix B接入和验收

目录：`scripts/metax_prefix_acceptance_20261025/`。

交付包括`contract.json`、独立官方CSV重算器`audit_official_csv.py`、证据门台账`admission.py`、CPU检查和完整README。

验收顺序固定为：确定版本补丁/源锁 → 局部prefix与完整Attention/KV → 长生成与A/A定位 → 真实分支覆盖及短回归 → 官方105题精度 → 原4K/16K各四轮 → 有价值正差反序确认 → 规则/TTFT和提交复核。

当前没有收到新的固定导出，长生成门未过，TTFT最终口径尚待确认，因此“采用阻止清单”应为blocked。不会因A/A也有差异自动放行，不会沿用研究试验的服务参数，不把研究TTFT改善换算成官方吞吐。

官方CSV重算器已使用旧沐曦原版/直写完整CSV复核，保留“没有收益”的原结论；额外注入缺轮、错误请求、token变化、NaN和额外轮均被拒绝。台账是审查辅助，不替代对原始证据和实际数值结果的判断。

## 天数真实服务2D混合路径诊断包

目录：`scripts/tianshu_mixed_diagnostic_20261025/`。

从既有成功覆盖包复制并保留Apache许可证、数值修复和守护，独立增加phase计数、样本/直方图配额、固定两窗口客户端、部分结果保留及收尾。固定S16，原FULL_DECODE_ONLY serve；不搜索S值。

4K诊断：4096输入/1024输出/64请求/并发64，300秒上限。排空确认后，16K诊断：16384输入/128输出/8请求/并发8，300秒上限。每次排空60秒，启动360秒，总任务1800秒，另留180秒回收。超时是partial，不能当官方成绩；排空不确认就停止后续窗口。

默认light模式记录已有CPU字段、Q shape/stride、可取得的grid/tile、2D/3D直接提交、capture/replay关联及分phase计数。每phase有独立配额，startup不能耗尽请求期样本。metadata构造ID不冒充真实step或层ID。GPU值不读取；缺值保留unknown。

可选torch_profiler每phase至多一窗口，只在非捕获2D提交开始，下一安全调用达到2秒/128提交边界时停止；远端工具能力尚未验证。导出后检查256MiB上限。活动汇总不冒充完整Attention或Graph内部计时；空trace和未关联时间保留未知，累计duration不当模型关键路径占比。

## GPU协调

相邻`scripts/competition_gpu_lease_20261025.py`提供本机inspect/acquire/release。创建采用独占方式，不覆盖已有锁，即使过期也需先确认旧任务。释放校验owner和lease ID，需要确认远端所属进程清理。测试仅在合成临时文件中验证互斥，本轮未创建实际GPU-LEASE.lock。

未来执行天数runner必须携带lease副本、当前host/boot pin、冻结精度前置与当前Linux CPU清理回执。远端副本不能替代本机锁持续有效和实时任务核验。没有满足这些条件时，脚本应停止，不能靠旧SSH/历史关机记录直接运行。

## 验证状态与限制

本机CPU检查通过：源与守护摘要、原serve、分phase样本和overflow、禁止GPU值读取、unknown Graph/capture-only拒绝、超时保留、空activity未知、官方CSV完整性及资源锁互斥。

首次本机检查发现负向测试未捕获预期RuntimeError，以及Windows默认临时目录的ACL阻止写fixture；已修正为捕获预期拒绝和使用工作区普通目录，后续检查通过。没有为此放宽任何科学或性能门。

**尚未验证：**新诊断包在Linux的进程清理、实际Triton/vLLM接入、真实请求采集和Profiler活动。已有原始守护测试证据不替代本次当前环境CPU前置门。本次无新GPU数据、无新吞吐成绩。

冻结SHA、两个CPU检查输出、B阻止清单、上传tar包与逐文件完整性回执保存在原工作区`outputs/competition-preparation-20261025/`。本交付未Git提交或推送，原有未提交文件不清理。

下一步先核对科研侧是否有可导出的prefix一致性定位结果；天数诊断可在独立、明确授权且不与科研冲突的预约窗口完成当前环境前置检查后采集，不直接追加数小时长测。


## 10月25日本轮上机更新

prefix B同服务串行诊断执行A,A,B,A，前三次生成token一致，第四次A从第3个token分歧；新增纯A控制两次即从第3个token分歧，B执行为0。因此B仍blocked，原路径/诊断状态的数值差异待归因；没有新增Math-500或正式吞吐成绩。V2与纯A原件分别59、43个文件已逐一校验，进程清理和12个源码摘要核验通过。详见原工作区outputs/metax-prefix-serial-gate-20261025-v2/README.md，本段覆盖前述“仅离线准备”的历史状态。天数本轮未上机或新预约。


## 10月25日20时段：原版无包装对照离线就绪

独立包work/metax-original-aa-20261025-v1不含候选/导入hook；冻结相同512/128输入、seed及cache_salt，最多4次原版串行请求，首次差异停止。11组合成CPU协议/归档检查通过，打包及逐文件摘要核验通过。当前科研天数G1持共享GPU锁，本轮不SSH、不上传、不启机。待其实际收尾并释放后，按竞赛专用实例重新核验再上机；无新数值或正式成绩。详见原工作区outputs/metax-original-aa-20261025-v1/README.md。


## 用户修订GPU协议：独立实例可并行

最新协议为原工作区outputs/session-split-20261025/GPU-PROTOCOL-v2.md。按平台+稳定实例ID加锁，不同GPU并行、同资源互斥；旧锁仅保护其指向资源。待测原版A/A控制器已迁移v2，科研天数重测不再阻止竞赛沐曦。科研侧AGENTS/HANDOFF已同步，科研正在运行的实验脚本和旧锁不动。11组本地协调测试及原版CPU检查通过，本次不产生实际GPU租约或费用。


## 10月25日20:43 无包装原版A/A短对照完成

OJP/C500/100GiB原版服务，同输入512/128、seed固定，前三次128-token输出一致，第四次第3个token分歧；第二token的logprob已变化。包不含候选或导入hook。因此本项目包装不是差异发生的必要条件，根因仍未确定，B仍不放行；没有新增正式精度或吞吐成绩。34个文件下载逐一校验，所属进程清理、GPU空闲和12个源码未变均已核验。竞赛实例已关机并刷新确认，本轮0.63元，其余实例未动。详见原工作区outputs/metax-original-aa-20261025-v1/RESULTS.md。

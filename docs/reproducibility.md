# 可复现实验规范

## 来源

赛事：[FlagOS S2 Track 2](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)。模型 MiniCPM5-2B。指定插件分支 flagos-2026-s2，FlagGems v5.3.5。公开源版本记录在 `source-lock.json`；实际部署版本另外记录，不能用源代码锁定文件替代环境检查。

## 原始基线

1. 先运行 `python3 scripts/check_environment.py`，保存版本、文件存在性、源码提交与脚本摘要。无 GPU 阶段可以做此检查；运行前还需在 GPU 模式检查硬件/显存/驱动。
2. 确认模型、数据集、插件、FlagGems 和原版 benchmark 均匹配赛事。预装环境有差异时记录并请求主办方确认；当前保留官方镜像原样以记录部署基线，不通过升级掩盖差异。天数 FlagGems tag 提交差异未解决前，不宣称已完成正式合规验证。
3. 天数运行 `bash scripts/serve-tianshu.sh`；沐曦运行 `bash scripts/serve-metax.sh`。服务就绪后运行 `bash scripts/eval-level3.sh`；之后运行 `bash scripts/benchmark.sh`。
4. 不修改 serve 参数、benchmark 脚本、模型、量化或投机采样。此处脚本仅保存赛事页面命令，没有实现优化。
5. Level 3 准确率下界 >=0.95；赛事文字原始基线0.962。`--ignore-errors` 会忽略部分错误，因此必须保存评测错误并核对实际样本数，不能只看单个百分比。
6. 两个负载固定为 `[4096,1024,64,256]` 与 `[16384,1024,64,128]`。记录 total tokens/s、输出中可取得的 TTFT、实际用时和错误。官方脚本内置每个场景RUNS=4、SKIP_FIRST=1：首轮预热，后三轮计入统计。保持原文件不变，报告三轮均值/标准差/范围及各次原始值；不要把官方示例当作本队结果。
7. 线上 TTFT 条款区间仍不完整，提交前需核实完整限制。完整页面可见上界 baseline+1%，但未说明 mean/P99 口径；当前不据此宣称测试合规。

## 日志

每次运行目录 `results/<platform>/<UTC timestamp>/<run type>-<repeat>/`，至少包含 manifest.json、命令、stdout/stderr、开始/结束UTC时间、退出码、summary.json。正式性能测试关闭 profiler；Profiling 单独进行，不能把其数字当作正式吞吐。

每次记录硬件/显存/驱动/运行时、Python/PyTorch/vLLM/evalscope/插件/FlagGems 精确版本与提交、工作区差异、模型来源版本、模型config摘要、原始benchmark摘要。未测量字段使用null并标注not_run，不使用0填补。

公开只提交人工核对的去敏摘要。原始日志和trace默认忽略。禁止提交模型权重、数据集大文件、凭据、token、SSH配置、远程访问地址、账户/券明细和私密环境信息。

## 比赛与科研隔离

比赛仓库只保存当前赛道允许的 MiniCPM5-2B 实验。科研仓库独立保存调优搜索成本、Shape泛化与端到端收益；其他模型或不符合比赛限制的实验需独立环境与目录。复用第三方源码时保留源许可证和归属，新增许可证不能覆盖第三方代码。

## 正确性与Profiling

当前没有优化实现。已实测环境与正确性见[执行状态](execution-status.md)；双平台正式负载吞吐已测量，完整热点仍待展开；沐曦客户端环境正式赛事认可待确认。新增算子时先检查官方算子测试，按dtype/Shape/边界情况/容差做数值验证，再跑模型Level3与正式吞吐回归。不能仅凭算子微基准改善认定端到端有收益。

热点表记录算子名、Shape/dtype、调用次数、总时间与占比、prefill/decode阶段、后端、对应trace。分析搜索空间、搜索时间/编译时间、缓存复用、未见Shape效果和调优回本点；具体热点需测量后填入。

## 沐曦HTTP客户端环境隔离

2026-10-11在64GiB主机内存实例完成原版负载，模型服务仍使用FL、参数不变。先进入官方镜像的MACA/conda环境；SSH默认系统Python可能不是框架Python。另一个终端仅对HTTP客户端隔离GPU与FL初始化：

```sh
VLLM_PLUGINS= CUDA_VISIBLE_DEVICES=-1 MACA_VISIBLE_DEVICES=-1 bash scripts/benchmark.sh
```

服务终端运行原始 `bash scripts/serve-metax.sh`。不要把客户端环境覆盖应用到服务端。benchmark文件和4K/16K负载保持原样；公开结果记录同时给出本次实际命令、工作目录和环境覆盖。此配置已通过完整八轮请求验证，不是算子优化成果；正式赛事是否接受客户端环境仍待主办方确认。

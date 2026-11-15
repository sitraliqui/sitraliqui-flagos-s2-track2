# FlagOS S2 赛道二：双平台合并候选 finalb1

本目录是待用户确认的提交候选，尚未上传比赛。优先保留已验证的沐曦 Prefix B；当前安装包不纳入 Prefix C、M64 激活、采样器、量化、投机或新的 Attention 探索实现；历史排除依据单独保存在 evidence。

## 交付内容与版本

- `vllm-plugin-FL/`、`FlagGems/`：两平台合并后的完整源码，保留原项目许可证。
- `wheels/validated/`：由上述源码构建的两份唯一当前候选安装包。
- `provenance/build-and-install.json`：版本、安装包 SHA256、构建依赖；`rebuild-receipt.json` 证明从交付源码再次构建得到逐字节相同的 wheel。
- `evidence/`：原版历史对照、旧分离版结果、本次合并包验收，分别标注版本，不混填成绩。
- `report.pdf`：用户于2026-11-15修订的当前技术报告。`REPORT.md`保留旧自动生成底稿，未同步本次文字修订；以PDF为准。
- `ACCEPTANCE.md`、`evidence/acceptance.json`：验收状态。
- `manifest.json`：完整交付文件清单。离线运行 `python -B tools/verify_bundle.py` 校验。

插件：`0.3.0rc0+g13eb9be69.finalb1`，SHA256 `7271142a9a1d0fec441a2b7e3d24cc0e44c8f8b2b21b41e865fa3cffbe6da0f6`。

FlagGems：`5.3.5+g4df52d916.finalb1`，SHA256 `307a615dba8a80edcbafa51cf168059b89d25a130b8d9caa3a795d281ed6f8be`。

当前目录由锁定源码归档重建，没有伪造新的 Git 提交哈希。上游基线提交、天数候选提交和每份源码文件的哈希见 provenance 与 manifest。

## 优化内容

沐曦：在原来选中的 MACA FlashAttention prefill 分支中，将 `[0] + prefill_seq_lens.tolist()` 的设备到主机同步及重建改为设备侧 `new_zeros + cat + int32 cumsum`。这里计算分段 KV 长度的累计偏移，`P[0]=0, P[i]=sum(L[:i])`，int32 累加与提示词前缀缓存策略保持原状。Attention 数学、KV Cache 语义、原有后端选择、decode/DCP/cascade 路径保留。只在真实 prefill 分支生效，短输入可能按原后端阈值走 decode，不能以一次短请求证明命中。

天数：保留 formal4 的有界 KV 写入/Decode Attention 融合、Graph 路由及稳定性改动。支持域以交付源码的 profiles/contract 为准，域外返回现有 FlagOS 路径，不改模型权重、官方 Benchmark 或 serve 参数。当前合并包的天数执行源码与旧 formal4 包逐字节相同；这不是合并包已经正式复测的证明。

两平台最终代码分别合在同一份插件仓库和同一份 FlagGems 仓库。平台开关只作用于相应实现；不要将历史独立安装包交替覆盖当前候选。

## 安装及运行复现

需要已有可用的官方 FlagOS 镜像、匹配的厂商驱动及 vLLM 环境；不使用本包升级驱动。沐曦本次栈与实际设备见当前验收 evidence 的 environment 文件。模型文件与 Math-500 数据集校验清单见 `provenance/model-and-dataset-lock.json`。权重不放入提交包。

在独立 Python 环境安装当前两份包，不安装 evidence 中的历史 wheel：

```bash
python -m venv --system-site-packages final-env
final-env/bin/python -m pip install --no-index --no-deps --no-compile wheels/validated/*.whl
```

沐曦保持原官方参数（下例模型路径可替换为同一锁定模型的实际位置）：

```bash
export VLLM_PLUGINS=fl
export CUDA_VISIBLE_DEVICES=1  # 本轮使用双卡实例的第二张卡；不是双卡并行成绩
export PYTHONNOUSERSITE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export MACA_PATH=/opt/maca
export PATH=/opt/conda/bin:/opt/mxdriver/bin:/opt/maca/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib
final-env/bin/python -u -B -m vllm.entrypoints.cli.main serve /workspace/MiniCPM5-2B \
  --port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

沐曦同时需要官方镜像内原有 MACA 动态库路径。本轮仅限制主机编译线程为 1 以控制峰值内存，没有改变模型、请求规模或 Attention 算法。

天数还需启用已归档方案的开关：

```bash
export VLLM_PLUGINS=fl
export VLLM_FL_KV_DECODE_FUSION=1
export VLLM_FL_KV_DECODE_PROFILE=unified
export VLLM_FL_ILUVATAR_STABLE_LINEAR=1
export VLLM_FL_ILUVATAR_OUTPLACE_ROPE=1
export FLAGGEMS_ILUVATAR_STABLE_SMALL_MM=1
final-env/bin/python -u -B -m vllm.entrypoints.cli.main serve /workspace/MiniCPM5-2B \
  --port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

天数该命令对应旧 formal4 方案的复现方式，本轮没有承诺可用预约或合并包天数正式成绩。

服务就绪后，在另一个终端按原口径执行：

```bash
evalscope eval --model minicpm --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}' \
  --work-dir ./accuracy --ignore-errors
python -B tools/accuracy_gate.py ./accuracy
python -u vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

105 题必须全部请求成功并保存预测与逐题评审；`--ignore-errors` 不代表允许漏题通过。四轮每轮请求数与 token 总数必须正确，官方汇总只使用第 2-4 轮，不能省略第一轮。总吞吐不是仅生成吞吐；Mean TTFT 也不是整轮完成时间。

实际运行回执优先于本说明示例。诊断观测代码不装入正式提交 wheel；独立诊断后须恢复原文件并再次校验。正式测试不能开启逐步重日志或把诊断结果写成正式成绩。

## 重新构建

按 `provenance/build-and-install.json` 分别准备记录的构建依赖，执行：

```bash
python -B tools/build_wheels.py --plugin-python /path/to/plugin-build/bin/python \
  --gems-python /path/to/gems-build/bin/python --out /path/to/new-rebuild-directory
```

脚本在临时副本构建，设置记录的版本与时间戳，比较 wheel 内实际包文件，保留构建回执。构建环境须已安装相应依赖，脚本不自动联网。

## 提交边界

最终比赛认可依赖主办方统一环境先复现基线、再运行候选。历史不同日期的成绩只作背景，不作为本次严格加速比。天数命中比例未知、合并包天数未复测等缺口必须保留披露。当前包还需用户最终审阅，禁止据此自动提交、充值或租用算力。

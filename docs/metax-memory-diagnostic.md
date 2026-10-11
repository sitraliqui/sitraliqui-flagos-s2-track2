# 沐曦主机内存分段诊断（2026-10-11）

## 结论与边界

大内存占用在 `torch.cuda.init()` 已经出现，尚未加载 MiniCPM5-2B 权重或捕获 CUDA Graph。它定位了初始化路径，尚未定位 PyTorch/MACA 内部的具体分配器或证明厂商缺陷。120/128GB 主机内存不是已证实的模型最低要求。

仅供诊断的 HTTP 压测客户端隐藏 GPU 并设置空 `VLLM_PLUGINS` 后，`vllm bench serve --help` 成功。**尚未运行完整 4K/16K 压测，沐曦仍没有有效吞吐成绩**；模型服务参数和官方 benchmark 文件均未修改。

## 实测

每阶段启动独立进程，约0.25秒采样 `/proc/<pid>/smaps_rollup`，另保留工作进程自测检查点。单位 GiB；下表为采样与检查点中观察到的最大值，可能遗漏瞬时峰值。RSS含共享页，PSS按共享比例分摊，不能把不同进程RSS简单相加。

| 阶段 | RSS | PSS | 退出状态 |
|---|---:|---:|---|
| 空 Python | 0.014 | 0.007 | 成功 |
| 仅导入 torch（隐藏 GPU） | 0.503 | 0.270 | 成功 |
| 仅导入 torch（GPU 可见） | 0.503 | 0.270 | 成功 |
| GPU 初始化及单元素张量 | 21.515 | 19.499 | 成功 |
| 导入 vLLM 并检测平台 | 22.015 | 19.956 | 成功 |
| 导入并注册 FL | 22.015 | 19.955 | 成功 |
| CLI 帮助，GPU可见且保留 FL | 22.535 | 20.361 | 成功 |
| CLI 帮助，隐藏 GPU且保留 FL | 1.003 | 0.679 | 失败：No CUDA GPUs |
| CLI 帮助，隐藏 GPU且跳过 FL | 0.948 | 0.616 | 成功，仅启动预检 |

`cuda_init_done`检查点RSS为22,558,840KiB（21.514GiB），PSS为20,445,706KiB（19.499GiB），匿名PSS约18.831GiB；之后一个GPU元素仅增加约1MiB。该阶段容器占用约19.00GiB，主要为匿名内存，增长不只是虚拟地址空间。进程退出后容器占用回落，因此不能称为持续泄漏。

隐藏GPU但保留FL的失败堆栈经过 `vllm_fl/platform.py:get_device_capability` 进入torch GPU初始化。跳过FL的客户端进入CPUPlatform并打印帮助，另有空vLLM wheel缺少原生扩展的警告；完整客户端是否依赖它仍需实测。

## 环境、方法与证据

C500：64GiB显存、64GiB主机内存；官方Python3.12.11、torch2.8.0+metax3.7.0.7、vLLM0.24.0+empty、插件0.3.0rc1.post1+g13eb9be69、FlagGems5.3.5、Triton3.0.0+metax3.7.0.7。驱动内核3.8.30、MACA3.7.0.38。完整版本和去敏检查点见 `results-20261011/metax-host-memory-diagnostic.json`。

主诊断05:01:14—05:07:32 UTC完成，逐案例180秒超时、容器54GiB停止阈值，均未触发；除预期的隐藏GPU+FL失败外均成功，无OOM。初次SSH环境缺少MACA_PATH造成的三项失败保留为环境混杂记录，随后从官方Jupyter进程继承MACA_PATH、库路径和PATH重测。

额外CPU客户端预检与主诊断后半段CLI阶段有短暂并行；该时段不使用容器总量进行单进程归因。表中为各自进程数据，GPU初始化阶段发生在并行之前。原始日志、采样、环境白名单及下载归档的SHA256已保存。归档未保存执行源码的摘要，公开脚本为当前复现版本，不追认其为历史源码身份校验。

GPU客户端启动预检（保留官方FL环境）：

```sh
/opt/conda/bin/python scripts/diagnose-host-memory.py --output /data/memory-diagnostic
```

CPU客户端启动预检（仅诊断，不是正式吞吐命令）：

```sh
/opt/conda/bin/python scripts/diagnose-host-memory.py --client-only --output /data/memory-diagnostic-client
```

两者均需先进入官方镜像正确的MACA环境。脚本无模型权重加载、无服务启动、无安装操作；不要将SSH默认系统Python误作官方框架Python。

## 下一步

1. 保留官方服务命令、模型、benchmark文件和4K/16K负载；验证仅HTTP客户端隔离GPU/FL后能否完整生成数据、加载tokenizer并完成请求。不能用帮助命令成功代替实际压测。
2. 确认赛事对客户端环境的要求和测量可比性；未经验证的客户端方案不作为正式成绩。若服务仍受内存限制，再考虑更大主机内存。
3. 对成功客户端重新测每轮成功数、总tokens/s、TTFT、重复波动和容器/PSS峰值。另需继续查明GPU初始化匿名内存的来源，暂不修改运行时或服务参数。

已下载并校验本轮证据，浏览器确认实例“已关机”。本轮仅使用现有普通按量实例；券实例后续按库存处理，累计费用仍遵守用户授权上限。

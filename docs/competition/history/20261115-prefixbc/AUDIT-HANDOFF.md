# Prefix B/C recovered performance audit

Recovered 2026-11-14 UTC from LAJNBONXYAKBAYPK. No experiment rerun.
15 files, 748901 bytes; every downloaded file SHA256 matched remote sha256sum.
Controller state: finished_needs_csv_audit. Both client_rc=0. B and C each have 8 SUCCESS rounds; every row successful requests equals requested count. No OOM, illegal memory access or traceback found in logs. Client CPU vllm._C import warnings were present but requests completed.

Same official configuration, B then C, 4 rounds per case with the first excluded. GPU1 only. 4K=4096 input/1024 output/concurrency64/256 prompts; 16K=16384/1024/64/128.

| Case | B output tok/s | C output tok/s | B TTFT ms | C TTFT ms |
|---|---:|---:|---:|---:|
| 4K | 1009.85 | 988.1966667 | 2956.47 | 2999.55 |
| 16K | 431.81 | 434.6433333 | 25973.98 | 25406.8433333 |

C relative to B: 4K throughput -2.14%, TTFT +1.46%; 16K throughput +0.66%, TTFT -2.18%. This is one sequential experiment, not statistical proof. C is not an all-metric improvement; preserve validated B. C has NOT run full 105-question correctness; prior six-request agreement is not a replacement.

User approved at most 30 CNY additional for follow-up, but current recovery showed performance already finished; do not repeat it blindly. No new experiment was launched by the recovery worker. Instance was restarted approximately 23:47:50 UTC at 8 CNY/h to retrieve logs, shutdown requested after verification. Final billing must come from platform.

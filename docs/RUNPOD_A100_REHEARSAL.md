# RunPod A100 Rehearsal

Use this only as a pre-Chapman infrastructure rehearsal. It proves CUDA visibility, MOABB preparation, exact window gates, materialized absolute config paths, the Slurm wrapper path, checkpoints, metrics, and report writing. It is not a scientific result.

## Budget Guard

Default limit: $5.

```bash
export RUNPOD_MAX_BUDGET_USD=5
export RUNPOD_MAX_SECONDS=3600
```

If the RunPod console shows a known hourly A100 price, set it so the script derives a tighter time cap:

```bash
export RUNPOD_MAX_GPU_USD_PER_HOUR=<console_price>
```

The script refuses `RUNPOD_MAX_BUDGET_USD > 5`. Use RunPod stop/terminate controls as an external guard too, because the MCP currently exposes pod lifecycle but not remote shell logs.

## Pod Shape

Target:

```text
GPU: 1x NVIDIA A100 80GB
CPU: 16 vCPU class
RAM: 128G class
Disk/volume: at least 100G mounted at /workspace
Image: CUDA PyTorch image compatible with Python 3.10/3.11 and CUDA 12.x
Ports: none required for the script path
```

Do not count a non-A100 GPU as a passed rehearsal. A cheaper CUDA GPU can validate imports, but it is only `cuda_smoke_only`.

## Commands Inside The Pod

From a clean checkout at the exact handoff commit:

```bash
cd /workspace
git clone https://github.com/ptlnextdoor/Kahlus-V1.git
cd Kahlus-V1
git checkout <COMMIT_HASH_FROM_HANDOFF>

python -m pip install --upgrade pip
python -m pip install -e '.[moabb,cluster]'

export RUNPOD_MAX_BUDGET_USD=5
bash scripts/cluster/runpod_a100_rehearsal.sh /workspace/neurotwin_data
```

Expected pass markers:

```text
torch_cuda_available=True
torch_cuda_device_count=1
torch_cuda_device_0=<contains A100>
preflight_passed=True
window_count=18144
window_counts_by_split=train:12096,val:2016,test:4032
runpod_rehearsal_passed=True
```

Expected artifacts:

```text
/workspace/neurotwin_data/prepared/moabb_benchmark/event_manifest.json
/workspace/neurotwin_data/prepared/moabb_benchmark/split_manifest.json
/workspace/neurotwin_data/prepared/moabb_benchmark/eval_audit.json
/workspace/neurotwin_data/runs/moabb_a100_smoke/checkpoint.pt
/workspace/neurotwin_data/runs/moabb_a100_smoke/checkpoint_best.pt
/workspace/neurotwin_data/runs/moabb_a100_smoke/metrics.json
/workspace/neurotwin_data/runs/moabb_a100_smoke/summary.json
outputs/configs/moabb_a100.runpod.yaml
```

## Stop Rule

Stop and delete the RunPod pod immediately after collecting logs/artifacts. Do not leave the pod running after the script exits or fails.

Do not make model-quality or paper-readiness claims from this rehearsal. It only reduces launch risk for Chapman.

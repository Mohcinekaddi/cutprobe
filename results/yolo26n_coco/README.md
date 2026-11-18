# YOLO26n results

These are the YOLO26n results reported in the paper, as written by CutProbe: the Pareto-optimal partitionings found
by the search, and the attack campaigns run on their cut tensors. Only the small output files are included (see
[Not included](#not-included)). The `rank` command of this repository recomputes the paper's rankings from these files.

## Setup

- **Model and data.** `models/yolo26n.onnx` on COCO val2017: 4,800 test images and 200 history images. The test images
  are listed in `attack/baseline_gpu.csv`. Clean mAP@.5: 55.95 %.
- **Platform.** Four devices (`configs/yolo26n_coco.yaml`):
  - `localhost_gpu0`: an NVIDIA RTX 6000 Ada GPU;
  - `localhost_cpu012`, `localhost_cpu34` and `localhost_cpu6789`: groups of 3, 2 and 4 CPU cores.
- **Front.** 14 Pareto-optimal partitionings, with 26 distinct cut tensors that form 29 distinct device outputs.
  Robustness uses the device semantics: a compromised device perturbs every tensor it sends.

## Files

| path | content |
|---|---|
| `search_results.json` | Output of `search`.<br>• The Pareto-optimal partitionings: `best_cuts`, `best_placements`.<br>• Six measurements of each one (`repeats`): the search's own measurement and five re-measurements. Throughput T and memory M are their means.<br>• Every evaluated candidate (`raw_history`). |
| `measurements.jsonl` | The 1,425 deployments measured during the search. |
| `analytical_model.json` | The analytical model of the first search level, calibrated on 12 deployments. |
| `search_results_pareto.png` | The fronts of the two stages (2, then 4 devices) and of the two levels of the search. |
| `attack*/` | The attack campaigns (see [Campaigns](#campaigns)).<br>• `attack_summary_gpu.csv` has one row per cut tensor and configuration: mAP@.5 with and without the attack, attack success rate, relative norm `l2_rel_pct`, and the mean and standard deviation of the clean and attacked feature maps.<br>• `attack_paired_breakdown.csv`, `fm_noise_stats.csv` and `synthesis.txt` are further campaign outputs.<br>• Each partitioning has its own folder, `gene_NNN/`. |
| `ranking_reference/` | Output of `rank` with the paper's three reference strengths: mGNA σ 1 γ 0.5, mFGSM ε 0.5, mJSMA θ 0.01. |
| `ranking_strength_grid/` | Output of `rank` with the 19 configurations of the paper's table of attack strengths. |
| `validation/` | Two checks:<br>• Deployment fidelity: `ncnn_check.json`, output of `check-ncnn`.<br>• Agreement of the GPU engine with the real MPI deployment for partitioning 12 on 200 images: `mpi_gpu_agreement_g12.csv`. The partitioning is given in `yolo26n_coco_g12.json`. |

In `ranking.csv` and `summary.txt`:
- Mean and Worst are the accuracy under attack, in % of the clean mAP@.5, averaged over the devices and on the weakest device.
- A device's accuracy under attack is the lowest over the configurations of the budget at its cut, including the null attack.
- "Stealthy" keeps only the (cut tensor, configuration) pairs whose shift Δ is at most 10 %.

## Campaigns

The results come from five campaigns on the same images, because the device semantics and some strengths were added
after the main campaign. The attacks are deterministic: the 672 (cut tensor, configuration) pairs that two folders
share give identical results. `rank` merges the folders.

| folder | configurations | cut tensors |
|---|---|---|
| `attack` | The main grid of `configs/yolo26n_coco.yaml` (25). | The 28 tensors that the next device receives (the earlier default semantics). |
| `attack_extra` | mFGSM ε 0.5; mGNA γ 0.5 with σ 0.08 and 1.2. | Same 28. |
| `attack_grid_prev_20261102` | The earlier main grid, with mFGSM ε 0.75 and 1 instead of 0.8 (26). | Same 28. |
| `attack_boundary` | mFGSM ε 1.25, 1.5, 2; mGNA γ 0.5 with σ 0.2 to 0.8; mJSMA θ 0.0002 to 0.005 (11). | Same 28. |
| `attack_device` | All 41 configurations. | The 14 device outputs where what the device sends differs from what the next device receives. |

## Reproducing the results

**Recompute the rankings from these files.** This needs no GPU and takes a few seconds:

```bash
cutprobe rank -c configs/yolo26n_coco.yaml --set workdir=../results/yolo26n_coco \
    --attack-dirs attack,attack_extra,attack_grid_prev_20261102,attack_boundary,attack_device \
    --tags mgna_g0p5_s1,mfgsm_eps_0p5,mjsma_t0p01 --band 10 --out results/yolo26n_coco/ranking_reference
```

`workdir` is relative to the configuration file. `rank` regenerates the formatted model `format_yolo26n.onnx` in
this folder; git ignores it. For `ranking_strength_grid/`, pass the 19 configurations to `--tags`; they are listed
at the top of `ranking_strength_grid/summary.txt`.

**Rerun the attacks on the same front.** This needs a GPU and the COCO data in `data/coco/` (see the main README).
The paper's main campaign of 25 configurations took 3.2 h on an RTX 6000 Ada. With the default device semantics,
this single campaign covers every cut tensor and configuration that the paper uses:

```bash
mkdir -p runs/yolo26n_coco && cp results/yolo26n_coco/search_results.json runs/yolo26n_coco/
GRID='[{attack: mfgsm, eps_mode: fixed, eps_rel: [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 1, 1.25, 1.5, 2]},
       {attack: mgna, gamma: [0.1, 1.0], sigma_rel: [0.01, 0.05, 0.1, 1.0, 2.5]},
       {attack: mgna, gamma: [0.5], sigma_rel: [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 2.5]},
       {attack: mjsma, theta: [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.08, 0.1]}]'
cutprobe attack -c configs/yolo26n_coco.yaml --set attack.batch=32 --set "attack.configs=$GRID"
cutprobe rank   -c configs/yolo26n_coco.yaml --tags mgna_g0p5_s1,mfgsm_eps_0p5,mjsma_t0p01 --band 10
```

`cutprobe search -c configs/yolo26n_coco.yaml` reruns the search itself. The measured throughput and memory depend on
the platform, so the front can differ on other hardware.

**Rerun the validation.** `cutprobe check-ncnn -c configs/yolo26n_coco.yaml` writes the deployment-fidelity check.

For the GPU/MPI agreement, run partitioning 12 on the same 200 images with each engine:

```bash
REF='[{attack: mfgsm, eps_mode: fixed, eps_rel: [0.5]}, {attack: mgna, gamma: [0.5], sigma_rel: [1.0]},
      {attack: mjsma, theta: [0.01]}]'
G12=$PWD/results/yolo26n_coco/validation/yolo26n_coco_g12.json
cutprobe attack     -c configs/yolo26n_coco.yaml --set workdir=../runs/agreement_gpu --set attack.genes=$G12 \
    --set attack.semantics=rank --set attack.max_images=200 --set "attack.configs=$REF" --no-report
cutprobe mpi-pareto -c configs/yolo26n_coco.yaml --set workdir=../runs/agreement_mpi --set "attack.configs=$REF" \
    --set autodice.deterministic=true --results $G12 --genes 1 --images 200 --history 200
```

- `autodice.deterministic=true` runs the MPI deployment with float32 kernels; `false` runs it with float16 kernels.
- The MPI hooks inject the attack at the receiving device. No skip connection bypasses a device in partitioning 12, so the GPU run uses the receiver's semantics (`rank`) and both engines perturb the same tensors.
- `mpi_gpu_agreement_g12.csv` compares the two runs for each cut and attack.

## Where the paper's numbers are

- **Ranking of the YOLO26n partitionings** (rank, π, T, M, Mean, Worst, weakest cut and attack): the block "whole budget, any" of `ranking_reference/summary.txt`. The next block gives the same ranking with only hard-to-detect attacks (Δ ≤ 10 %).
- **Front averages with the whole strength grid as budget**: the mean of the `mean_pct` column of `ranking_strength_grid/ranking.csv` over the rows with `budget_kind == budget`, for each visibility. This gives 9.5 % of the clean mAP@.5, and 42.8 % with only Δ ≤ 10 %.
- **Cut tensors that one attack halves** (4 of the 26): `map50_attack / map50_baseline` in `attack_summary_gpu.csv`, at the reference strengths.
- **Shift Δ of a (cut tensor, configuration) pair**: `100 · max(|std_attacked − std_clean|, |mean_attacked − mean_clean|) / std_clean`, computed from `attack_summary_gpu.csv`. The relative norm of the perturbation is `l2_rel_pct`.
- **Deployment fidelity**: `validation/ncnn_check.json`. The largest relative difference is 1.2·10⁻⁴ over 308 layer outputs.
- **GPU engine against the MPI deployment**: `validation/mpi_gpu_agreement_g12.csv` (partitioning 12, 200 images). mAP@.5 differs by at most 0.08 point without attack. Under attack, it differs by at most 0.60 point with float32 kernels and 1.00 point with float16 kernels.

## Not included

These outputs are left out because of their size; a rerun writes them again:
- per-image predictions (`predictions_gpu.csv`, up to 247 MB per campaign);
- per-cut files (`layers/`, up to 2.3 GB);
- clean detections (`baseline_gpu_detections.csv.gz`);
- the COCO ground truth (`ground_truth.csv`);
- the per-partitioning figures (`gene_NNN/attack_comparison.png`);
- logs and deployment builds.

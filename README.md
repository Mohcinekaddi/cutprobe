# CutProbe

CutProbe helps select partitionings of a distributed DNN that are both efficient and robust to a compromised
edge device. It extends AutoDiCE: it searches the Pareto-optimal partitionings for throughput and memory, attacks
every cut tensor of these partitionings with label-free feature-map attacks (mFGSM, mGNA, mJSMA), and ranks the
partitionings by their accuracy under attack.

## Requirements

- Linux, Python 3.8, an NVIDIA GPU with CUDA (attack campaigns)
- MPI (`mpic++`, `mpirun`), a C++17 compiler, OpenCV 4 and Vulkan development files (deployments)
- AutoDiCE, built with its ncnn tools:

```bash
git clone https://github.com/parrotsky/AutoDiCE third_party/AutoDiCE
cd third_party/AutoDiCE && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_MPI=ON -DNCNN_OPENMP=ON \
      -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=ON -DNCNN_BUILD_BENCHMARK=OFF ..
make -j"$(nproc)"
```

AutoDiCE can also live elsewhere: set `AUTODICE_ROOT` to its folder.

Python packages:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## Models and data

`models/yolo26n.onnx` is included. The other configurations expect their ONNX model in `models/` and the datasets
in `data/`:

```
data/coco/val2017/                              COCO images
data/coco/annotations/instances_val2017.json
data/imagenet/val/  data/imagenet/test.csv      ImageNet images and (filepath, label) list
data/imagenet/synset_words.txt  data/imagenet/bench.JPEG
```

Paths in a configuration file are relative to that file.

## Usage

Every command reads a configuration file (`configs/*.yaml`); `--set key=value` overrides one entry. The `cutprobe`
script of this folder runs `cli.py` with `$PYTHON` (default `python`). Add the folder to the `PATH`, or call
`python cli.py` directly.

```bash
export PATH="$PWD:$PATH"
cutprobe inspect   -c configs/yolo26n_coco.yaml --layers
cutprobe deploy    -c configs/yolo26n_coco.yaml --cuts 0,82,85,233 --placement localhost_cpu012,localhost_gpu0,localhost_cpu34,localhost_cpu6789
cutprobe calibrate -c configs/yolo26n_coco.yaml
cutprobe search    -c configs/yolo26n_coco.yaml
cutprobe attack    -c configs/yolo26n_coco.yaml --set attack.batch=32
cutprobe rank      -c configs/yolo26n_coco.yaml --tags mgna_g0p5_s1,mfgsm_eps_0p5,mjsma_t0p01 --band 10
cutprobe report    -c configs/yolo26n_coco.yaml
```

Two more commands validate the engines:
- `check-ncnn` compares every layer of the ncnn deployment with ONNX Runtime. `deploy` and `search` also run it.
- `mpi-pareto` runs the attacks on the real MPI deployment of Pareto-optimal partitionings, to compare it with the GPU campaign.

`platform.resources` lists one resource per pipeline stage, e.g. `localhost_gpu0` (GPU 0) or `localhost_cpu012`
(CPU cores 0 to 2). Results are written to `runs/<configuration name>/`. `full_pipeline.sh` runs the search and
the attack campaign for several models in sequence.

At the end of `attack`, the Pareto-optimal partitionings are ranked by robustness: for every device of a
partitioning, the most damaging configuration of the budget at its cut gives its accuracy under attack, and the
partitionings are ranked by the mean of these accuracies over the devices, with the worst one alongside (in % of
the clean accuracy). The ranking is given for the whole budget, per attack and per strength, against every attack
and against the attacks whose shift of the feature-map mean and standard deviation stays within
`report.stealth_band` (hard to detect). It is written to `runs/<name>/attack/ranking/` (`ranking.csv`,
`summary.txt`); `rank` recomputes it for another budget (`--tags`), band (`--band`) or set of campaign folders
(`--attack-dirs`).

## Results

`results/yolo26n_coco/` holds the YOLO26n results reported in the paper:
- the search outputs;
- the attack campaigns on the cut tensors of the Pareto front;
- the rankings recomputed from them with `rank`;
- the validation checks.

Its README explains how each file was produced and where each number of the paper comes from.

## Licence of the included model

`models/yolo26n.onnx` is the YOLO26n model of Ultralytics, distributed under the AGPL-3.0 licence
(https://ultralytics.com/license).

# FreeMoCap · 多段骨盆轨迹评测

[评测目录](../README.md) · [评分脚本](evaluate.py) · [参考结果](results.csv) · [综述首页](../../README.md)

> 用 Qualisys 光学动捕作参考，检查 FreeMoCap 在跑台步行中估计的骨盆轨迹。六段录制各自评分后等权平均，骨盆位置误差为 **17.20 mm**，根轨迹误差（RTE）为 **0.02780%**。

## 一、数据与准备

本例取 FreeMoCap 验证数据集 v1.0 中编号最小的三名参与者（001–003），每人两段完整录制。仓库已附六对骨盆轨迹。安装 NumPy 后，即可在 CPU 上离线评分。

| 输入 | 内容 |
| --- | --- |
| 视频估计 | FreeMoCap 的多相机 MediaPipe 三维轨迹，每帧取左右髋的中点 |
| 动捕参考 | 同次录制、同一帧的 Qualisys 骨盆中心 |
| [随附轨迹](pelvis_tracks.npz) | 每段一个数组，每行依次为帧号、参考坐标（x、y、z）和估计坐标（x、y、z），坐标单位为米 |

原始数据发布的是已同步并完成空间配准的轨迹。两套系统的骨盆关键点均记作 `hips_center`。[数据下载](https://doi.org/10.5281/zenodo.23041223) · [MediaPipe 关键点定义](https://github.com/freemocap/validation/blob/main/validation/utils/freemocap_model_info/mediapipe_model_info.yaml) · [Qualisys 关键点定义](https://github.com/freemocap/validation/blob/main/qualisys_markerset/qualisys_model_info.yaml)

## 二、运行评测

### 1. 准备环境

在仓库根目录打开终端，进入示例目录。以下命令适用于 Linux 或 macOS。

```bash
cd evaluation/freemocap_multivideo_rte
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

### 2. 计算六段轨迹的误差

```bash
mkdir -p runs
python evaluate.py --csv runs/results.csv
```

终端会列出每段的评分以及两种汇总值。新生成的 `runs/results.csv` 保存各段的计分帧数、参考路径长度和误差，可与[仓库中的参考结果](results.csv)逐行核对。

<details>
<summary>可选步骤 · 从官方 Parquet 文件重新提取轨迹</summary>

从数据页下载 `sub-001`、`sub-002`、`sub-003` 三个 ZIP，按随附的 `SHA256SUMS.txt` 校验，解压后应有 `data/sub-001/`、`data/sub-002/` 和 `data/sub-003/`。在当前终端运行以下命令。[官方数据与校验值](https://doi.org/10.5281/zenodo.23041223)

```bash
python -m pip install pyarrow
mkdir -p runs
python prepare_tracks.py data --output runs/pelvis_tracks.npz
python evaluate.py --input runs/pelvis_tracks.npz --csv runs/from_source.csv
```

提取脚本选取 `trajectory=3d_xyz` 与 `keypoint=hips_center` 的记录，检查双方帧号是否连续且逐一匹配，再把毫米换算为米。

</details>

## 三、结果解读

### 1. 每段轨迹的结果

| 参与者 / 录制 | 计分帧数 | 平均位置误差（mm）↓ | RTE（%）↓ |
| --- | ---: | ---: | ---: |
| 001 / 01 | 7,887 | 15.83 | 0.02421 |
| 001 / 02 | 8,497 | 20.34 | 0.02748 |
| 002 / 01 | 5,617 | 21.96 | 0.04039 |
| 002 / 02 | 6,115 | 18.70 | 0.02929 |
| 003 / 01 | 6,903 | 13.48 | 0.02432 |
| 003 / 02 | 7,224 | 12.87 | 0.02113 |

每段独立做一次刚体对齐，允许旋转和平移，尺度固定为 1。平均位置误差是对齐后各帧三维距离的均值。RTE 用该均值除以**同一段参考骨盆在实验室坐标系中的累计路径长度**，两者统一用米，再乘以 100%。例如，001 / 01 的平均误差约为 0.01583 m，参考路径长约 65.39 m，因此 RTE 为 0.02421%。[WHAM 的 RTE 实现](https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/eval/eval_utils.py#L464-L482)

### 2. 六段怎样汇总

- **各段等权平均。** 六个 RTE 各占六分之一，得到 **0.02780%**。同样对六个平均位置误差取均值，得到 **17.20 mm**。
- **按计分帧数加权。** 每段的权重为其计分帧数占总帧数的比例。六段共 **42,243 帧**，全部成功配对，得到 **0.02725%**。

这里的百分比描述固定多相机跑台录制。比较不同录制时，可结合毫米误差和参考路径长度阅读，因为即便毫米误差相近，录制时长和骨盆累计路径也会改变 RTE。用 WHAM 原版函数对相同轨迹复算，各段结果的最大绝对差小于 0.000001 个百分点。

## 四、数据署名

Aaron Cherian，*FreeMoCap Validation Dataset: Multi-Camera Markerless Motion Capture of Treadmill Gait and Standing Balance*，v1.0。随附轨迹从原始数据中提取，已按帧配对并换算单位，沿用 CC BY 4.0 许可。评分与结果表由本示例计算。[原始数据](https://doi.org/10.5281/zenodo.23041223) · [FreeMoCap 验证项目](https://github.com/freemocap/validation) · [CC BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)

# 六段跑台步行录制的骨盆轨迹评测

本例使用 FreeMoCap 验证数据集 v1.0。选择编号最小的三名参与者（sub-001 至 sub-003），每人取两次跑台步行录制。按编号和动作选取试次，每次都使用完整的已发布轨迹。视频侧使用 FreeMoCap 的 MediaPipe 跟踪结果，参考侧使用同次录制的 Qualisys 光学动捕。数据作者发布了同步和空间配准后的三维轨迹，原始视频未随这次数据发布。[数据及使用许可](https://doi.org/10.5281/zenodo.23041223) · [验证研究说明](https://github.com/freemocap/validation)

## 运行

在本目录运行以下命令。随附的 `pelvis_tracks.npz` 包含六次录制的帧号、参考骨盆坐标和估计骨盆坐标，评分可离线完成。

```bash
python3 -m pip install numpy
python3 evaluate.py --csv results.csv
```

[逐段结果表](results.csv) 包含总帧数、计分帧数、参考骨盆累计路径（米）、平均三维位置误差（毫米）和根轨迹误差 RTE（%）。

| 参与者 / 录制 | 配对帧数 | 平均位置误差（mm） | RTE（%） |
|---|---:|---:|---:|
| 001 / 01 | 7,887 | 15.83 | 0.02421 |
| 001 / 02 | 8,497 | 20.34 | 0.02748 |
| 002 / 01 | 5,617 | 21.96 | 0.04039 |
| 002 / 02 | 6,115 | 18.70 | 0.02929 |
| 003 / 01 | 6,903 | 13.48 | 0.02432 |
| 003 / 02 | 7,224 | 12.87 | 0.02113 |

共计 42,243 帧，全部成功配对。六次录制等权平均为 **17.20 mm、0.02780% RTE**；按各次计分帧数加权为 **0.02725% RTE**。用 WHAM 原版 `compute_rte` 对相同坐标复算，六次结果与本实现的最大绝对差小于 0.000001 个百分点。

## 如何计分

1. 用每次录制的帧号逐一匹配两套系统的 `hips_center`。两边都将骨盆定义为左右髋中心的中点，原始单位为毫米。[MediaPipe 关键点定义](https://github.com/freemocap/validation/blob/main/validation/utils/freemocap_model_info/mediapipe_model_info.yaml) · [Qualisys 关键点定义](https://github.com/freemocap/validation/blob/main/qualisys_markerset/qualisys_model_info.yaml)
2. 对每段独立拟合一次旋转和平移，使两条骨盆轨迹对齐，尺度固定为 1。然后计算同一帧两个骨盆位置间的三维距离，取这些距离的算术平均。
3. 用该段的平均距离除以参考骨盆全段相邻帧的累计路径长度，再乘以 100%。该长度来自实验室坐标系下骨盆的运动，包括上下和左右起伏。跑台带面走过的里程不参与计算。[RTE 公式参考实现](https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/eval/eval_utils.py#L464-L482)
4. 主结果取六个百分比的算术平均，每次录制权重相同。脚本另外报告按有效帧数加权的均值，便于对比汇总方式。

这是对固定多相机跑台数据新增的一项评分。各段 RTE 同时受误差、时长和骨盆累计路径影响，请结合毫米误差阅读。综述表 8 使用移动单目 EMDB2 数据、其他预测方法和其相应评测协议。

## 从官方数据重新提取

从数据页下载 `sub-001`、`sub-002`、`sub-003` 三个 ZIP，按官方 SHA256SUMS.txt 校验，解压到同一个 `data` 目录。在本目录执行：

```bash
python3 -m pip install numpy pyarrow
python3 prepare_tracks.py data --output pelvis_tracks.npz
python3 evaluate.py --csv results.csv
```

每条输入仅取官方 Parquet 文件中 `trajectory=3d_xyz`、`keypoint=hips_center` 的记录，按整数帧号排序，再将毫米换算为米。归档中每个数组有七列，依次为帧号、参考 x/y/z、视频估计 x/y/z。脚本检查两侧帧号连续且完全对应。

### 数据署名

Aaron Cherian，*FreeMoCap Validation Dataset: Multi-Camera Markerless Motion Capture of Treadmill Gait and Standing Balance*，v1.0。随附的骨盆轨迹摘自该数据，按试次配对并换算单位后分发，沿用 CC BY 4.0 许可。RTE 及结果表由本示例计算。[原始数据](https://doi.org/10.5281/zenodo.23041223) · [CC BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)

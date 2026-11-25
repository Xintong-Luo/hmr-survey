# 在一段 EMDB 实拍序列上计算 GLAMR 的根轨迹误差

在 EMDB 作者发布的室外行走示例上，GLAMR 的预测与参考骨盆轨迹相差 **19.19% RTE**。本例使用作者随论文图 9 提供的既有预测，在 CPU 上完成评分。数据来自 P9 的 `80_outdoor_walk_big_circle`，人物编号 0，使用该文件从 0 到 2239 的全部 2,240 帧。[EMDB 官方示例](https://github.com/eth-ait/emdb/tree/9a4eab677181a3789bda7ba5c36ab8cff797380c#visualize-glamr)

## 运行

在本目录执行以下命令。脚本从 EMDB 官方仓库下载约 12.4 MB 的固定版本样本，并校验 SHA-256。评分只依赖 NumPy。

```bash
python3 -m pip install numpy
python3 evaluate.py --output result.json
```

若已下载同一版本的 `P9_80_seed1.pkl`，可离线运行。加上 `--plot` 会生成轨迹图。

```bash
python3 -m pip install matplotlib
python3 evaluate.py --input P9_80_seed1.pkl --output result.json --plot trajectories.png
```

## 结果怎样读

| 量 | 本次运行 |
|---|---:|
| 参考骨盆轨迹走过的总长度 | 106.3839 m |
| 对齐后逐帧三维骨盆距离的均值 | 20.4173 m |
| 根轨迹误差（RTE） | **19.1920%** |

计算时从示例文件读取同一人物、同一帧的参考坐标 `gt[0].root_trans` 和预测坐标 `person_data[0].root_trans_world`。把整段预测轨迹向参考轨迹做一次刚体对齐，允许旋转和平移，尺度固定为 1。对齐后，计算每一帧的三维欧氏距离并取平均。最后除以参考轨迹相邻帧之间距离的总和，再乘以 100%。例如本次运行得到 20.4173 ÷ 106.3839 × 100% ≈ 19.19%。[WHAM 的 RTE 计算函数](https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/eval/eval_utils.py#L464-L482)

![左图为骨盆轨迹在世界坐标 X–Z 平面的投影，右图为逐帧三维误差。](trajectories.png)

左图中两条轨迹共用坐标原点，圆点标出各自的第一帧。右图中虚线是整段均值。可在 [result.json](result.json) 查看完整精度和样本校验值。

## 核验与适用范围

- 本次计算用 Python 3.12.3 与 NumPy 2.5.3。将同两条轨迹输入 WHAM 的原版 `compute_rte`（PyTorch 2.6.0，float32），得到 19.192026%；本脚本用 float64 得到 19.192046%，两者相差约 0.000020 个百分点。
- 该数值用于演示综述表 8 的 RTE 计算。参考轨迹取自公开示例内部的骨盆根关节，帧范围也由该示例确定。表 8 的全测试集结果另有数据划分、有效帧筛选和模型平移量定义，两者应分开报告。[EMDB 示例读取程序](https://github.com/eth-ait/emdb/blob/9a4eab677181a3789bda7ba5c36ab8cff797380c/visualize_GLAMR.py) · [WHAM 完整评测入口](https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/eval/evaluate_emdb.py)
- 样本由 EMDB 作者发布，预测方法为 GLAMR。下载和使用该样本请参阅上游说明。[EMDB](https://eth-ait.github.io/emdb/) · [GLAMR](https://github.com/NVlabs/GLAMR) · [示例附带的许可](https://github.com/eth-ait/emdb/blob/9a4eab677181a3789bda7ba5c36ab8cff797380c/assets/GLAMR/LICENSE.txt)

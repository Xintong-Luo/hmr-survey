# 表 7 · HaMeR 在 FreiHAND 上的手部网格评测

| 测试集 | 指标（越小越好） | HaMeR 原论文表 1 | 综述表 7 | 本例实测 | 保留一位小数 |
| --- | --- | ---: | ---: | ---: | ---: |
| FreiHAND v2 全部 3,960 张测试图 | PA-MPVPE（mm） | 5.7 | 5.7 | 5.705628 | **5.7** |

第一次运行建议选 **A**，用本仓库保存的逐图预测在 CPU 上重新评分。要检验从图像到结果的整个过程，选 **B**，用 GPU 重新生成预测，再运行相同的评分脚本。[原论文表 1](https://arxiv.org/pdf/2312.05251#page=5) · [本例原始结果](result.json)

## A. 用已有预测复算表 7（CPU）

**1. 下载两个文件。** 以下命令均在本示例目录运行。若从仓库首页开始，先执行：

```bash
cd evaluation/freihand_table7
mkdir -p data models runs
```

| 要准备的文件 | 下载与放置 |
| --- | --- |
| FreiHAND 测试图与参考顶点 | 下载 **Evaluation set with annotations**（约 724 MiB），保存为 `data/FreiHAND_pub_v2_eval.zip`。评分时直接读取压缩包。[官方 ZIP](https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip) · [数据集说明](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) |
| MANO v1.2 右手模型 | 登录 MANO 官网，点 **Download → Models & Code**，下载 `mano_v1_2.zip`。解压，从 `mano_v1_2/models/` 取出 `MANO_RIGHT.pkl`，放到本例的 `models/MANO_RIGHT.pkl`。[官方下载页](https://mano.is.tue.mpg.de/download.php) |

逐图预测已经随本例提供，文件是 [predictions/hamer_freihand_params.npz](predictions/hamer_freihand_params.npz)（约 2.2 MiB）。它保存图像编号、手部旋转、形状和相机参数。评分时用 MANO 生成每张图的 778 个预测顶点，再读取 FreiHAND 顶点标注。

**2. 安装 CPU 评分环境。** 以下为 Linux x86-64、Python 3.10 的命令。已配置好该环境的读者可直接进入第 3 步。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.3.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation -r requirements-score.txt
```

**3. 运行评分。** 结果会打印到终端，并写入 `runs/from_saved_predictions.json`。

```bash
OMP_NUM_THREADS=2 python evaluate.py \
  --reference data/FreiHAND_pub_v2_eval.zip \
  --parameters predictions/hamer_freihand_params.npz \
  --mano models/MANO_RIGHT.pkl \
  --output runs/from_saved_predictions.json
```

## B. 从测试图像重新推理（GPU）

需要 A 中的两个下载文件、一块支持 CUDA 的 NVIDIA GPU，以及下列 HaMeR 资源。本例使用作者发布的预训练权重。本次复测使用 Tesla T4（16 GB 显存），批大小为 16。

**1. 安装推理环境。** 从本示例目录新开一个终端，使用独立环境。此环境也可以执行 A 的评分命令。

```bash
python3.10 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install pip==24.3.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install --no-build-isolation -r requirements.txt
```

**2. 下载代码、权重和测试清单。** 代码和权重均固定到本次复测的版本。`freihand_val.npz` 提供与作者评测一致的图像顺序和裁剪框。[HaMeR 官方评测说明](https://github.com/geopavlakos/hamer#evaluation) · [作者发布的权重与配置](https://huggingface.co/spaces/geopavlakos/HaMeR/tree/aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a/_DATA/hamer_ckpts)

```bash
mkdir -p vendor models data runs
git clone https://github.com/geopavlakos/hamer.git vendor/hamer
git -C vendor/hamer checkout 3a01849f4148352e9260b69bf28b65d1671a4905

weights_url='https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a/_DATA/hamer_ckpts'
curl -fL --retry 3 "$weights_url/checkpoints/hamer.ckpt" -o models/hamer.ckpt
curl -fL --retry 3 "$weights_url/model_config.yaml" -o models/model_config.yaml
curl -fL --retry 3 'https://www.dropbox.com/scl/fi/7ip2vnnu355e2kqbyn1bc/hamer_evaluation_data.tar.gz?rlkey=nb4x10uc8mj2qlfq934t5mdlh&dl=1' -o data/hamer_evaluation_data.tar.gz

tar -xzf data/hamer_evaluation_data.tar.gz -C data hamer_evaluation_data/freihand_val.npz
unzip -q -o data/FreiHAND_pub_v2_eval.zip 'evaluation/rgb/*.jpg' -d data
```

**3. 生成预测，再评分。** 第一个命令读取 `data/evaluation/rgb/` 下的 3,960 张图像，第二个命令用新生成的预测计算 PA-MPVPE。

```bash
python predict_parameters.py \
  --repo vendor/hamer \
  --checkpoint models/hamer.ckpt --config models/model_config.yaml \
  --meta data/hamer_evaluation_data/freihand_val.npz --rgb data/evaluation/rgb \
  --output runs/hamer_freihand_params.npz

OMP_NUM_THREADS=2 python evaluate.py \
  --reference data/FreiHAND_pub_v2_eval.zip \
  --parameters runs/hamer_freihand_params.npz --mano models/MANO_RIGHT.pkl \
  --output runs/from_images.json
```

如果使用 HaMeR 官方 `eval.py --dataset FREIHAND-VAL` 得到了 `freihand-val.json`，也可以把完整的顶点预测直接传入评分脚本：

```bash
python evaluate.py --reference data/FreiHAND_pub_v2_eval.zip \
  --submission freihand-val.json --output runs/from_submission.json
```

## 怎样与表 7 对比

先确认输出中 `images` 为 **3960**、`vertices_per_image` 为 **778**，然后读取 **`pa_mpvpe_mm`**。每张图单独按 FreiHAND 官方方法对齐预测网格与参考网格，求对应顶点的平均欧氏距离，再对全部图像取平均。脚本已把米换算成毫米，可以直接与表 7 的 **HaMeR / 独立手部 FreiHAND / PA-MPVPE** 一行比较。[FreiHAND 原始评分代码](https://github.com/lmb-freiburg/freihand/blob/master/eval.py)

我们从发布包解压后的目录运行了两条路线（Linux，Python 3.10.22，PyTorch 2.6.0）。

| 运行方式 | 对应输出文件 | PA-MPVPE（mm） | 按表 7 精度 |
| --- | --- | ---: | ---: |
| A · CPU 对随包预测重新评分 | `runs/from_saved_predictions.json` | 5.705628416 | **5.7** |
| B · Tesla T4 从 RGB 重新推理后评分 | `runs/from_images.json` | 5.705628359 | **5.7** |

两条路线均与表 7 一致。使用 FreiHAND 原始评分函数独立核验随包预测，同样得到 **5.7056 mm**。[随包预测的参考输出](result.json)

## 引用

- Pavlakos 等，*Reconstructing Hands in 3D with Transformers*，CVPR 2024（HaMeR 方法、权重与论文结果）。[论文](https://arxiv.org/abs/2312.05251)
- Zimmermann 等，*FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images*，ICCV 2019（数据与评测协议）。[数据集与引用信息](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)
- Romero、Tzionas、Black，*Embodied Hands: Modeling and Capturing Hands and Bodies Together*，SIGGRAPH Asia 2017（MANO 手部模型）。[项目与引用信息](https://mano.is.tue.mpg.de/) · [模型许可](https://mano.is.tue.mpg.de/license.html)

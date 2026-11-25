# 固定与移动相机下的三维人体网格重建综述

*Three-dimensional human mesh reconstruction with fixed and moving cameras: a survey*

![固定相机与移动相机从不同视角观测三维人体网格的概念示意](assets/survey-cover.jpg)

<a id="contents"></a>

## 目录

| 栏目 | 常用入口 |
| --- | --- |
| [**一、算法与模型**](#algorithms) | [单目重建](#alg-body) · [移动相机](#alg-moving) · [多目重建](#alg-multi) |
| [**二、数据集**](#datasets) | [按任务查找](#ds-tasks) · [训练与实拍数据](#ds-more) · [测试集与协议](#data-by-table) |
| [**三、评价指标**](#metrics) | [局部骨架与网格](#metrics-local) · [世界运动与接触](#metrics-world) · [相机轨迹](#metrics-camera) |
| [**四、评测示例**](#evaluation) | [手部网格误差](evaluation/freihand_table7/README.md) · [单段轨迹](evaluation/emdb_glamr_rte/README.md) · [多段轨迹](evaluation/freemocap_multivideo_rte/README.md) |
| [**五、延伸阅读**](#reading) | 人体网格综述、人体化身与群组空间布局 |

<a id="algorithms"></a>

## 一、算法与模型

按重建任务展开条目，可查看每种方法的年份、主要用途、原论文和作者资源。

<a id="alg-foundation"></a>

### 1. 人体表示与姿态估计

<a id="models"></a>

<details>
<summary><strong>人体模型与网格标注</strong>（12 项）</summary>

| 模型或工具 | 年份 | 用途 | 论文 | 实现、模型或数据 |
| --- | --- | --- | --- | --- |
| SCAPE | 2005 | 分别学习个体体型差异与姿态引起的局部形变 | [原论文](https://doi.org/10.1145/1073204.1073207) | [作者项目与扫描数据申请](https://ai.stanford.edu/~drago/Projects/scape/scape.html) |
| SMPL | 2015 | 以体型、关节旋转和姿态修正驱动统一拓扑的身体网格 | [原论文](https://doi.org/10.1145/2816795.2818013) | [模型下载](https://smpl.is.tue.mpg.de/) · [SMPL 系列加载代码](https://github.com/vchoutas/smplx) |
| MANO / SMPL+H | 2017 | MANO 描述手部形状与姿态，HaMeR 复测使用该模型。SMPL+H 将它与身体模型结合 | [原论文](https://doi.org/10.1145/3130800.3130883) | [模型与示例代码，注册下载](https://mano.is.tue.mpg.de/download.php) |
| FLAME | 2017 | 共同表达头部身份形状、表情、颈部和下颌等关节运动 | [原论文](https://doi.org/10.1145/3130800.3130813) | [模型下载](https://flame.is.tue.mpg.de/) · [TF_FLAME 实现与示例](https://github.com/TimoBolkart/TF_FLAME) |
| SMPL-X | 2019 | 将身体、双手和面部接入同一拓扑和运动链 | [原论文](https://doi.org/10.1109/CVPR.2019.01123) | [模型与下载](https://smpl-x.is.tue.mpg.de/) · [代码](https://github.com/vchoutas/smplx) |
| STAR | 2020 | 在身体模型中学习稀疏、局部的姿态形变 | [原论文](https://doi.org/10.1007/978-3-030-58539-6_36) | [代码](https://github.com/ahmedosman/STAR) · [模型](https://star.is.tue.mpg.de/) |
| GHUM / GHUML | 2020 | 在非线性形状空间中联合表示全身、双手与面部 | [原论文](https://doi.org/10.1109/CVPR42600.2020.00622) | [作者资源与模型申请表](https://github.com/google-research/google-research/tree/master/ghum) |
| SUPR | 2022 | 学习全身模型及可单独使用的身体部件模型 | [原论文](https://doi.org/10.1007/978-3-031-20086-1_33) | [代码](https://github.com/ahmedosman/SUPR) · [模型](https://supr.is.tue.mpg.de/) |
| SKEL | 2023 | 通过生物力学骨架驱动身体表面 | [原论文](https://doi.org/10.1145/3618381) | [作者实现与模型获取说明](https://github.com/MarilynKeller/SKEL) |
| 周敏等的 SMPL 引导残缺点云重建 | 2026 | 输入为不完整点云，结合语义、多尺度特征和顶点偏移细化网格 | [期刊原文](https://www.cjig.cn/zh/article/doi/10.11834/jig.250262/) | 暂未找到作者公开的代码链接 |
| MoSh / MoSh++ | 2014 / 2019 | 从动捕标记拟合体型与动作。MoSh 提供 Human3.6M 的一套参考网格，MoSh++ 用于构建 AMASS | [MoSh](https://doi.org/10.1145/2661229.2661273) · [MoSh++ / AMASS](https://amass.is.tue.mpg.de/) | [MoSh 项目](https://is.mpg.de/en/code/mosh-motion-and-shape-capture-from-sparse-markers) · [MoSh++ 代码](https://github.com/nghorbani/moshpp) |
| NeuralAnnot | 2022 | 为 InterHand2.6M、Human3.6M 等数据集生成参数化网格拟合标注 | [原论文](https://arxiv.org/abs/2011.11232) | [作者发布的拟合标注及使用说明](https://github.com/mks0601/NeuralAnnot_RELEASE) |

网格示意素材可从 [SOMA-X（MANO、SMPL-X）](https://github.com/NVlabs/SOMA-X/blob/main/docs/data_assets.md) 和 [TF_FLAME（FLAME）](https://github.com/TimoBolkart/TF_FLAME) 获取。

</details>

<a id="monocular"></a>

<details>
<summary><strong>三维姿态、定位与关联</strong>（17 项）</summary>

| 方法 | 年份 | 任务 | 原论文 | 代码与项目 |
| --- | ---: | --- | --- | --- |
| Martinez 等（2D-to-3D baseline） | 2017 | 从二维关键点提升根相对三维关节 | [论文](https://doi.org/10.1109/iccv.2017.288) | [代码](https://github.com/una-dinosauria/3d-pose-baseline) |
| Integral Regression | 2018 | 对关节热图求期望，获得连续坐标 | [论文](https://doi.org/10.1007/978-3-030-01231-1_33) | [代码](https://github.com/JimmySuen/integral-human-pose) |
| HEMlets Pose | 2019 | 用相邻关节的相对深度监督三维姿态 | [论文](https://doi.org/10.1109/iccv.2019.00243) | [推理代码](https://github.com/redrock303/HEMlets) |
| ContextPose | 2021 | 用骨长关系约束多视图特征聚合 | [论文](https://doi.org/10.1109/cvpr46437.2021.00617) | [代码](https://github.com/ShirleyMaxx/ContextPose-PyTorch-release) |
| HiPART | 2025 | 按层级自回归生成稠密骨架，缓解遮挡 | [论文](https://doi.org/10.1109/cvpr52734.2025.01566) | 暂未找到作者公开的代码链接 |
| Orientation Keypoints | 2022 | 预测骨段的虚拟定向点，恢复轴向旋转 | [论文](https://doi.org/10.1109/tpami.2021.3136136) | 暂未找到作者公开的代码链接 |
| VideoPose3D | 2019 | 用时间卷积从二维序列恢复三维关节 | [论文](https://doi.org/10.1109/cvpr.2019.00794) | [代码](https://github.com/facebookresearch/VideoPose3D) |
| MixSTE | 2022 | 交替编码关节间和跨帧的依赖 | [论文](https://doi.org/10.1109/cvpr52688.2022.01288) | [代码](https://github.com/JinluZhang1126/MixSTE) |
| MotionBERT | 2023 | 用掩码训练获得可迁移的运动表示 | [论文](https://doi.org/10.1109/iccv51070.2023.01385) | [代码](https://github.com/Walter0807/MotionBERT) |
| Contact and Human Dynamics | 2020 | 结合足接触与动力学优化单目人体运动 | [论文](https://doi.org/10.1007/978-3-030-58558-7_5) | [代码](https://github.com/davrempe/contact-human-dynamics) |
| RootNet | 2019 | 结合人体框与焦距估计根关节深度 | [论文](https://doi.org/10.1109/iccv.2019.01023) | [代码](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE) |
| MeTRAbs | 2021 | 融合二维关节和度量骨架，估计相机系绝对姿态 | [论文](https://doi.org/10.1109/tbiom.2020.3037257) | [代码](https://github.com/isarandi/metrabs) |
| SMAP | 2020 | 在整幅图像中同时估计多人的根深度和相对关节深度 | [论文](https://doi.org/10.1007/978-3-030-58555-6_33) | [代码](https://github.com/zju3dv/SMAP) |
| PHALP | 2022 | 用外观、三维位置和姿态保持跨帧身份 | [论文](https://doi.org/10.1109/cvpr52688.2022.00276) | [代码](https://github.com/brjathu/PHALP) |
| Learnable Triangulation | 2019 | 结合可学习权重或体积特征进行多视图三角化 | [论文](https://doi.org/10.1109/iccv.2019.00781) | [代码](https://github.com/karfly/learnable-triangulation-pytorch) |
| VoxelPose | 2020 | 在共享体素空间定位多人并预测三维关节 | [论文](https://doi.org/10.1007/978-3-030-58452-8_12) | [代码](https://github.com/microsoft/voxelpose-pytorch) |
| MVGFormer | 2024 | 交替更新多视图二维关节和三维姿态 | [论文](https://doi.org/10.1109/cvpr52733.2024.00074) | [代码](https://github.com/XunshanMan/MVGFormer) |

</details>

<a id="alg-body"></a>

### 2. 单目人体重建

<a id="single-image"></a>

<details>
<summary><strong>单图身体回归与概率建模</strong>（29 项）</summary>

| 方法 | 年份 | 任务 | 原论文 | 代码与项目 |
| --- | ---: | --- | --- | --- |
| HMR | 2018 | 迭代回归 SMPL 姿态、体型和弱透视参数 | [论文](https://doi.org/10.1109/cvpr.2018.00744) | [代码](https://github.com/akanazawa/hmr) |
| HMR 2.0 / 4DHumans | 2023 | 以视觉 Transformer 恢复局部网格，并用于人物跟踪 | [论文](https://doi.org/10.1109/iccv51070.2023.01358) | [代码](https://github.com/shubham-goel/4D-Humans) |
| GraphCMR | 2019 | 沿网格拓扑卷积，直接回归顶点 | [论文](https://doi.org/10.1109/cvpr.2019.00463) | [代码](https://github.com/nkolot/GraphCMR) |
| Pose2Mesh | 2020 | 用二维姿态逐层恢复网格 | [论文](https://doi.org/10.1007/978-3-030-58571-6_45) | [代码](https://github.com/hongsukchoi/Pose2Mesh_RELEASE) |
| DecoMR | 2020 | 借助图像与表面的稠密对应回归网格 | [论文](https://doi.org/10.1109/cvpr42600.2020.00708) | [代码](https://github.com/zengwang430521/DecoMR) |
| I2L-MeshNet | 2020 | 用一维位置分布估计关节和网格顶点 | [论文](https://doi.org/10.1007/978-3-030-58571-6_44) | [代码](https://github.com/mks0601/I2L-MeshNet_RELEASE) |
| METRO | 2021 | 用 Transformer 建模关节与顶点的非局部关系 | [论文](https://doi.org/10.1109/cvpr46437.2021.00199) | [代码](https://github.com/microsoft/MeshTransformer) |
| Mesh Graphormer | 2021 | 融合图卷积和注意力预测人体网格 | [论文](https://doi.org/10.1109/iccv48922.2021.01270) | [代码](https://github.com/microsoft/MeshGraphormer) |
| FastMETRO | 2022 | 用交叉注意力连接图像特征与几何表示 | [论文](https://doi.org/10.1007/978-3-031-19769-7_20) | [代码](https://github.com/postech-ami/FastMETRO) |
| HybrIK | 2021 | 解析求摆动并学习扭转，使关节估计驱动网格 | [论文](https://doi.org/10.1109/cvpr46437.2021.00339) | [代码](https://github.com/jeffffffli/HybrIK) |
| NIKI | 2023 | 用可逆神经网络建立关节与旋转的映射 | [论文](https://doi.org/10.1109/cvpr52729.2023.01243) | [代码](https://github.com/jeffffffli/NIKI) |
| Pavlakos 等（关节与轮廓） | 2018 | 借助关节热图和人体轮廓恢复姿态及体型 | [论文](https://doi.org/10.1109/cvpr.2018.00055) | [作者项目](https://geopavlakos.github.io/projects/humanshape/)；暂未找到作者公开的代码链接 |
| HoloPose | 2019 | 融合身体部件和稠密图像证据估计人体 | [论文](https://doi.org/10.1109/cvpr.2019.01114) | 暂未找到作者公开的代码链接 |
| DenseRaC | 2019 | 以稠密姿态渲染与观测比较来估计参数 | [论文](https://doi.org/10.1109/iccv.2019.00785) | 暂未找到作者公开的代码链接 |
| DaNet | 2022 | 分解身体部件特征，再聚合以回归姿态与体型 | [论文](https://doi.org/10.1109/TPAMI.2020.3042341) | [代码](https://github.com/HongwenZhang/DaNet-3DHumanReconstruction) |
| DensePose | 2018 | 估计图像像素与人体表面之间的稠密对应 | [论文](https://doi.org/10.1109/cvpr.2018.00762) | [代码](https://github.com/facebookresearch/DensePose) |
| PARE | 2021 | 用部件注意力提取遮挡条件下的图像特征 | [论文](https://doi.org/10.1109/iccv48922.2021.01094) | [代码](https://github.com/mkocabas/PARE) |
| PyMAF | 2021 | 将当前网格投影到多尺度特征图，迭代细化参数 | [论文](https://doi.org/10.1109/iccv48922.2021.01125) | [代码](https://github.com/HongwenZhang/PyMAF) |
| ReFit | 2023 | 用循环拟合逐次校正预测网格 | [论文](https://doi.org/10.1109/iccv51070.2023.01346) | [代码](https://github.com/yufu-wang/ReFit) |
| Global-to-Pixel | 2024 | 动态匹配局部像素与人体结构，对全局预测作残差修正 | [论文](https://doi.org/10.1007/978-3-031-72640-8_27) | 暂未找到作者公开的代码链接 |
| 跨表示对齐（Gong 等） | 2022 | 联合稀疏关键点和稠密对应细化人体 | [论文](https://doi.org/10.1007/978-3-031-19769-7_13) | 暂未找到作者公开的代码链接 |
| SPIN | 2019 | 用回归初始化拟合，并将拟合结果用于训练 | [论文](https://doi.org/10.1109/iccv.2019.00234) | [代码](https://github.com/nkolot/SPIN) |
| TokenHMR | 2024 | 用离散姿态表示和自适应损失减轻投影偏差 | [论文](https://doi.org/10.1109/cvpr52733.2024.00132) | [代码](https://github.com/saidwivedi/TokenHMR) |
| ProHMR | 2021 | 建模给定图像时的人体姿态条件分布 | [论文](https://doi.org/10.1109/iccv48922.2021.01140) | [代码](https://github.com/nkolot/ProHMR) |
| 层级运动学概率模型（Sengupta 等） | 2021 | 沿人体运动链学习各关节的旋转概率分布 | [论文](https://doi.org/10.1109/iccv48922.2021.01103) | [代码](https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman) |
| HuManiFlow | 2023 | 在旋转流形上按祖先关节条件估计姿态与体型分布 | [论文](https://doi.org/10.1109/cvpr52729.2023.00463) | [代码](https://github.com/akashsengupta1997/HuManiFlow) |
| LAPP / ProPose | 2023 | 从骨方向构造人体网格估计的解析后验 | [论文](https://doi.org/10.1109/cvpr52729.2023.00848) | [代码](https://github.com/NetEase-GameAI/ProPose) |
| DPMesh | 2024 | 用预训练扩散模型的特征增强遮挡人体表示 | [论文](https://doi.org/10.1109/cvpr52733.2024.00111) | [代码](https://github.com/RAMMUSLeo/DPMesh) |
| MEGA | 2025 | 通过掩码生成式自编码器恢复人体网格 | [论文](https://doi.org/10.1109/cvpr52734.2025.00505) | [代码](https://github.com/g-fiche/MEGA) |

</details>

<a id="video"></a>

<details>
<summary><strong>视频身体回归与测试时自适应</strong>（13 项）</summary>

| 方法 | 年份 | 任务 | 原论文 | 代码与项目 |
| --- | ---: | --- | --- | --- |
| HMMR | 2019 | 从单目裁剪序列学习三维人体动态 | [论文](https://doi.org/10.1109/cvpr.2019.00576) | [代码](https://github.com/akanazawa/human_dynamics) |
| VIBE | 2020 | 用时序编码器和动作判别器约束视频网格 | [论文](https://doi.org/10.1109/cvpr42600.2020.00530) | [代码](https://github.com/mkocabas/VIBE) |
| TCMR | 2021 | 融合当前帧两侧的信息，改善网格时间一致性 | [论文](https://doi.org/10.1109/cvpr46437.2021.00200) | [代码](https://github.com/hongsukchoi/TCMR_RELEASE) |
| DTS-VIBE | 2022 | 融合 RGB 和光流，并通过跨帧对应约束视频预测 | [论文](https://doi.org/10.1109/wacv51458.2022.00071) | 暂未找到作者公开的代码链接 |
| 部件动态模型（Lee 和 Lee） | 2021 | 分别预测不同身体部件的姿态变化及不确定性 | [论文](https://doi.org/10.1109/iccv48922.2021.01215) | 暂未找到作者公开的代码链接 |
| MAED | 2021 | 用多层时空注意力与运动链解码估计视频姿态和体型 | [论文](https://doi.org/10.1109/iccv48922.2021.01279) | [代码](https://github.com/ziniuwan/maed) |
| TFNet | 2023 | 用频域特征引导时序聚合 | [论文](https://doi.org/10.1109/icassp49357.2023.10095498) | [作者仓库（仅发布说明）](https://github.com/Changboyang/TFNet) |
| MEVA | 2021 | 先压缩动作得到平滑估计，再以图像特征恢复细节 | [论文](https://doi.org/10.1007/978-3-030-69541-5_20) | [代码](https://github.com/ZhengyiLuo/MEVA) |
| DiffMesh | 2025 | 在扩散过程中建模相邻帧的网格特征变化 | [论文](https://doi.org/10.1109/wacv61041.2025.00479) | [代码](https://github.com/zczcwh/DiffMesh) |
| PMCE | 2023 | 协同更新骨架与网格，用视频特征补充体型 | [论文](https://doi.org/10.1109/iccv51070.2023.01374) | [代码](https://github.com/kasvii/PMCE) |
| BOA | 2021 | 在目标视频上执行双层在线适应 | [论文](https://doi.org/10.1109/cvpr46437.2021.01033) | [代码](https://github.com/syguan96/BOA) |
| DynaBOA | 2023 | 动态调整双层在线适应过程 | [论文](https://doi.org/10.1109/tpami.2022.3194167) | [代码](https://github.com/syguan96/DynaBOA) |
| CycleAdapt | 2023 | 循环更新图像回归器与序列去噪器 | [论文](https://doi.org/10.1109/iccv51070.2023.01362) | [代码](https://github.com/hygenie1228/CycleAdapt_RELEASE) |

</details>

<a id="fitting"></a>

<details>
<summary><strong>人体模型拟合与优化</strong>（13 项）</summary>

| 方法 | 年份 | 任务 | 原论文 | 代码与项目 |
| --- | ---: | --- | --- | --- |
| Guan 等（单图拟合） | 2009 | 结合单图关键点与轮廓估计人体姿态、体型 | [论文](https://doi.org/10.1109/iccv.2009.5459300) | 暂未找到作者公开的代码链接 |
| SMPLify | 2016 | 以二维关节和先验分阶段拟合 SMPL | [论文](https://doi.org/10.1007/978-3-319-46454-1_34) | [模型与拟合代码（注册）](https://smplify.is.tue.mpg.de/) |
| UP-3D | 2017 | 以模型拟合生成图像的三维人体标注 | [论文](https://doi.org/10.1109/cvpr.2017.500) | [代码](https://github.com/classner/up) |
| EFT | 2021 | 对单个样本微调预训练回归器，改善人体拟合 | [论文](https://doi.org/10.1109/3dv53792.2021.00015) | [代码](https://github.com/facebookresearch/eft) |
| SCOPE | 2021 | 利用运动树的稀疏结构求解人体参数更新 | [论文](https://doi.org/10.1109/iccv48922.2021.01126) | [代码](https://github.com/fantaosha/SCOPE) |
| KAMA | 2021 | 由相机空间关节解析求解身体关节旋转和整体位姿 | [论文](https://doi.org/10.1109/3dv53792.2021.00078) | [作者项目](https://research.nvidia.com/publication/2021-12_kama-3d-keypoint-aware-body-mesh-articulation)；暂未找到作者公开的代码链接 |
| IKOL | 2023 | 以可微 Gauss–Newton 逆运动学层协调骨架与网格 | [论文](https://doi.org/10.1609/aaai.v37i3.25454) | [代码](https://github.com/Juzezhang/IKOL) |
| PLIKS | 2023 | 结合相机内参与顶点投影进行伪线性逆运动学求解 | [论文](https://doi.org/10.1109/cvpr52729.2023.00063) | [代码](https://github.com/karShetty/PLIKS) |
| 学习梯度下降（Song 等） | 2020 | 根据观测误差的梯度学习人体模型更新步长与方向 | [论文](https://doi.org/10.1007/978-3-030-58565-5_44) | [代码](https://github.com/InpatientJam/Learned-Gradient-Descent) |
| HUND | 2021 | 通过部件和关键点对齐误差循环优化 GHUM 状态 | [论文](https://doi.org/10.1109/cvpr46437.2021.01425) | [推理代码](https://github.com/google-research/google-research/tree/master/ghum/inference/hund) |
| SMPLify-DC | 2021 | 在拟合中加入自接触约束，减少身体穿透 | [论文](https://doi.org/10.1109/cvpr46437.2021.00986) | [代码](https://github.com/muelea/tuch) |
| Pose-NDF | 2022 | 用神经距离场限制可行人体姿态 | [论文](https://doi.org/10.1007/978-3-031-20065-6_33) | [代码](https://github.com/garvita-tiwari/PoseNDF) |
| ScoreHMR | 2024 | 以图像重投影误差引导扩散模型细化姿态 | [论文](https://doi.org/10.1109/cvpr52733.2024.00092) | [代码](https://github.com/statho/ScoreHMR) |

</details>

<a id="alg-whole"></a>

### 3. 手、脸与全身重建

<a id="parts"></a>

<details>
<summary><strong>独立手部与面部恢复</strong>（17 项）</summary>

| 方法 | 年份 | 任务 | 原论文 | 代码与项目 |
| --- | ---: | --- | --- | --- |
| Boukhayma 等（单手 MANO 回归） | 2019 | 从单图恢复手部参数并用关节和掩码监督 | [论文](https://doi.org/10.1109/cvpr.2019.01110) | [代码](https://github.com/boukhayma/3dhand) |
| Hand Graph CNN（Ge 等） | 2019 | 回归顶点图像坐标及根相对深度 | [论文](https://doi.org/10.1109/cvpr.2019.01109) | [代码](https://github.com/3d-hand-shape/hand-graph-cnn) |
| MobRecon | 2022 | 用关节特征提升和深度可分螺旋卷积解码手部网格 | [论文](https://doi.org/10.1109/cvpr52688.2022.01989) | [代码](https://github.com/SeanChenxy/HandMesh) |
| HandOccNet | 2022 | 为手物遮挡引入特征注入和自注意力 | [论文](https://doi.org/10.1109/cvpr52688.2022.00155) | [代码](https://github.com/namepllet/HandOccNet) |
| IntagHand | 2022 | 以双手交互注意力和图卷积恢复双手网格 | [论文](https://doi.org/10.1109/cvpr52688.2022.00278) | [代码](https://github.com/Dw1010/IntagHand) |
| InterWild | 2023 | 分开恢复单手形状与双手相对平移 | [论文](https://doi.org/10.1109/cvpr52729.2023.01633) | [代码](https://github.com/facebookresearch/InterWild) |
| InterNet（InterHand2.6M 基线） | 2020 | 从单图估计双手姿态与根部相对深度 | [论文](https://doi.org/10.1007/978-3-030-58565-5_33) | [代码](https://github.com/facebookresearch/InterHand2.6M) |
| HaMeR | 2024 | 从单手图像回归 MANO 姿态与形状 | [论文](https://doi.org/10.1109/cvpr52733.2024.00938) | [代码](https://github.com/geopavlakos/hamer) |
| 3DDFA | 2016 | 用三维可变形模型回归大姿态人脸的稠密对齐 | [论文](https://doi.org/10.1109/cvpr.2016.23) | [作者主页所列实现](https://github.com/cleardusk/3DDFA) |
| 3DDFA-V2 | 2020 | 以投影反馈提高稠密人脸对齐的稳定性与速度 | [论文](https://doi.org/10.1007/978-3-030-58529-7_10) | [代码](https://github.com/cleardusk/3DDFA_V2) |
| PRNet | 2018 | 用 UV 位置图编码人脸顶点的三维坐标 | [论文](https://doi.org/10.1007/978-3-030-01264-9_33) | [代码](https://github.com/YadiraF/PRNet) |
| FaceVerse | 2022 | 以粗细两层模型刻画人脸几何细节 | [论文](https://doi.org/10.1109/cvpr52688.2022.01969) | [代码](https://github.com/LizhenWangT/FaceVerse) |
| RingNet | 2019 | 学习同身份图像间的一致性以估计形状和表情 | [论文](https://doi.org/10.1109/cvpr.2019.00795) | [代码](https://github.com/soubhiksanyal/RingNet) |
| MICA | 2022 | 结合身份特征与统一扫描监督估计度量人脸形状 | [论文](https://doi.org/10.1007/978-3-031-19778-9_15) | [代码](https://github.com/Zielon/MICA) |
| DECA | 2021 | 结合身份细节和表情参数生成 UV 位移 | [论文](https://doi.org/10.1145/3450626.3459936) | [代码](https://github.com/yfeng95/DECA) |
| HiFace | 2023 | 用静态位移图和动态位移图恢复表情细节 | [论文](https://doi.org/10.1109/iccv51070.2023.00834) | [作者项目](https://project-hiface.github.io/)；暂未找到作者公开的代码链接 |
| EMOCA | 2022 | 在 DECA 上加入情绪感知的表情约束 | [论文](https://doi.org/10.1109/cvpr52688.2022.01967) | [代码](https://github.com/radekd91/emoca) |

</details>

<a id="whole-body"></a>

<details>
<summary><strong>统一全身人体恢复</strong>（10 项）</summary>

| 方法 | 年份 | 任务 | 原论文 | 代码与项目 |
| --- | ---: | --- | --- | --- |
| SMPLify-X / VPoser | 2019 | 用全身关键点、姿态先验与碰撞约束拟合 SMPL-X | [论文](https://doi.org/10.1109/cvpr.2019.01123) | [拟合代码](https://github.com/vchoutas/smplify-x) · [VPoser](https://github.com/nghorbani/human_body_prior) |
| ExPose | 2020 | 由全身估计引导手部和面部的高分辨率裁剪 | [论文](https://doi.org/10.1007/978-3-030-58607-2_2) | [代码](https://github.com/vchoutas/expose) |
| PIXIE | 2021 | 按置信度融合身体、手部和面部专家特征 | [论文](https://doi.org/10.1109/3dv53792.2021.00088) | [代码](https://github.com/yfeng95/PIXIE) |
| Hand4Whole | 2022 | 以掌指关节特征改善全身模型的腕部方向 | [论文](https://doi.org/10.1109/cvprw56347.2022.00257) | [代码](https://github.com/mks0601/Hand4Whole_RELEASE) |
| PyMAF-X | 2023 | 通过全身网格对齐与肘腕旋转协调手身连接 | [论文](https://doi.org/10.1109/tpami.2023.3271691) | [代码](https://github.com/HongwenZhang/PyMAF-X) |
| OSX | 2023 | 采用部件感知 Transformer 进行单阶段全身恢复 | [论文](https://doi.org/10.1109/cvpr52729.2023.02027) | [代码](https://github.com/IDEA-Research/OSX) |
| HybrIK-X | 2025 | 以混合逆运动学连接全身关节与参数模型旋转 | [论文](https://doi.org/10.1109/tpami.2025.3528979) | [代码](https://github.com/jeffffffli/HybrIK) |
| SMPLer-X | 2023 | 扩大训练数据和模型规模以恢复表达性人体 | [论文](https://doi.org/10.52202/075280-0506) | [代码](https://github.com/MotrixLab/SMPLer-X) |
| SMPLest-X | 2026 | 用任务标记解码器回归表达性人体 | [论文](https://doi.org/10.1109/tpami.2025.3618174) | [代码](https://github.com/MotrixLab/SMPLest-X) |
| DPoser-X | 2025 | 用联合扩散先验约束身体、手部与面部姿态 | [论文](https://doi.org/10.1109/iccv51701.2025.00931) | [代码](https://github.com/moonbow721/DPoser-X) |

</details>

<a id="alg-moving"></a>

### 4. 移动相机与世界坐标重建

<a id="cameras"></a>

<details>
<summary><strong>相机感知的单目网格</strong>（9 项）</summary>

| 方法 | 年份 | 任务 | 论文 | 代码或项目 |
| --- | --- | --- | --- | --- |
| SPEC | 2021 | 从全图估计视场、俯仰和滚转，作为人体回归的相机条件 | [论文](https://doi.org/10.1109/iccv48922.2021.01085) | [代码](https://github.com/mkocabas/SPEC) |
| CLIFF | 2022 | 把人体框位置和尺度关联回全图，并在全图坐标中施加重投影监督 | [论文](https://doi.org/10.1007/978-3-031-20065-6_34) | [代码](https://github.com/huawei-noah/noah-research/tree/master/CLIFF) |
| CameraHMR（HumanFoV、CamSMPLify） | 2025 | 利用人体框、焦距以及透视拟合生成的伪标签恢复身体 | [论文](https://doi.org/10.1109/3dv66043.2025.00146) | [代码](https://github.com/pixelite1201/CameraHMR) |
| PersPose | 2025 | 将裁剪内参编码为像素射线，用虚拟旋转校正离轴投影 | [论文](https://doi.org/10.1109/ICCV51701.2025.00760) | [代码](https://github.com/KenAdamsJoseph/PersPose) |
| Multi-RoI | 2024 | 对同一人体使用多个裁剪，约束各裁剪的相机预测在全图中一致 | [论文](https://doi.org/10.1007/978-3-031-72970-6_25) | [代码](https://github.com/CptDiaos/Multi-RoI) |
| Zolly | 2023 | 利用透视失真与表面对应估计近摄人体深度及网格 | [论文](https://doi.org/10.1109/ICCV51070.2023.00363) | [代码](https://github.com/MotrixLab/Zolly) |
| BLADE | 2025 | 以估计的骨盆深度约束单图人体，并用轮廓匹配求焦距与平移 | [论文](https://doi.org/10.1109/cvpr52734.2025.02048) | [代码](https://github.com/NVlabs/blade) |
| D&D | 2022 | 在动态相机参照中建模惯性项，通过接触与受约束动力学更新运动 | [论文](https://doi.org/10.1007/978-3-031-20065-6_28) | [代码](https://github.com/jeffffffli/DnD) |
| Human Mesh Recovery from Multiple Shots | 2022 | 在镜头切换时分开整体朝向与内部姿态，约束跨镜头动作连续性 | [论文](https://doi.org/10.1109/cvpr52688.2022.00154) | [代码](https://github.com/geopavlakos/multishot) |

</details>

<a id="world-motion"></a>

<details>
<summary><strong>人体运动与相机轨迹联合估计</strong>（17 项）</summary>

| 方法 | 年份 | 任务 | 论文 | 代码或项目 |
| --- | --- | --- | --- | --- |
| Camera Motion Agnostic | 2022 | 由局部关节旋转序列估计整体旋转和平移增量，累积人体轨迹 | [论文](https://doi.org/10.3390/s22207975) | 暂未找到作者公开的代码链接 |
| TRACE | 2023 | 以全图时序特征、光流与运动偏移恢复首帧相机参照下的多人轨迹 | [论文](https://doi.org/10.1109/cvpr52729.2023.00855) | [代码](https://github.com/Arthur151/ROMP) |
| ProxyCap | 2024 | 在人体坐标中预测姿态、相机与接触，再用图像及足部反馈修正运动 | [论文](https://doi.org/10.1109/cvpr52733.2024.00191) | 暂未找到作者公开的代码链接 |
| WHAM | 2024 | 结合人体运动、相机角速度和足部接触恢复世界运动 | [论文](https://doi.org/10.1109/cvpr52733.2024.00202) | [代码](https://github.com/yohanshin/WHAM) |
| GVHMR | 2024 | 在重力与观察方向构成的参照中预测人体，利用相机旋转与静止肢体修正轨迹 | [论文](https://doi.org/10.1145/3680528.3687565) | [代码](https://github.com/zju3dv/GVHMR) |
| WHAC | 2024 | 以人体运动定标视觉里程计平移，联合恢复度量相机与人体轨迹 | [论文](https://doi.org/10.1007/978-3-031-72754-2_2) | [代码](https://github.com/MotrixLab/WHAC) |
| TRAM（含 VIMO） | 2024 | 用人体遮罩和预测度量深度定标背景 SLAM，将局部人体变换到世界坐标 | [论文](https://doi.org/10.1007/978-3-031-73247-8_27) | [代码](https://github.com/yufu-wang/tram) |
| HAC | 2025 | 以人体与场景接触点的深度关系估计相机平移尺度 | [论文](https://doi.org/10.1109/ICCV51701.2025.00573) | [代码](https://github.com/MartaYang/HumansAsCheckerboards) |
| MBTI | 2025 | 按实际采样时刻处理不同帧率，以掩码视觉 SLAM 建立世界参照 | [论文](https://doi.org/10.1109/iccv51701.2025.01076) | 暂未找到作者公开的代码链接 |
| PromptHMR / PromptHMR-Vid | 2025 | 通过提示选择人物，视频版结合时序解码、相机轨迹和场景深度 | [论文](https://doi.org/10.1109/cvpr52734.2025.00115) | [代码](https://github.com/yufu-wang/PromptHMR) |
| HumanMM | 2025 | 分别估计镜头内运动，借切换帧人体关键点连接镜头间相机旋转 | [论文](https://doi.org/10.1109/CVPR52734.2025.00190) | [作者仓库（数据入口，代码待发布）](https://github.com/zhangyuhong01/HumanMM-code) |
| PhysHMR | 2025 | 利用二维关节对应的世界射线，驱动物理控制策略恢复人体运动 | [论文](https://doi.org/10.1145/3757377.3763951) | [代码](https://github.com/fengq1a0/PhysHMR) |
| GLAMR | 2022 | 补全遮挡动作并优化根部轨迹与相机外参 | [论文](https://doi.org/10.1109/cvpr52688.2022.01076) | [代码](https://github.com/NVlabs/GLAMR) |
| SLAHMR | 2023 | 利用背景 SLAM、人体运动先验、地面与接触优化世界运动 | [论文](https://doi.org/10.1109/cvpr52729.2023.02033) | [代码](https://github.com/vye16/slahmr) |
| HuMoR | 2021 | 学习人体运动状态转移先验，为视频拟合及 SLAHMR 提供约束 | [论文](https://doi.org/10.1109/iccv48922.2021.01129) | [代码](https://github.com/davrempe/humor) |
| PACE | 2024 | 用人体动作场和背景点重投影分阶段优化人体与逐帧相机位姿 | [论文](https://doi.org/10.1109/3dv62453.2024.00103) | [作者项目](https://nvlabs.github.io/PACE/) · 暂未找到作者公开的代码链接 |
| BodySLAM | 2022 | 把人体运动模型纳入相机定位与场景建图 | [论文](https://doi.org/10.1007/978-3-031-19842-7_38) | [作者项目](https://srl.cit.tum.de/research/projects/bodyslam) · 暂未找到作者公开的代码链接 |

</details>

<a id="interaction"></a>

<details>
<summary><strong>多人跟踪与人景交互</strong>（19 项）</summary>

| 方法 | 年份 | 任务 | 论文 | 代码或项目 |
| --- | --- | --- | --- | --- |
| 4DHumans / HMR 2.0 | 2023 | 单图网格回归与跨帧多人跟踪，用于世界运动与人景交互评测的基线对比 | [论文](https://doi.org/10.1109/iccv51070.2023.01358) | [代码](https://github.com/shubham-goel/4D-Humans) |
| Human3R | 2026 | 同步恢复人体、场景点图和相机参数，并用人体标记库维持跨帧身份 | [论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/467c017e2639255e9c5d91a2e582f95a-Abstract-Conference.html) | [代码](https://github.com/fanegg/Human3R) |
| MultiPhys | 2024 | 在同一物理场景同步控制多人，细化运动、碰撞与足地接触 | [论文](https://doi.org/10.1109/cvpr52733.2024.00226) | [代码](https://github.com/nicolasugrinovic/multiphys) |
| MultiPly | 2024 | 利用分层观测、深度次序和非穿透协调多人重建 | [论文](https://doi.org/10.1109/cvpr52733.2024.00019) | [代码](https://github.com/eth-ait/MultiPly) |
| PROX | 2019 | 将人体与扫描场景对齐，通过接触和非穿透改善身体放置 | [论文](https://doi.org/10.1109/iccv.2019.00237) | [代码](https://github.com/mohamedhassanmus/prox) |
| PhysCap | 2020 | 在标定相机和已知地面下，用物理跟踪与接触预测细化动作 | [论文](https://doi.org/10.1145/3414685.3417877) | [作者演示代码](https://github.com/soshishimada/PhysCap_demo_release) |
| NeuralPhysCap | 2021 | 通过可微控制与动力学循环恢复物理一致的运动 | [论文](https://doi.org/10.1145/3450626.3459825) | [作者演示代码](https://github.com/soshishimada/Neural_Physcap_Demo) |
| 轨迹优化（Gärtner 等） | 2022 | 用适配人体尺寸的物理角色优化单目三维动作轨迹 | [论文](https://doi.org/10.1109/cvpr52688.2022.01276) | [作者项目](https://gartner.io/trajectory/) · 暂未找到作者公开的代码链接 |
| Neural MoCon | 2022 | 用神经运动控制结合给定的场景几何估计人体动作 | [论文](https://doi.org/10.1109/cvpr52688.2022.00631) | [作者出版目录](https://www.buzhenhuang.com/publications/) · 暂未找到作者公开的代码链接 |
| 软接触动力学（Xie 等） | 2021 | 通过可微物理和接触力模型从视频估计、合成动作 | [论文](https://doi.org/10.1109/iccv48922.2021.01133) | [作者项目](https://research.nvidia.com/labs/toronto-ai/physics-pose-estimation-project-page/) · 暂未找到作者公开的代码链接 |
| PhysDynPose | 2025 | 以人体及相机初值、地形高度图和参考足接触细化运动 | [论文](https://doi.org/10.1109/cvprw67362.2025.00374) | [代码](https://github.com/aidilayce/PhysDynPose) |
| PHOSA | 2020 | 结合身体、物体模板、遮挡和接触，估计单图人—物空间关系 | [论文](https://doi.org/10.1007/978-3-030-58610-2_3) | [代码](https://github.com/facebookresearch/phosa) |
| HolisticMesh | 2021 | 从图像估计人体、物体与室内布局，再优化空间关系 | [论文](https://doi.org/10.1109/cvpr46437.2021.00040) | [代码](https://github.com/ZZWENG/holistic_mesh) |
| PhySIC | 2025 | 估计遮挡场景并协调人体与场景尺度，约束接触与支撑 | [论文](https://doi.org/10.1145/3757377.3763862) | [代码](https://github.com/YuxuanSnow/Phy-SIC) |
| MOVER | 2022 | 在固定相机下利用人体运动产生的遮挡、自由空间和接触线索恢复布局 | [论文](https://doi.org/10.1109/cvpr52688.2022.00393) | [代码](https://github.com/yhw-yhw/MOVER) |
| SynCHMR | 2024 | 用人体尺度校准建图深度，再用时序场景点云细化世界网格 | [论文](https://doi.org/10.1109/CVPR52733.2024.00122) | [作者项目](https://paulchhuang.github.io/synchmr/) · 暂未找到作者公开的代码链接 |
| DUSt3R | 2024 | 恢复多视图共享点图，为人景联合建模提供场景几何 | [论文](https://doi.org/10.1109/cvpr52733.2024.01956) | [代码](https://github.com/naver/dust3r) |
| MASt3R | 2024 | 在三维点图上学习稠密匹配特征，为多视图关联提供对应 | [论文](https://doi.org/10.1007/978-3-031-73220-1_5) | [代码](https://github.com/naver/mast3r) |
| VGGT | 2025 | 从多视图图像预测相机参数、深度、点图与点轨迹 | [论文](https://doi.org/10.1109/cvpr52734.2025.00499) | [代码](https://github.com/facebookresearch/vggt) |

</details>

<a id="alg-multi"></a>

### 5. 多目人体重建

<a id="multiview"></a>

<details>
<summary><strong>标定多目、参数融合与序列拟合</strong>（13 项）</summary>

| 方法 | 年份 | 任务 | 论文 | 代码或项目 |
| --- | --- | --- | --- | --- |
| LMT / LT-fitting | 2023 | LMT 融合可见顶点并拟合 SMPL。对照方法 LT-fitting 先对关节作三角化，再拟合人体模型 | [论文](https://doi.org/10.1109/wacv56688.2023.00287) | [作者项目](https://sites.google.com/view/learnable-mesh-triangulation/home) · 暂未找到作者公开的代码链接 |
| Vertex Heatmap Representation Learning | 2023 | 用 AMASS 学习高维顶点热图的潜在表示，约束多视图网格 | [论文](https://doi.org/10.1109/icip49359.2023.10222297) | 暂未找到作者公开的代码链接 |
| Shape-Aware HPS | 2019 | 跨视图共享局部姿态和体型，分别估计每幅图像的朝向与投影参数 | [论文](https://doi.org/10.1109/iccv.2019.00445) | [代码](https://github.com/williamljb/HumanMultiView) |
| PaFF | 2023 | 将 SMPL 顶点重投影至各视图，迭代融合像素对齐反馈 | [论文](https://doi.org/10.1609/aaai.v37i1.25179) | [作者仓库（准备脚本）](https://github.com/Kairobo/PaFF) |
| Progressive Multi-view HMR | 2023 | 将多视图观测汇入人体规范坐标，利用热图与轮廓渐进重建 | [论文](https://doi.org/10.1609/aaai.v37i1.25144) | 暂未找到作者公开的代码链接 |
| Mesh-aligned Voxel Interpolation | 2025 | 沿初始网格采样多尺度体素特征，分别细化定位与人体形态 | [论文](https://doi.org/10.1016/j.inffus.2024.102651) | 暂未找到作者公开的代码链接 |
| MuVS | 2017 | 结合多视图关节和轮廓拟合 SMPL，在序列中共享体型并平滑轨迹 | [论文](https://doi.org/10.1109/3dv.2017.00055) | [代码](https://github.com/YinghaoHuang91/MuVS) |
| Shape-aware Multi-Person | 2021 | 用可靠二维观测、三维关节候选和身体形态交替更新多人骨架与网格 | [论文](https://doi.org/10.1109/iccv48922.2021.01097) | [作者仓库（代码待发布）](https://github.com/zj-dong/Multi-Person-Pose-Estimation) |
| BundleMoCap | 2023 | 用稀疏潜在关键帧及插值表示多目动作，降低序列优化成本 | [论文](https://doi.org/10.1145/3626495.3626511) | [作者项目](https://moverseai.github.io/bundle/) · 暂未找到作者公开的代码链接 |
| BundleMoCap++ | 2024 | 引入球面姿态先验、流形更新和高阶插值以改进多目序列拟合 | [论文](https://doi.org/10.1016/j.cviu.2024.104190) | 暂未找到作者公开的代码链接 |
| HeatFormer | 2025 | 学习对齐观测热图与当前模型热图，分别评测不同视图配置 | [论文](https://doi.org/10.1109/cvpr52734.2025.00601) | [代码](https://github.com/kyotovision-public/HeatFormer) |
| Monocular Models are Strong Learners | 2026 | 在逐视图特征和虚拟视图参数之间交替更新，结合解剖点重投影细化人体 | [论文](https://doi.org/10.1007/978-3-032-36839-3_10) | 暂未找到作者公开的代码链接 |
| Training-free Multi-view 4D | 2026 | 在标定相机参考下融合单目全身预测，用几何和跟踪恢复多人序列 | [论文](https://doi.org/10.1109/wacv61042.2026.00039) | [代码](https://github.com/Att100/MV4DHMR) |

</details>

<a id="moving-multiview"></a>

<details>
<summary><strong>精细多目、人体辅助标定与移动多目</strong>（11 项）</summary>

| 方法 | 年份 | 任务 | 论文 | 代码或项目 |
| --- | --- | --- | --- | --- |
| DiffProxy | 2026 | 生成分割与 UV 代理观测，按跨视图分歧加权拟合 SMPL-X | [论文](https://doi.org/10.1007/978-3-032-37098-3_17) | [代码](https://github.com/wrk226/DiffProxy) |
| Total Capture（Adam） | 2018 | 联合多视图三角化关键点与立体点云，拟合身体、双手和面部 | [论文](https://doi.org/10.1109/cvpr.2018.00868) | [作者项目](https://www.cs.cmu.edu/~hanbyulj/totalcapture/) · 暂未找到作者公开的代码链接 |
| Lightweight Total Capture | 2021 | 用身体定位引导手脸关联，在稀疏视图下恢复多人的完整运动 | [论文](https://doi.org/10.1109/iccv48922.2021.00551) | 暂未找到作者公开的代码链接 |
| AvatarPose | 2024 | 利用逐人外观与轮廓约束多视图近距离交互姿态 | [论文](https://doi.org/10.1007/978-3-031-73668-1_13) | [代码](https://github.com/eth-ait/AvatarPose) |
| Dynamic Multi-Person Mesh Recovery（DMMR） | 2021 | 以人体关节初始化外参，联合优化未标定多目下的多人动作与相机 | [论文](https://doi.org/10.1109/3dv53792.2021.00080) | [代码](https://github.com/boycehbz/DMMR) |
| Human Semantics | 2024 | 用直立人体及预测身高估计相机，再联合细化人体和机位 | [论文](https://doi.org/10.1109/tcsvt.2023.3328371) | [代码](https://github.com/boycehbz/DMMR) |
| EasyRet3D | 2025 | 结合跨视图轨迹、自动标定、地面接触和时序 SMPL 优化 | [论文](https://doi.org/10.1109/wacv61041.2025.00309) | 暂未找到作者公开的代码链接 |
| HSfM | 2025 | 用人体统计尺度与场景点图联合恢复同步时刻的人体、相机和场景 | [论文](https://doi.org/10.1109/cvpr52734.2025.02044) | [代码](https://github.com/hongsukchoi/HSfM_RELEASE) |
| AirCap | 2019 | 利用多架无人机的图像及机载位姿初值联合优化相机与身体 | [论文](https://doi.org/10.1109/iccv.2019.00091) | [作者飞行系统代码](https://github.com/robot-perception-group/AirCap) |
| AirPose / AirPose+ | 2022 | 在飞行相机间共享局部构型，分别估计整体放置，并可经序列优化细化 | [论文](https://doi.org/10.1109/lra.2022.3145494) | [代码](https://github.com/robot-perception-group/AirPose) |
| SmartMocap | 2023 | 联合估计固定与移动相机组合中的相机轨迹和人体运动 | [论文](https://doi.org/10.1109/lra.2023.3264743) | [代码](https://github.com/robot-perception-group/SmartMocap) |

</details>

<a id="alg-tools"></a>

### 6. 基础工具与代表方法

<a id="geometry"></a>

<details>
<summary><strong>几何估计与跟踪</strong>（8 项）</summary>

| 方法 | 年份 | 任务 | 论文 | 代码或项目 |
| --- | --- | --- | --- | --- |
| 张正友平面标定 | 2000 | 由多幅平面标定板图像估计相机内参和畸变 | [论文](https://doi.org/10.1109/34.888718) | [OpenCV 实现示例](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) |
| 束调整 | 2000 | 通过重投影残差联合细化相机与三维结构 | [论文](https://doi.org/10.1007/3-540-44480-7_21) | [Ceres 实现示例](https://github.com/ceres-solver/ceres-solver/blob/master/examples/bundle_adjuster.cc) |
| DROID-SLAM | 2021 | 以稠密对应与束调整估计相机运动，相关人体方法还会屏蔽动态区域 | [论文](https://arxiv.org/abs/2108.10869) | [代码](https://github.com/princeton-vl/DROID-SLAM) |
| DPVO | 2023 | 用学习的图像块迭代估计相机运动，在视觉里程计配置中为 HMR 2.0、WHAM、GVHMR 提供相机运动估计 | [论文](https://arxiv.org/abs/2208.04726) | [代码](https://github.com/princeton-vl/DPVO) |
| SORT（3D 跟踪改写） | 2016 | Li 等（2026）将 SORT 改用于三维人体包围框的跨帧关联 | [论文](https://doi.org/10.1109/ICIP.2016.7533003) | [原始二维 SORT 代码](https://github.com/abewley/sort) |
| One Euro Filter | 2012 | 根据运动速度自适应调节平滑强度，用于三维跟踪输出 | [论文](https://doi.org/10.1145/2207676.2208639) | [代码](https://github.com/casiez/OneEuroFilter) |
| Graph-Based | 2021 | 将多视图人体关节关联建模为图，用于多人三维姿态恢复 | [论文](https://doi.org/10.1109/iccv48922.2021.01096) | [代码](https://github.com/wusize/multiview_pose) |
| Faster VoxelPose | 2022 | 用正交投影加速体素空间的多人三维姿态恢复 | [论文](https://doi.org/10.1007/978-3-031-20068-7_9) | [代码](https://github.com/AlvinYH/Faster-VoxelPose) |

</details>

<a id="components"></a>

<details>
<summary><strong>投影几何与实验组件</strong>（4 项）</summary>

| 组件 | 用途 | 论文或文档 | 实现 |
| --- | --- | --- | --- |
| Ray3D（2022） | 从相机内参构造三维射线，并结合外参恢复人体绝对位置 | [原论文](https://doi.org/10.1109/CVPR52688.2022.01277) | [作者代码](https://github.com/YxZhxn/Ray3D) |
| ResNet-50 / ResNet-152 | MobRecon 使用 ResNet-50 提取图像特征，LMT 使用 ResNet-152 | [ResNet 原论文](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) | [作者实现](https://github.com/KaimingHe/deep-residual-networks) |
| ViT | WHAM 评测中使用的一种视觉编码器 | [ViT 原论文](https://research.google/pubs/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/) | [作者代码](https://github.com/google-research/vision_transformer) |
| ARKit | 设备跟踪提供参考相机旋转，可与纯视觉估计的旋转对照 | [Apple 世界跟踪文档](https://developer.apple.com/documentation/arkit/understanding-world-tracking) | [ARCamera 接口](https://developer.apple.com/documentation/arkit/arcamera) |

</details>

<a id="timeline"></a>

<details>
<summary><strong>代表方法与发表年份</strong>（5 项）</summary>

| 方法 | 年份 | 任务 | 论文 | 作者发布 |
| --- | --- | --- | --- | --- |
| CoMotion | 2025 | 在单目视频中同时检测、跟踪多人并恢复其三维运动 | [ICLR 论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cdc3507fa48f85dd2dc7cf12cc47239f-Abstract-Conference.html) | [推理演示与权重](https://github.com/apple-aiml-research/ml-comotion) |
| SAM 3D Body | 2025 发布 / 2026 发表 | 单幅图像中的全身网格恢复，涵盖身体、手和脚 | [CVPR 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_SAM_3D_Body_Robust_Full-Body_Human_Mesh_Recovery_CVPR_2026_paper.html) | [推理代码、权重与数据入口](https://github.com/facebookresearch/sam-3d-body) |
| OnlineHMR | 2026 | 从视频流逐步恢复世界坐标系中的人体运动 | [CVPR 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_OnlineHMR_Video-based_Online_World-Grounded_Human_Mesh_Recovery_CVPR_2026_paper.html) | [代码](https://github.com/Tsukasane/Video-OnlineHMR) |
| HumanBA | 2026 | 在束调整中分解相机与人体运动，联合优化二者 | [CVPR 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_HumanBA_Human-Aware_Bundle_Adjustment_via_Global_Human-Camera_Decoupling_CVPR_2026_paper.html) | [作者仓库，核对时为空](https://github.com/MartaYang/HumanBA) |
| UniSH | 2026 | 在一次前馈推理中统一重建场景与人体 | [CVPR 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_UniSH_Unifying_Scene_and_Human_Reconstruction_in_a_Feed-Forward_Pass_CVPR_2026_paper.html) | [代码与权重](https://github.com/murphylmf/UniSH) |

</details>

[↑ 返回目录](#contents)

<a id="datasets"></a>

## 二、数据集

各条目列出采集方式、可用标注、数据论文，以及下载或申请入口。

<a id="ds-tasks"></a>

### 1. 按重建任务查找

<a id="data-body"></a>

<details>
<summary><strong>身体、手部与面部</strong>（7 项）</summary>

| 资源 | 观测、标注与用途 | 原始论文 | 数据获取与工具 |
| --- | --- | --- | --- |
| AGORA | 合成多人、遮挡与扫描配准人体。按注册协议获取数据，测试预测通过评测服务器提交。 | [论文](https://doi.org/10.1109/cvpr46437.2021.01326) | [注册下载](https://agora.is.tue.mpg.de/download.php) · [评测代码](https://github.com/pixelite1201/agora_evaluation) |
| BEDLAM | CVPR 2023 版本。合成动作视频、人体网格与相机参数，可按相机配置取子集。 | [论文](https://doi.org/10.1109/cvpr52729.2023.00843) | [注册下载](https://bedlam.is.tue.mpg.de/download.php) · [训练与评测](https://github.com/pixelite1201/BEDLAM) · [渲染工具](https://github.com/PerceivingSystems/bedlam_render) |
| FreiHAND | 多目采集的单手数据。v2 完整测试包包含 3,960 张图像和公开顶点参考。 | [论文](https://doi.org/10.1109/iccv.2019.00090) | [数据主页](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) · [测试包及标注](https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip) · [评测代码](https://github.com/lmb-freiburg/freihand) |
| InterHand2.6M | 单手与交互双手。交互双手评测采用 v1.0 的 5 fps 子集和后续 MANO 拟合标注。 | [论文](https://doi.org/10.1007/978-3-030-58565-5_33) | [图像与标注](https://mks0601.github.io/InterHand2.6M/) · [基线与评测](https://github.com/facebookresearch/InterHand2.6M) · [NeuralAnnot 拟合](https://github.com/mks0601/NeuralAnnot_RELEASE) |
| NoW | 单张面部图像与中性头部扫描，用于身份形状评估。下载页区分验证参考与测试提交。 | [论文](https://doi.org/10.1109/cvpr.2019.00795) | [注册下载](https://now.is.tue.mpg.de/download.php) · [评测代码](https://github.com/soubhiksanyal/now_evaluation) |
| EHF | 表达性全身图像、扫描与 SMPL-X 注册表面。登录后取 EHF 评测数据。 | [论文](https://doi.org/10.1109/cvpr.2019.01123) | [SMPL-X 项目下载](https://smpl-x.is.tue.mpg.de/download.php) · [模型读取](https://github.com/vchoutas/smplx) |
| UBody | 真实上半身视频与 SMPL-X 伪标签，可检验裁剪及遮挡条件下的部件恢复。 | [论文](https://doi.org/10.1109/cvpr52729.2023.02027) | [作者申请表](https://docs.google.com/forms/d/e/1FAIpQLSehgBP7wdn_XznGAM2AiJPiPLTqXXHw5uX9l7qeQ1Dh9HoO_A/viewform) · [OSX 数据处理与评测](https://github.com/IDEA-Research/OSX) |

</details>

<a id="data-motion"></a>

<details>
<summary><strong>移动相机与世界运动</strong>（3 项）</summary>

| 资源 | 观测、标注与用途 | 原始论文 | 数据获取与工具 |
| --- | --- | --- | --- |
| 3DPW | 手持视频、惯性辅助拟合人体和相机位姿。按任务区分局部构型误差和全局误差。 | [论文](https://doi.org/10.1007/978-3-030-01249-6_37) | [下载与协议](https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html) · [挑战赛评测代码](https://github.com/miraymen/3dpw-eval) |
| EMDB（EMDB1 / EMDB2） | 电磁跟踪与手持 RGB-D。EMDB1 用于评估局部身体重建，EMDB2 用于评估世界坐标系中的人体运动。 | [论文](https://doi.org/10.1109/iccv51070.2023.01345) | [官方申请](https://emdb.ait.ethz.ch/) · [数据说明](https://eth-ait.github.io/emdb/) · [评测工具](https://github.com/eth-ait/emdb) |
| MoviCam | 移动相机人体、相机与地形参考，含 8 段序列。评测按平地和非平地分组。 | [论文](https://doi.org/10.1109/cvprw67362.2025.00374) | [原始数据与配套文件](https://doi.org/10.17617/3.DGK9OM) · [下载与目录说明](https://github.com/aidilayce/PhysDynPose/blob/main/DATASET.md) · [代码](https://github.com/aidilayce/PhysDynPose) |

</details>

<a id="data-multiview"></a>

<details>
<summary><strong>同步多目与多人交互</strong>（3 项）</summary>

| 资源 | 观测、标注与用途 | 原始论文 | 数据获取与工具 |
| --- | --- | --- | --- |
| HuMMan | 多目 RGB-D、手机采集、人体参数及扫描。作者分子集发布，重建任务选 HuMMan-Recon。 | [论文](https://doi.org/10.1007/978-3-031-20071-7_33) | [重建子集与工具](https://caizhongang.com/projects/HuMMan/recon.html) · [作者数据仓库](https://huggingface.co/datasets/caizhongang/HuMMan) |
| Hi4D | 同步多目记录双人近距离互动，提供逐人注册网格与相对位置。 | [论文](https://doi.org/10.1109/cvpr52729.2023.01632) | [申请下载](https://hi4d.ait.ethz.ch/) · [读取与可视化](https://github.com/yifeiyin04/Hi4D) |
| EgoHumans | 多人自视点与外部视点、多目拟合人体和身份关联。HSfM 使用同步外部视图。 | [论文](https://doi.org/10.1109/iccv51070.2023.01814) | [数据与测试标注](https://github.com/rawalkhirodkar/egohumans/blob/main/assets/DOWNLOAD.md) · [处理代码](https://github.com/rawalkhirodkar/egohumans) |

</details>

<a id="data-scene"></a>

<details>
<summary><strong>人景交互与运动先验</strong>（3 项）</summary>

| 资源 | 观测、标注与用途 | 原始论文 | 数据获取与工具 |
| --- | --- | --- | --- |
| RICH | 多目拟合人体、独立场景扫描及几何接触。相机真值的覆盖按具体配置核对。 | [论文](https://doi.org/10.1109/cvpr52688.2022.01292) | [注册下载](https://rich.is.tue.mpg.de/download.php) · [数据工具](https://github.com/paulchhuang/rich_toolkit) |
| BEHAVE | 多目 RGB-D 下的人与物体交互，含双方几何、接触及对应注册。 | [论文](https://doi.org/10.1109/cvpr52688.2022.01547) | [许可与下载](https://virtualhumans.mpi-inf.mpg.de/behave/license.html) · [作者代码](https://github.com/bharat-b7/BEHAVE) |
| AMASS | 多源标记点动捕经 MoSh++ 统一拟合为人体模型运动，主要用作时序先验；无配套实拍 RGB。 | [论文](https://doi.org/10.1109/iccv.2019.00554) | [注册后按来源下载](https://amass.is.tue.mpg.de/download.php) · [使用工具](https://github.com/nghorbani/amass) |

</details>

<a id="ds-more"></a>

### 2. 训练数据与多机位实拍

<a id="data-training"></a>

<details>
<summary><strong>网格监督、训练与外部评测</strong>（12 项）</summary>

| 资源 | 观测、标注与用途 | 原始论文 | 数据获取与工具 |
| --- | --- | --- | --- |
| UP-3D | 将图像和 SMPL 拟合配对，作为网格训练监督。 | [论文](https://doi.org/10.1109/cvpr.2017.500) | [数据包与许可](https://files.is.tuebingen.mpg.de/classner/up/#datasets) · [作者代码](https://github.com/classner/up) |
| SURREAL | 合成动作、RGB 与三维人体标注，需同意许可并取得下载凭据。 | [论文](https://doi.org/10.1109/cvpr.2017.492) | [申请与下载](https://www.di.ens.fr/willow/research/surreal/data/) · [生成与下载脚本](https://github.com/gulvarol/surreal) |
| ARCTIC | 双手操作关节物体的多视角记录。外部相机评测使用 p1 协议。 | [论文](https://doi.org/10.1109/cvpr52729.2023.01244) | [注册下载](https://arctic.is.tue.mpg.de/download.php) · [设置与评测](https://github.com/zc-alexfan/arctic) |
| Human3.6M | 受控多目动作与三维关节。网格评测使用 MoSh 或 NeuralAnnot 生成的参考网格，两套拟合应分别比较。 | [论文](https://doi.org/10.1109/tpami.2013.248) | [官方注册与访问](http://vision.imar.ro/human3.6m/description.php) · [NeuralAnnot 拟合](https://github.com/mks0601/NeuralAnnot_RELEASE) |
| HUMBI | 密集同步视图，提供身体、手和脸等部件的分别拟合结果。 | [论文](https://doi.org/10.1109/cvpr42600.2020.00306) | [数据主页](https://humbi-data.net/) · [大学授权入口](https://license.umn.edu/product/humbi-dataset---high-resolution-multiview-images-of-human-body-expressions) · [可视化工具](https://github.com/zhixuany/HUMBI) |
| CHI3D | 近距离双人交互及接触。多人评测采用原作选定的序列。使用后续 SMPL-X 拟合时需注明版本。 | [论文](https://doi.org/10.1109/cvpr42600.2020.00724) | [申请与数据下载](https://ci3d.imar.ro/download) · [数据说明](https://ci3d.imar.ro/chi3d) |
| Ego-Exo4D | 第一、第三人称同步技能活动。单时刻相机布局评测仅使用同步的外部视图。 | [论文](https://doi.org/10.1109/cvpr52733.2024.01834) | [申请与配置](https://docs.ego-exo4d-data.org/getting-started/) · [下载工具](https://docs.ego-exo4d-data.org/download/) |
| ImageNet | 图像分类数据集，用于 MobRecon 的预训练。 | [论文](https://doi.org/10.1109/CVPR.2009.5206848) | [官方下载](https://www.image-net.org/download) · [ILSVRC 数据与工具](https://www.image-net.org/challenges/LSVRC/) |
| COCO-WholeBody | COCO 图像上的全身二维关键点标注，供 OSX 训练。原始图像和附加标注分别获取。 | [论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/768_ECCV_2020_paper.php) | [全身标注与工具](https://github.com/jin-s13/COCO-WholeBody) · [COCO 图像](https://cocodataset.org/#download) |
| MPII Human Pose | 自然场景二维人体姿态数据，供 OSX 训练。 | [论文](https://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html) | [图像与标注](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download) · [评测工具](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/evalution) |
| MPI-INF-3DHP | 含标定多视角图像与三维关节标注，供 WHAM 训练。 | [论文](https://arxiv.org/abs/1611.09813) | [作者数据页与下载脚本](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) |
| InstaVariety | 自然场景人体视频，用于时序训练。作者发布视频清单、轨迹、划分及预处理数据，原视频依赖来源可用性。 | [论文](https://doi.org/10.1109/cvpr.2019.00576) | [数据与下载脚本](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md) |

</details>

<a id="data-capture"></a>

<details>
<summary><strong>移动多机位实拍</strong>（3 项）</summary>

| 资源 | 观测、标注与用途 | 原始论文 | 数据获取与工具 |
| --- | --- | --- | --- |
| AirCap 实拍 | 无人机视图按时间近邻匹配，用身体惯性与差分 GPS 辅助参考。 | [论文](https://doi.org/10.1109/iccv.2019.00091) | [注册下载](https://aircapdata.is.tue.mpg.de/download.php) · [人体估计器](https://github.com/robot-perception-group/Aircap_Pose_Estimator) |
| AirPose 实拍 | 两架无人机一架悬停、一架绕飞。作者将真实录像和合成数据分包，实拍主要用于定性评价。 | [论文](https://doi.org/10.1109/lra.2022.3145494) | [下载说明与文件名](https://github.com/robot-perception-group/AirPose#airpose) · [作者数据目录](https://keeper.mpdl.mpg.de/d/1cae0814c4474f5a8e19/) |
| SmartMocap 手机实拍 | 两台固定手机和两台移动手机，人工同步。作者的 VolleyDay 采集与 RICH 实验分别发布。 | [论文](https://doi.org/10.1109/lra.2023.3264743) | [作者下载与脚本](https://github.com/robot-perception-group/SmartMocap#download-data) · [项目注册](https://smartmocap.is.tue.mpg.de/register.php) |

</details>

<a id="data-by-table"></a>

### 3. 测试集与评测协议

| 评测任务 | 测试数据 | 对比结果时需核对的条件 |
| --- | --- | --- |
| 局部身体、手部与全身网格 | [EMDB](https://eth-ait.github.io/emdb/) 的 EMDB1、[FreiHAND](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)、[EHF](https://smpl-x.is.tue.mpg.de/) | 基础身体、独立手部和统一人体使用不同参考网格。FreiHAND 的手部评测采用 **PA-MPVPE**。 |
| 移动单目下的世界运动 | [EMDB](https://eth-ait.github.io/emdb/) 的 EMDB2 | 视觉配置下，WHAM 和 GVHMR 使用 DPVO 估计的相机旋转。使用参考相机旋转的结果需单独比较。 |
| 地形与足部接触 | [MoviCam 下载与文件说明](https://github.com/aidilayce/PhysDynPose/blob/main/DATASET.md) | 平地和非平地分别统计。核对相机初始化、参考接触和场景扫描的使用方式。 |
| 同步多目网格重建 | [Human3.6M](http://vision.imar.ro/human3.6m/description.php) | 两个协议均以 S9、S11 为测试对象，分别采用 MoSh 和 NeuralAnnot 拟合网格。4 视图结果应在使用同一参考网格的协议内比较。 |

其他监督数据包括 [DensePose 的稠密表面对应标注](https://github.com/facebookresearch/DensePose)、[EFT 的拟合标签](https://github.com/facebookresearch/eft) 和 [PROX 人景实拍与拟合数据](https://prox.is.tue.mpg.de/download.php)。

[↑ 返回目录](#contents)

<a id="metrics"></a>

## 三、评价指标

**↓ 越小越好，↑ 越大越好。** 比较两个结果前，请核对坐标系、对齐方式、单位和参与评分的关节或顶点。

<a id="metrics-local"></a>

### 1. 局部骨架、网格与时间连续性

| 指标 | 单位 | 如何读取结果 | 论文、协议与实现 |
| --- | --- | --- | --- |
| MPJPE ↓ | mm | 对应关节的平均欧氏距离。报告关节集合、坐标系及骨盆平移对齐方式。Human3.6M 的 MoSh 参考组使用未作骨盆对齐的关节误差。 | [LMT 原论文](https://doi.org/10.1109/WACV56688.2023.00287) · [HeatFormer 评测](https://github.com/kyotovision-public/HeatFormer/blob/master/eval.py) |
| PVE / MPVE / MPVPE ↓ | mm | 对应顶点的平均欧氏距离，要求网格拓扑和顶点编号一致。EMDB1 身体与 EHF 全身网格先作骨盆平移对齐。手部与面部的局部误差分别按腕根和颈部对齐。 | [SMPLer-X 论文与评测入口](https://github.com/MotrixLab/SMPLer-X) · [WHAM 评测代码](https://github.com/yohanshin/WHAM/blob/main/lib/eval/evaluate_emdb.py) |
| PA-MPJPE / PA-MPVPE（PA-PVE）↓ | mm | 逐样本拟合相似变换后再算关节或顶点误差。FreiHAND 的手部网格由对应顶点独立配准；去除整体平移、朝向与尺度后，评的是局部形状。 | [相似配准](https://doi.org/10.1109/34.88573) · [FreiHAND 官方评分](https://github.com/lmb-freiburg/freihand/blob/master/eval.py) · [HaMeR 完整复测](evaluation/freihand_table7/README.md) |
| SmartMocap 的 RA-MPJPE / MPVPE ↓ | cm | RA-MPJPE 对齐人体根部位置、朝向与体型后评价内部姿态；该作的 MPVPE 只改变体型参数以评价体表形状。 | [SmartMocap 原论文](https://doi.org/10.1109/LRA.2023.3264743) · [作者代码](https://github.com/robot-perception-group/SmartMocap) |
| 关节旋转角误差 ↓ | ° | LMT 比较预测与参考关节旋转的角距离。非根关节旋转相对父关节，骨盆的旋转给出整体朝向。 | [LMT 原论文，式 10](https://doi.org/10.1109/WACV56688.2023.00287) · [作者项目](https://sites.google.com/view/learnable-mesh-triangulation/home) |
| 加速度误差 ↓ | m/s² 或 mm/frame² | 比较预测与参考关节轨迹的二阶差分。EMDB1 所引结果以秒为时间单位（m/s²），TCMR 在 3DPW 上按帧计（mm/frame²）。 | [TCMR 的逐帧实现](https://github.com/hongsukchoi/TCMR_RELEASE/blob/master/lib/utils/eval_utils.py) · [WHAM 的 EMDB 时间单位换算](https://github.com/yohanshin/WHAM/blob/main/lib/eval/evaluate_emdb.py#L165-L166) |
| F-score@5 mm / @15 mm ↑ | 0–1 | 衡量预测表面与参考表面在指定距离阈值内的匹配，用两个方向的匹配率计算调和平均。须注明阈值及配准方式。 | [FreiHAND 论文](https://doi.org/10.1109/ICCV.2019.00090) · [官方 F-score 实现](https://github.com/lmb-freiburg/freihand/blob/master/eval.py) |
| NoW 扫描到网格距离 ↓ | mm | 从中性脸参考扫描测到预测表面。MICA 与 DECA 的对比使用固定尺度配准。允许缩放的配准另行评分。 | [NoW 原项目](https://now.is.tue.mpg.de/) · [固定尺度评分](https://github.com/soubhiksanyal/now_evaluation/blob/main/scan2mesh_computations_metrical.py) |

InterHand2.6M 按手分别对齐根部，再用参考骨长恢复尺度。UBody 的图内 MPVPE 只统计参考投影落在画面内的顶点。[IntagHand 论文](https://doi.org/10.1109/CVPR52688.2022.00278) · [UBody 评测实现](https://github.com/IDEA-Research/OSX/blob/main/data/UBody/UBody.py)

<a id="metrics-world"></a>

### 2. 世界运动、多人布局与场景接触

| 指标 | 单位 | 如何读取结果 | 论文、协议与实现 |
| --- | --- | --- | --- |
| WA-MPJPE₁₀₀ / W-MPJPE₁₀₀ ↓ | mm | EMDB2 按 100 帧分窗。WA₁₀₀ 用整个窗口拟合相似变换，W₁₀₀ 只用前两帧确定变换，再评价整个窗口。两种协议均允许尺度对齐。 | [GVHMR 原论文](https://doi.org/10.1145/3680528.3687565) · [官方评测](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/utils/eval/eval_utils.py) |
| W-MPJPE（整段）/ 多人共同 PA-MPJPE ↓ | mm | MoviCam 在初始帧对齐后评整段，原作未详列全部变换参数。Hi4D 的共同 PA 对全序列所有人物只拟合一次相似变换，保留人物间相对布局。 | [MoviCam 论文](https://doi.org/10.1109/CVPRW67362.2025.00374) · [MultiPhys 论文](https://doi.org/10.1109/CVPR52733.2024.00226) · [作者代码](https://github.com/nicolasugrinovic/multiphys) |
| RTE（根部平移误差）↓ | % | EMDB2 按固定尺度刚体对齐整段根部轨迹，将平均位置误差除以参考根部累计路径长度，再乘 100。 | [WHAM 论文](https://openaccess.thecvf.com/content/CVPR2024/html/Shin_WHAM_Reconstructing_World-grounded_Humans_with_Accurate_3D_Motion_CVPR_2024_paper.html) · [官方实现](https://github.com/yohanshin/WHAM/blob/main/lib/eval/eval_utils.py) |
| FS（脚滑）↓ | mm | EMDB2 在参考足部 4 个顶点相邻帧移动不足 0.01 m 时统计预测位移。MoviCam 在参考接触时统计脚趾滑动，接触标签采用距场景 5 cm 的阈值。 | [WHAM 代码](https://github.com/yohanshin/WHAM/blob/main/lib/eval/eval_utils.py) · [MoviCam 论文](https://doi.org/10.1109/CVPRW67362.2025.00374) |
| 场景穿透帧比例 ↓ / 平均离场景高度 ↓ | % / mm | MoviCam 用参考场景网格和高度图判断穿透及悬空。两项合看才能区分脚落在支撑面上与人体漂离场景。 | [PhysDynPose 论文](https://doi.org/10.1109/CVPRW67362.2025.00374) · [场景与接触数据说明](https://github.com/aidilayce/PhysDynPose/blob/main/DATASET.md) |
| 多人召回率（Recall）↑ | % | AvatarPose 评测先按 500 mm 阈值匹配并去重，再对有效匹配者统计 MPJPE。召回率给出成功匹配的参考人物比例。 | [AvatarPose 论文](https://doi.org/10.1007/978-3-031-73668-1_13) · [作者的匹配与评分脚本](https://github.com/eth-ait/AvatarPose/blob/main/scripts/print_loss.py) |

<a id="metrics-camera"></a>

### 3. 相机位置与朝向

| 指标 | 单位 | 如何读取结果 | 论文、协议与实现 |
| --- | --- | --- | --- |
| TE / s-TE ↓ | m | HSfM 在同一时刻比较多部相机的中心位置。TE 先作刚体配准，s-TE 进一步允许统一缩放；这两个误差评价同刻布局。 | [HSfM 原论文](https://doi.org/10.1109/CVPR52734.2025.02044) · [作者代码](https://github.com/hongsukchoi/HSfM_RELEASE) |
| MCPE / MCOE ↓ | cm / rad | SmartMocap 分别计算相机中心的位置距离与朝向旋转的测地距离，人体根部位置与朝向另行计算。RICH 结果只覆盖具有相机真值的固定机位。 | [SmartMocap 原论文](https://doi.org/10.1109/LRA.2023.3264743) · [作者代码](https://github.com/robot-perception-group/SmartMocap) |
| 人体根部 MPE / MOE ↓ | cm / rad | SmartMocap 逐帧比较预测与参考 SMPL 根平移的欧氏距离、根朝向的旋转测地距离，再取序列平均。 | [SmartMocap 原论文，式 11–16](https://arxiv.org/pdf/2209.13906#page=6) · [作者代码](https://github.com/robot-perception-group/SmartMocap) |
| RRA@15 / CCA@15 / s-CCA@15 ↑ | 0–1 | HSfM 评测中，RRA 用 15° 阈值统计相机对的相对旋转准确率。CCA 的距离阈值为参考相机中心至质心的最大距离的 15%，s-CCA 再允许尺度对齐。 | [HSfM 指标定义，补充材料 S.4](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Muller_Reconstructing_People_Places_CVPR_2025_supplemental.pdf) · [项目](https://muelea.github.io/hsfm/) |

[↑ 返回目录](#contents)

<a id="evaluation"></a>

## 四、评测示例

| 示例 | 数据范围 | 实测结果 | 运行与结果 |
| --- | --- | --- | --- |
| **1. 手部网格 · HaMeR** | FreiHAND 全部 3,960 张测试图，每张图像评估 778 个顶点 | PA-MPVPE **5.7056 mm** | [运行说明](evaluation/freihand_table7/README.md) · [结果文件](evaluation/freihand_table7/result.json) |
| **2. 单段轨迹 · GLAMR** | EMDB 一段室外行走序列，2,240 帧 | 根轨迹误差（RTE）**19.19%** | [运行说明](evaluation/emdb_glamr_rte/README.md) · [结果文件](evaluation/emdb_glamr_rte/result.json) |
| **3. 多段轨迹 · FreeMoCap** | 3 名参与者、6 段跑台步行录制，共 42,243 帧 | 各段 RTE 等权平均 **0.02780%** | [运行说明](evaluation/freemocap_multivideo_rte/README.md) · [逐段结果](evaluation/freemocap_multivideo_rte/results.csv) |

HaMeR 用 CPU 对已有预测评分，或用 GPU 从图像重新推理，按一位小数报告均为 **5.7 mm**，与综述表 7 一致。两个轨迹示例分别演示单段评分和跨录制汇总，结果仅对应各自列出的样本。

<details>
<summary>查看评测文件目录</summary>

```text
evaluation/
├── freihand_table7/           HaMeR 手部网格评测
├── emdb_glamr_rte/            GLAMR 单段轨迹评分
└── freemocap_multivideo_rte/  FreeMoCap 多段轨迹评分
```

每个目录内的 `README.md` 都列出了数据来源和运行命令。

</details>

[↑ 返回目录](#contents)

<a id="reading"></a>

## 五、延伸阅读

| 阅读主题 | 综述与研究论文 |
| --- | --- |
| 单目人体网格、姿态与网格恢复 | [Tian 等，2023，单目网格恢复](https://doi.org/10.1109/TPAMI.2023.3298850) · [Liu 等，2024，姿态估计与网格恢复](https://doi.org/10.1016/j.neucom.2024.128049) |
| 穿衣人体与高保真几何 | [陈鸿鹄等，2024](https://doi.org/10.11834/jig.230646) · [Yang 等，2025，逼真穿衣人体](https://doi.org/10.1007/s00371-024-03641-7) · [Chen 等，2026，传统与扩散方法](https://doi.org/10.1007/s44443-025-00460-1) |
| 稠密非刚性重建与神经场人体化身 | [Tretschk 等，2023](https://doi.org/10.1111/cgf.14774) · [Gu 等，2025](https://doi.org/10.1016/j.imavis.2024.105341) |
| 多视图姿态估计与手部标注 | [Nogueira 等，2025，多视图姿态](https://doi.org/10.1016/j.imavis.2025.105437) · [Ohkawa 等，2023，手部标注与学习](https://doi.org/10.1007/s11263-023-01856-0) |
| 群组空间布局的应用 | [黄琳等，2024，基于 VR 孪生实验的群组疏散时空行为分析](https://doi.org/10.11834/jrs.20243381) |

[↑ 返回目录](#contents)

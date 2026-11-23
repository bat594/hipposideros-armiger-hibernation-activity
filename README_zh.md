# 大蹄蝠冬眠期活动分析

本仓库提供一套用于红外视频中冬眠期大蹄蝠（*Hipposideros armiger*）群体活动量化的 Python 分析流程，包含 ROI 帧差分析、活动量化、质量控制、时间分析和阈值敏感度评估，并提供图2–图7的绘图脚本及相应派生数据表。

**作者：** 欧阳珂（Ke Ouyang），吉首大学

## 软件模块

- 基于 ROI 的红外视频帧差处理与活动量化。
- 时间校正、逐日汇总和质量控制。
- 日活动模式及24小时节律分析。
- 比较 `noise_multiplier` 为 k=5、k=6、k=7 时主要活动量指标的阈值敏感度分析。
- 人工验证分析及图2–图7的复现脚本。

## 目录结构

```text
src/video_pipeline/             视频处理与 ROI 活动量化程序
analysis/                       逐日汇总、质量控制和节律分析
analysis/threshold_sensitivity/ 阈值敏感度分析
analysis/config/                输入配置模板
data/                           验证数据和派生研究数据
figure_reproduction/            图2–图7绘图脚本
validation/                     人工验证分析
docs/                           数据定义、来源和可复现性说明
```

## 环境安装

推荐使用 Python 3.10 或更新版本。

```bash
python -m pip install -r requirements.txt
```

也可以使用 Conda：

```bash
conda env create -f environment.yml
conda activate hibernating-bat-analysis
```

绘图脚本要求系统中安装真实 Arial 字体文件。若无法找到 Arial，程序会报错，不会自动替换为其他字体。

## 复现图2–图7

图件所需的派生输入表已包含在 `data/` 中。在仓库根目录运行：

```bash
python run_figure_reproduction.py
```

也可以单独运行 `figure_reproduction/` 中对应图件的脚本。绘图结果在本地生成，不包含在本源代码压缩包中。本版本不包含图1。

## 视频处理

视频处理主入口为：

```bash
python src/video_pipeline/bat_multi_roi_v5_4_rc2_single_decode.py --help
```

ROI 配置模板位于 `src/video_pipeline/config/`。运行完整视频处理流程需要原始红外视频，以及与摄像机和 ROI 布局相匹配的配置。本仓库不包含原始视频和完整的逐视频中间结果，因此不能仅凭本压缩包从原始视频完整重跑整个流程。

## 逐日分析与节律分析

节律分析程序位于 `analysis/run_hibernation_rhythm_analysis_v54.py`，配置模板为 `analysis/config/rhythm_analysis_inputs_template.json`。模板包含人工复核决策及日期、数据来源信息，但不包含逐日2秒合并活动量文件。运行分析前需在本地提供这些文件。输入字段和分析范围见 `docs/REPRODUCIBILITY.md`。

## 阈值敏感度分析

`analysis/threshold_sensitivity/` 用于比较主要活动量指标 `group_motion_load_per_s_v5_4_primary` 在 k=5、k=6、k=7 下的结果，以 k=6 作为参考设置。比较指标包括总活动量、原始及日归一化5分钟活动曲线、时间序列排序一致性、活动峰值时间，以及18:00–24:00的活动比例。

当前五日期配置需要15份逐日合并 CSV（每个日期、每个阈值各一份），这些输入文件未包含在源代码包中。每份文件须包含 `window_end_datetime` 和 `group_motion_load_per_s_v5_4_primary` 两列。分析仅使用三个阈值共同拥有的时间戳，并要求共同时间覆盖率至少达到24小时的95%。具体方法和输出定义见 `analysis/threshold_sensitivity/README.md`。

## 数据范围与限制

仓库包含人工验证表、活动与节律派生数据、去标识化复核记录、逐日行为汇总，以及复现图2–图7所需的输入表。以下材料不包含在本仓库中：

- 原始红外视频；
- 完整的逐视频2秒中间结果；
- 原始行为工作簿；
- 阈值敏感度分析所需的15份阈值专属逐日合并 CSV；
- 已生成的图件文件。

已提供的派生数据可用于复现图2–图7，但本仓库并非从原始视频开始的完整复现包。更多数据定义和可复现性说明见 `docs/`。

## 许可证

- 软件、脚本和项目文档：MIT License，见 `LICENSE`。
- `data/` 中处理后的研究数据：CC BY 4.0，见 `LICENSE-DATA.md`。
- 第三方依赖和材料仍遵循各自的许可证。

## 引用

引用元数据见 `CITATION.cff`。请引用实际使用的软件版本。只有在对应 Zenodo 记录正式发布，且 DOI 能够解析到该版本归档后，才应在论文中引用该版本 DOI。

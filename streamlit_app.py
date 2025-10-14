"""Interactive Streamlit dashboard for analysing SA-MOO sweep results."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import yaml
from pandas.api.types import is_numeric_dtype

REPO_ROOT = Path(__file__).resolve().parent
SWEEP_RUNS_DIR = REPO_ROOT / "sweep_runs"

PARAM_COLUMN_ALIASES: Dict[str, str] = {
    "target_image_id": "Target Image",
    "population_size": "Population Size",
    "num_generations": "Generations",
    "fixed_k": "Fixed K",
    "crossover_prob": "Crossover Prob",
    "zero_sample_prob": "Zero-sample Prob",
    "perturbation_mode": "Perturbation Mode",
    "is_targeted_attack": "Is Targeted",
    "enable_continuous_perturbation": "Continuous Mode",
    "edge_guidance_enabled": "Edge Guidance",
    "dct_low_frequency_enabled": "DCT Low Freq",
    "target_class_id": "Target Class",
}

NUMERIC_COLUMNS = [
    "composite_score",
    "success",
    "l2",
    "l0",
    "psnr",
    "lpips",
    "mean_delta_e00",
    "max_delta_e00",
    "orig_confidence",
    "adv_confidence",
    "confidence_drop",
    "runtime_sec",
]

NUMERIC_PATTERN = re.compile(r"^[0-9mpEe\.+-]+$")

PARAMETER_DESCRIPTIONS = [
    ("Target Image", "CIFAR-10 测试集中使用的目标图像编号 (0-9999)"),
    ("Population Size", "进化算法每一代维护的个体数量，越大搜索越全面，但耗时更久"),
    ("Generations", "进化迭代轮数，决定算法可尝试的总步数"),
    ("Fixed K", "稀疏约束 K：允许被修改的像素/通道数量上限"),
    ("Crossover Prob", "交叉操作概率，用于父代之间交换信息"),
    ("Zero-sample Prob", "初始化/变异时抽到 0 扰动的概率，用于控制稀疏程度"),
    ("Perturbation Mode", "扰动作用的颜色空间：rgb_sim / channel / v_channel"),
    ("Is Targeted", "是否目标攻击：True 表示强制分类到指定类别"),
    ("Target Class", "目标攻击时的目标类别索引"),
    ("Continuous Mode", "连续扰动配置（cont_off / cont_low / cont_half / cont_full）"),
    ("Edge Guidance", "是否启用语义边缘引导来偏向重要区域"),
    ("DCT Low Freq", "是否启用 DCT 低频约束，以限制扰动主要集中在低频"),
]

METRIC_DESCRIPTIONS = [
    ("success", "攻击是否成功（1 成功 / 0 失败）"),
    ("l2", "扰动的 L2 范数，衡量整体扰动幅度"),
    ("l0", "扰动的 L0 范数，等同于实际修改的像素/通道数量"),
    ("psnr", "峰值信噪比，值越高代表与原图越相似"),
    ("lpips", "感知损失指标，越低表示人眼感知差异越小"),
    ("mean_delta_e00", "CIELAB 颜色空间的平均 ΔE00，与颜色感知差异相关"),
    ("max_delta_e00", "CIELAB 中的最大 ΔE00"),
    ("orig_confidence", "模型对真实标签的原始置信度"),
    ("adv_confidence", "加入扰动后模型对真实标签的置信度"),
    ("confidence_drop", "orig_confidence - adv_confidence，越高代表攻击越有效"),
    ("runtime_sec", "单次实验耗时（秒）"),
    ("composite_score", "基于 ranking.priority 计算出的综合评分，用于排序"),
]


def _slug_to_filename(slug: str, max_length: int = 120) -> str:
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:10]
    cutoff = max_length - len(digest) - 2
    truncated = slug[:cutoff]
    return f"{truncated}__{digest}"


def _decode_numeric_token(token: str) -> Optional[float]:
    if not NUMERIC_PATTERN.match(token):
        return None
    candidate = token.replace("p", ".")
    candidate = candidate.replace("m", "-")
    try:
        value = float(candidate)
    except ValueError:
        return None
    if candidate.lower() not in {"nan", "inf", "-inf"} and value.is_integer():
        return int(value)
    return value


def _parse_slug(slug: str) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for part in slug.split("__"):
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        raw_lower = raw_value.lower()
        if raw_lower in {"true", "false"}:
            values[key] = raw_lower == "true"
            continue
        if raw_lower in {"targeted", "untargeted"}:
            values[key] = raw_lower == "targeted"
            continue
        numeric_value = _decode_numeric_token(raw_value)
        if numeric_value is not None:
            values[key] = numeric_value
            continue
        values[key] = raw_value
    return values


@st.cache_data(show_spinner=False)
def load_summary(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return pd.DataFrame(data)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_config_yaml(path_str: str) -> Optional[str]:
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def augment_with_parameters(df: pd.DataFrame) -> pd.DataFrame:
    parsed_rows: List[Dict[str, object]] = []
    for slug in df["slug"].tolist():
        parsed_rows.append(_parse_slug(slug))
    params_df = pd.DataFrame(parsed_rows)
    for key, alias in PARAM_COLUMN_ALIASES.items():
        if key in params_df.columns:
            df[alias] = params_df[key]
        else:
            df[alias] = None
    return df


def prepare_numeric_parameter_matrix(df: pd.DataFrame, parameter_columns: List[str]) -> pd.DataFrame:
    numeric_data: Dict[str, pd.Series] = {}
    for col in parameter_columns:
        if col not in df.columns:
            continue
        series = df[col]
        if series.dropna().empty:
            continue
        if is_numeric_dtype(series):
            numeric_data[col] = series.astype(float)
        elif series.dtype == bool:
            numeric_data[col] = series.astype(float)
        else:
            categorical = pd.Categorical(series.fillna("Unknown"))
            numeric_data[col] = pd.Series(categorical.codes, index=series.index, dtype=float)
    return pd.DataFrame(numeric_data)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    st.sidebar.subheader("Filters")

    status_options = sorted(filtered["status"].dropna().unique().tolist())
    selected_status = st.sidebar.multiselect(
        "Status",
        status_options,
        default=status_options,
    )
    if selected_status:
        filtered = filtered[filtered["status"].isin(selected_status)]

    success_values = sorted(filtered["success"].dropna().unique().tolist())
    if success_values:
        default_success = success_values
        selected_success = st.sidebar.multiselect(
            "Success flag",
            success_values,
            default=default_success,
        )
        if selected_success:
            filtered = filtered[filtered["success"].isin(selected_success)]

    categorical_filters = {
        "Perturbation Mode": filtered["Perturbation Mode"].dropna().unique().tolist(),
        "Is Targeted": filtered["Is Targeted"].dropna().unique().tolist(),
        "Continuous Mode": filtered["Continuous Mode"].dropna().unique().tolist(),
        "Edge Guidance": filtered["Edge Guidance"].dropna().unique().tolist(),
        "DCT Low Freq": filtered["DCT Low Freq"].dropna().unique().tolist(),
    }
    for label, options in categorical_filters.items():
        if not options:
            continue
        default = options
        selected = st.sidebar.multiselect(label, options, default=default)
        if selected:
            filtered = filtered[filtered[label].isin(selected)]

    for column in NUMERIC_COLUMNS:
        if column not in filtered.columns:
            continue
        col_data = filtered[column].dropna()
        if col_data.empty:
            continue
        min_val = float(col_data.min())
        max_val = float(col_data.max())
        if min_val == max_val:
            continue
        step = (max_val - min_val) / 100 or 0.01
        selected_range = st.sidebar.slider(
            column,
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=step,
        )
        filtered = filtered[(filtered[column] >= selected_range[0]) & (filtered[column] <= selected_range[1])]

    return filtered


def render_table(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filtered runs")
    if df.empty:
        st.info("No runs match the current filters.")
        return df

    sort_options = [col for col in ["composite_score", "confidence_drop", "l2", "runtime_sec"] if col in df.columns]
    sort_metric = st.selectbox("Sort by", sort_options or ["index"], index=0)
    descending = st.checkbox("Sort descending", value=True)

    total_rows = len(df)
    if total_rows <= 10:
        top_n = total_rows
        st.caption(f"显示全部 {total_rows} 条记录")
    else:
        slider_step = 5 if total_rows - 10 >= 5 else max(1, total_rows - 10)
        default_value = min(total_rows, 50)
        top_n = st.slider(
            "Show top N rows",
            min_value=10,
            max_value=total_rows,
            value=default_value,
            step=slider_step,
        )

    if sort_metric in df.columns:
        df_sorted = df.sort_values(by=sort_metric, ascending=not descending).head(top_n)
    else:
        df_sorted = df.sort_index(ascending=not descending).head(top_n)
    display_columns = [
        "index",
        "slug",
        "status",
        "composite_score",
        "success",
        "l2",
        "l0",
        "psnr",
        "lpips",
        "confidence_drop",
        "runtime_sec",
        "Perturbation Mode",
        "Is Targeted",
        "Continuous Mode",
        "Edge Guidance",
        "DCT Low Freq",
    ]
    existing_columns = [col for col in display_columns if col in df_sorted.columns]
    st.dataframe(df_sorted[existing_columns], use_container_width=True)
    return df_sorted


def render_visualizations(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Add or relax filters to see visualizations.")
        return

    st.subheader("Metric scatter plot")
    numeric_available = [col for col in NUMERIC_COLUMNS if col in df.columns]
    if len(numeric_available) >= 2:
        default_x = numeric_available[1] if len(numeric_available) > 1 else numeric_available[0]
        default_y = numeric_available[0]
        x_axis = st.selectbox("X axis", numeric_available, index=numeric_available.index(default_x))
        y_axis = st.selectbox("Y axis", numeric_available, index=numeric_available.index(default_y))
        color_by = st.selectbox("Color by", ["Is Targeted", "Perturbation Mode", "Edge Guidance", "DCT Low Freq"], index=0)
        hover_cols = [col for col in ["index", "slug", "composite_score", "runtime_sec"] if col in df.columns]
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by if color_by in df.columns else None,
            hover_data=hover_cols,
            title=f"{y_axis} vs {x_axis}",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Aggregate by parameter")
    aggregatable_params = [alias for alias in PARAM_COLUMN_ALIASES.values() if alias in df.columns and df[alias].notna().any()]
    if aggregatable_params:
        parameter = st.selectbox("Group by", aggregatable_params, index=0)
        metric = st.selectbox("Metric", ["success", "l2", "confidence_drop", "runtime_sec", "composite_score"], index=0)
        agg_df = (
            df.groupby(parameter)
            .agg(
                runs=("index", "count"),
                success_rate=("success", "mean"),
                avg_metric=(metric, "mean"),
                median_metric=(metric, "median"),
            )
            .reset_index()
            .sort_values(by="avg_metric", ascending=False)
        )
        agg_df["success_rate"] = agg_df["success_rate"].round(3)
        agg_df["avg_metric"] = agg_df["avg_metric"].round(3)
        agg_df["median_metric"] = agg_df["median_metric"].round(3)
        st.dataframe(agg_df, use_container_width=True)
    else:
        st.info("No parameter metadata available for aggregation.")


def render_parameter_insights(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Add or relax filters以查看参数与指标的关系。")
        return

    parameter_cols = [alias for alias in PARAM_COLUMN_ALIASES.values() if alias in df.columns and df[alias].notna().any()]
    metric_cols = [col for col in NUMERIC_COLUMNS if col in df.columns and df[col].notna().any()]

    if not parameter_cols or not metric_cols:
        st.info("当前数据未能解析出参数信息，无法分析参数和指标之间的关系。")
        return

    st.subheader("参数 ↔ 指标 洞察")
    st.caption("通过箱线图、并行分类、并行坐标和相关性热图来直观理解参数对攻击指标的影响。")

    default_metric = "confidence_drop" if "confidence_drop" in metric_cols else metric_cols[0]
    metric_focus = st.selectbox("关注的指标", metric_cols, index=metric_cols.index(default_metric))

    # 箱线图：同一参数不同取值下的指标分布
    box_param = st.selectbox("箱线图参数", parameter_cols, index=0)
    base_columns = [box_param, metric_focus, "slug", "success"]
    seen = set()
    unique_cols = []
    for col in base_columns:
        if col not in df.columns or col in seen:
            continue
        unique_cols.append(col)
        seen.add(col)
    box_df = df[unique_cols].dropna(subset=[box_param, metric_focus])
    if not box_df.empty:
        box_df = box_df.copy()
        original_values = box_df[box_param]
        # 保持原有排序，数值型则按升序排列
        if pd.api.types.is_numeric_dtype(original_values):
            category_values = sorted(pd.unique(original_values))
        else:
            # 对字符串/布尔等非数值类型维持出现顺序
            _, idx = np.unique(original_values.astype(str), return_index=True)
            category_values = list(original_values.iloc[np.sort(idx)])
        category_labels = [str(val) for val in category_values]
        display_col = f"{box_param} (分类轴)"
        box_df[display_col] = pd.Categorical(
            [str(val) for val in original_values], categories=category_labels, ordered=True
        )
        fig_box = px.box(
            box_df,
            x=display_col,
            y=metric_focus,
            color=display_col,
            points="all",
            hover_data={"slug": True, "success": True, box_param: True},
            labels={display_col: box_param},
        )
        fig_box.update_layout(
            title=f"{metric_focus} vs {box_param}",
            xaxis_title=box_param,
            legend_title=box_param,
        )
        fig_box.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=category_labels,
            tickmode="array",
            tickvals=category_labels,
            ticktext=category_labels,
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("选定参数缺少数据，无法绘制箱线图。")

    # 并行分类：查看多参数组合与成功/失败之间的关系
    parallel_cats_params = st.multiselect(
        "并行分类维度",
        options=parameter_cols,
        default=parameter_cols[: min(4, len(parameter_cols))],
    )
    if parallel_cats_params:
        cat_df = df[parallel_cats_params + ["success"]].dropna()
        if not cat_df.empty:
            cat_df = cat_df.copy()
            cat_df["Attack outcome"] = cat_df["success"].map({1: "成功", 0: "失败"}).fillna("未知")
            fig_cats = px.parallel_categories(
                cat_df,
                dimensions=parallel_cats_params + ["Attack outcome"],
                color="success",
                color_continuous_scale=px.colors.sequential.Viridis,
            )
            st.plotly_chart(fig_cats, use_container_width=True)
        else:
            st.info("所选维度没有足够的数据绘制并行分类图。")

    # 并行坐标：将参数和值映射到数值空间，观察整体趋势
    numeric_param_df = prepare_numeric_parameter_matrix(df, parameter_cols)
    if not numeric_param_df.empty:
        numeric_param_df = numeric_param_df.join(df[[metric_focus]], how="left")
        available_numeric_dims = numeric_param_df.columns.tolist()
        default_dims = [col for col in available_numeric_dims if col != metric_focus][:3]
        selected_dims = st.multiselect(
            "并行坐标维度（数值化后）",
            options=[col for col in available_numeric_dims if col != metric_focus],
            default=default_dims,
        )
        if selected_dims:
            pc_df = numeric_param_df[selected_dims + [metric_focus]].dropna()
            if not pc_df.empty:
                fig_pc = px.parallel_coordinates(
                    pc_df,
                    dimensions=selected_dims + [metric_focus],
                    color=metric_focus,
                    color_continuous_scale=px.colors.sequential.Viridis,
                )
                st.plotly_chart(fig_pc, use_container_width=True)
            else:
                st.info("所选维度在过滤后没有可用数据绘制并行坐标。")

    # 参数与指标的相关性热图
    if not numeric_param_df.empty:
        metrics_for_join = [col for col in metric_cols if col not in numeric_param_df.columns]
        corr_source = numeric_param_df.join(df[metrics_for_join], how="left") if metrics_for_join else numeric_param_df.copy()
        corr_matrix = corr_source.corr(numeric_only=True)
        param_cols_numeric = [col for col in numeric_param_df.columns if col in corr_matrix.columns]
        metric_cols_numeric = [col for col in metric_cols if col in corr_matrix.columns]
        if "orig_confidence" in metric_cols_numeric:
            metric_cols_numeric = [col for col in metric_cols_numeric if col != "orig_confidence"] + ["orig_confidence"]
        if param_cols_numeric and metric_cols_numeric:
            heatmap = corr_matrix.loc[param_cols_numeric, metric_cols_numeric]
            fig_heat = px.imshow(
                heatmap,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                origin="lower",
                aspect="auto",
                zmin=-1,
                zmax=1,
            )
            fig_heat.update_layout(title="参数-指标相关性热图")
            st.plotly_chart(fig_heat, use_container_width=True)


def render_glossary() -> None:
    st.header("核心参数释义")
    for name, description in PARAMETER_DESCRIPTIONS:
        st.markdown(f"**{name}**：{description}")

    st.header("关键指标释义")
    for name, description in METRIC_DESCRIPTIONS:
        st.markdown(f"**{name}**：{description}")

    st.header("阅读提示")
    st.markdown(
        """
        - 建议先在 **Filters** 中锁定特定任务场景（如目标/非目标攻击、扰动模式），再观察参数与指标的关系。
        - `confidence_drop` 与 `success` 往往能直接体现攻击效果，而 `l2`/`l0`/`lpips` 更关注扰动代价与视觉质量。
        - 在选择连续扰动 (Continuous Mode) 时，务必结合 `Fixed K` 及 `Zero-sample Prob` 观察是否导致扰动过大。
        - 当 `Edge Guidance` 或 `DCT Low Freq` 启用时，可以对照 `harvest` 中的视觉结果确认其实际影响。
        
          **参数 ↔ 指标 页面阅读指南**

          1. **箱线图**
              - 横轴每个刻度代表一个参数取值（已按实际取值顺序排列）。
              - 盒体高度代表该取值下指标分布（中位数、四分位范围、异常值）。
              - 可快速比较同一指标在不同参数取值下的稳定性和上限/下限。

          2. **并行分类图**
              - 将多个类别型参数串联，宽条表示样本量，大面积区域表示常见组合。
              - 最后一列是攻击结果（成功/失败），观察颜色流向即可发现哪些组合更易成功。

          3. **并行坐标图**
              - 所有参数已数值化/归一化为 0~1 区间，线条颜色表示所选指标。
              - 将参数值与指标共同观察，可以定位既成功又低 L2 的组合，例如，找寻颜色偏暖（指标高）的线同时位于左下或右上。

          4. **相关性热图**
              - 行为参数（数值化后），列为指标，颜色深浅表示正/负相关程度。
              - `orig_confidence` 被放在最右侧，可快速查看原始置信度与其他指标的关系。
        """
    )


def render_run_details(df: pd.DataFrame, sweep_dir: Path) -> None:
    if df.empty:
        return
    st.subheader("Run details")
    indices = df["index"].astype(int).tolist()
    selected_index = st.selectbox("Select run index", indices)
    row = df[df["index"].astype(int) == selected_index].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Composite score", f"{row.get('composite_score', float('nan')):.3f}")
    col2.metric("Success", f"{row.get('success', float('nan')):.2f}")
    col3.metric("L2", f"{row.get('l2', float('nan')):.3f}")

    st.write("**Slug**")
    st.code(row["slug"], language="text")
    st.write("**Run directory**")
    st.code(row.get("run_dir", ""), language="text")

    slug = row["slug"]
    slug_safe = _slug_to_filename(slug)
    config_path = sweep_dir / "configs" / f"{int(row['index']):03d}_{slug_safe}.yaml"
    config_text = load_config_yaml(str(config_path))
    if config_text:
        st.write("**Config snapshot**")
        st.code(config_text, language="yaml")
    else:
        st.info("Config file not found (maybe keep_configs was set to false).")

    harvest_dir = sweep_dir / "harvest" / slug_safe
    if harvest_dir.exists():
        st.write("**Harvested artefacts**")
        image_files = [
            "adversarial_result.png",
            "final_perturbed.png",
            "final_perturbed_preprocessed.png",
            "convergence.png",
        ]
        show_images = st.checkbox("Show images", value=False)
        for name in image_files:
            path = harvest_dir / name
            if not path.exists():
                continue
            if show_images and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(str(path), caption=name, use_column_width=True)
            else:
                st.write(name)
    else:
        st.info("No harvested artefacts for this run.")


def main() -> None:
    st.set_page_config(page_title="SA-MOO Sweep Dashboard", layout="wide")
    st.title("SA-MOO Sweep Analytics Dashboard")
    st.caption("Explore parameter sweeps, compare metrics, and drill into individual runs.")

    if not SWEEP_RUNS_DIR.exists():
        st.error("No sweep_runs directory found. Run run_parameter_sweep.py first.")
        return

    sweep_dirs = sorted([p for p in SWEEP_RUNS_DIR.iterdir() if p.is_dir()], reverse=True)
    if not sweep_dirs:
        st.error("No sweep runs available yet.")
        return

    sweep_labels = [p.name for p in sweep_dirs]
    selected_label = st.sidebar.selectbox("Sweep run", sweep_labels, index=0)
    sweep_dir = next(p for p in sweep_dirs if p.name == selected_label)

    summary_csv = sweep_dir / "summary.csv"
    summary_json = sweep_dir / "summary.json"
    if summary_csv.exists():
        df = load_summary(str(summary_csv))
    elif summary_json.exists():
        df = load_summary(str(summary_json))
    else:
        st.error("summary.csv or summary.json not found in the selected sweep directory.")
        return

    df = augment_with_parameters(df)
    filtered_df = apply_filters(df)

    tab_table, tab_viz, tab_params, tab_details, tab_guide = st.tabs([
        "Table",
        "Visualizations",
        "Parameter insights",
        "Run details",
        "Guide",
    ])

    with tab_table:
        displayed_df = render_table(filtered_df)
    with tab_viz:
        render_visualizations(filtered_df)
    with tab_params:
        render_parameter_insights(filtered_df)
    with tab_details:
        render_run_details(filtered_df if not filtered_df.empty else df, sweep_dir)
    with tab_guide:
        render_glossary()


if __name__ == "__main__":
    main()

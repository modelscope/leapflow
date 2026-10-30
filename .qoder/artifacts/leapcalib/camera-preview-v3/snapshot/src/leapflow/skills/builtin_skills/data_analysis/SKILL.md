---
name: data_analysis
description: "CSV/JSON data analysis with statistical summaries, quality checks, and visualization guidance"
version: 1.0.0
metadata:
  leapflow:
    category: "analysis"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "analysis"
    tags: ["data", "analysis", "statistics", "CSV", "JSON", "visualization"]
    requires_tools: ["file_read", "shell_run"]
platforms: []
triggers:
  - "analyze data"
  - "data analysis"
  - "CSV analysis"
  - "statistics"
  - "数据分析"
  - "统计分析"
  - "summarize this data"
  - "data quality check"
---

# Data Analysis

## Purpose

Analyze structured data (CSV, JSON, TSV, Parquet) through a systematic pipeline:
inspect the data, assess quality, compute descriptive statistics, detect patterns,
and produce actionable insights with visualization recommendations.  This skill
turns raw data into understanding.

## Guiding Principles

1. **Look before you compute** — Always inspect raw data (head, tail, shape,
   dtypes) before running any analysis.  Assumptions about structure are the
   leading cause of wrong results.
2. **Quality gates** — Missing values, duplicates, type mismatches, and encoding
   errors must be identified and reported before analysis proceeds.
3. **Context over numbers** — A mean is meaningless without understanding what
   the column represents.  Always relate statistics back to the domain.
4. **Appropriate methods** — Choose statistical measures that match the data
   distribution.  Median and IQR for skewed data; mean and std for normal.
5. **Reproducibility** — Every analysis step must be scriptable and repeatable.
   Provide the exact commands or code used.

## Workflow

### Phase 1 — Data Ingestion and Inspection

1. Read the file with `file_read` to examine the first 20–50 rows and understand
   structure.
2. Determine format: CSV (detect delimiter, quoting, encoding), JSON (flat vs
   nested), TSV, or other.
3. Record:
   - **Shape**: number of rows × columns.
   - **Columns**: name, inferred type (numeric, categorical, datetime, text).
   - **Sample values**: 3–5 example values per column.
4. If the file is large (>10k rows), use `shell_run` with Python or
   command-line tools for efficient processing:
   ```
   python3 -c "import pandas as pd; df=pd.read_csv('data.csv'); print(df.shape); print(df.dtypes); print(df.head())"
   ```

### Phase 2 — Data Quality Assessment

Evaluate data health before analysis:

- **Missing values**: count per column, percentage, pattern (random vs
  systematic — e.g., all missing for certain dates or categories).
- **Duplicates**: exact row duplicates and near-duplicates on key columns.
- **Type issues**: numeric columns stored as strings, inconsistent date formats,
  mixed types within a column.
- **Outliers**: values beyond 3σ or 1.5×IQR from the median — flag but do not
  remove without domain justification.
- **Encoding**: check for mojibake, BOM markers, or mixed encodings.

Produce a quality summary:
```
Data Quality Report:
  Rows: 15,234 | Columns: 12
  Missing: 3 columns with >5% missing (col_a: 12%, col_b: 7%, col_c: 6%)
  Duplicates: 43 exact duplicates found
  Type issues: col_price has 18 non-numeric entries
  Outliers: col_age has 5 values > 120
```

Recommend cleaning actions for each issue found.

### Phase 3 — Descriptive Statistics

Compute statistics appropriate to each column type:

**Numeric columns**:
- Central tendency: mean, median, mode.
- Dispersion: std, variance, IQR, range.
- Shape: skewness, kurtosis.
- Quantiles: 5th, 25th, 50th, 75th, 95th percentiles.

**Categorical columns**:
- Unique count and cardinality ratio (unique/total).
- Top-N value frequency (with percentages).
- Rare categories (appearing < 1% of rows).

**Datetime columns**:
- Range (min to max), span.
- Frequency/granularity (daily, monthly, irregular).
- Gaps: missing dates in an otherwise regular series.

**Cross-column**:
- Correlation matrix for numeric pairs (flag |r| > 0.7).
- Contingency tables for categorical pairs when relevant.

### Phase 4 — Pattern Detection and Insights

Go beyond summary statistics:

1. **Trends**: for time-series data, identify upward/downward trends,
   seasonality, and change points.
2. **Segmentation**: group by categorical columns and compare numeric
   distributions across groups.
3. **Anomalies**: data points that are statistically unusual and may indicate
   errors, fraud, or interesting phenomena.
4. **Relationships**: notable correlations, dependencies, or interactions
   between columns.

Each insight must include:
- **What** was found (specific numbers).
- **Why** it might matter (domain interpretation).
- **Confidence level** (strong evidence vs. suggestive pattern).

### Phase 5 — Visualization Recommendations

For each key finding, recommend the most effective chart:

| Data Pattern | Recommended Chart |
|---|---|
| Distribution of one variable | Histogram or KDE plot |
| Comparison across categories | Bar chart (horizontal for many categories) |
| Trend over time | Line chart with confidence band |
| Relationship between two numerics | Scatter plot with regression line |
| Part-of-whole composition | Stacked bar or treemap (not pie) |
| Multivariate relationships | Heatmap (correlation) or parallel coordinates |
| Outlier detection | Box plot or violin plot |

Provide ready-to-run code (matplotlib/seaborn or the project's preferred library)
for the top 3 recommended visualizations.

### Phase 6 — Report

Produce a structured analysis report:

```
## Dataset Overview
<shape, source, time range, key columns>

## Data Quality
<issues found, cleaning applied or recommended>

## Key Statistics
<table of most important metrics>

## Insights
1. <Finding with supporting numbers>
2. ...

## Recommended Visualizations
<code for top charts>

## Next Steps
<suggested deeper analyses or actions>
```

## Error Handling

| Situation | Action |
|---|---|
| File too large for memory | Use chunked reading (`chunksize` in pandas) or sample first 10k rows with a disclaimer. |
| Encoding errors | Try UTF-8, then Latin-1, then detect with `chardet`; report encoding used. |
| All numeric columns are actually IDs | Flag that statistical summaries are meaningless for ID columns; exclude from analysis. |
| No clear structure (freeform text file) | Report that the file is not structured tabular data; suggest NLP-based analysis instead. |

## Limitations

- This skill does not perform machine learning (regression, classification).
  It provides the exploratory analysis that informs modeling decisions.
- Visualization code is generated but not rendered; the user must execute it
  in their environment.
- Very large datasets (>100M rows) may require distributed tools beyond
  the scope of this skill.

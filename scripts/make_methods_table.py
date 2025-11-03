import pandas as pd
from pathlib import Path

csv_path = Path('analysis/methods_comparison.csv')
md_path = Path('analysis/methods_comparison.md')
tex_path = Path('docs/table_methods.tex')

if not csv_path.exists():
    raise SystemExit(f"Missing CSV: {csv_path}")

df = pd.read_csv(csv_path)
# Sort by F2 descending
df = df.sort_values('f2', ascending=False)

# Format columns
def fmt(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)

def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)

rows = []
rows.append(["Method", "F2", "PR AUC", "Model Size (MB)", "Throughput (ex/s)"])
for _, r in df.iterrows():
    rows.append([
        r['method'],
        fmt(r['f2']),
        fmt(r['pr_auc']),
        fmt(r['model_size_mb']),
        fmt_int(r['throughput_eps']),
    ])

# Write Markdown table
md_lines = []
md_lines.append("| " + " | ".join(rows[0]) + " |")
md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
for row in rows[1:]:
    md_lines.append("| " + " | ".join(map(str, row)) + " |")
md_path.parent.mkdir(parents=True, exist_ok=True)
md_path.write_text("\n".join(md_lines), encoding='utf-8')
print(f"Wrote Markdown table: {md_path}")

# Write LaTeX table (booktabs)
tex_lines = []
tex_lines.append("% Auto-generated from analysis/methods_comparison.csv")
tex_lines.append("\\begin{table}[t]")
tex_lines.append("  \\centering")
tex_lines.append("  \\caption{Methods comparison: performance and efficiency}")
tex_lines.append("  \\begin{tabular}{lrrrr}")
tex_lines.append("    \\toprule")
tex_lines.append("    Method & F2 & PR~AUC & Model Size (MB) & Throughput (ex/s) \\\ ")
tex_lines.append("    \\midrule")
for row in rows[1:]:
    tex_lines.append(
        "    {} & {} & {} & {} & {} \\".format(row[0], row[1], row[2], row[3], row[4])
    )
tex_lines.append("    \\bottomrule")
tex_lines.append("  \\end{tabular}")
tex_lines.append("  \\label{tab:methods-comparison}")
tex_lines.append("\\end{table}")
tex_path.parent.mkdir(parents=True, exist_ok=True)
tex_path.write_text("\n".join(tex_lines), encoding='utf-8')
print(f"Wrote LaTeX table: {tex_path}")

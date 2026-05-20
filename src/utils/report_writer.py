"""
EDA Report Generation Utilities.
Provides functions to write structured Markdown reports from profiling results.
"""
from typing import Dict, Any, List
from pathlib import Path
import os


def format_null_table(null_stats: Dict[str, Dict[str, Any]]) -> str:
    """Format null stats as a Markdown table."""
    if not null_stats:
        return "No missing values detected.\n"
    
    lines = ["| Column | Null Count | % Missing |", "|--------|-----------|-----------|"]
    for col, stats in sorted(null_stats.items(), key=lambda x: x[1]['pct'], reverse=True):
        lines.append(f"| `{col}` | {stats['count']:,} | {stats['pct']:.2f}% |")
    return "\n".join(lines) + "\n"


def format_schema_table(schema: Dict[str, str]) -> str:
    """Format schema as a Markdown table."""
    lines = ["| Column | Data Type |", "|--------|-----------|"]
    for col, dtype in schema.items():
        lines.append(f"| `{col}` | {dtype} |")
    return "\n".join(lines) + "\n"


def format_cardinality_table(cardinality: Dict[str, int]) -> str:
    """Format cardinality as a Markdown table."""
    if not cardinality:
        return "No low-cardinality columns.\n"
    
    lines = ["| Column | Unique Values |", "|--------|--------------|"]
    for col, count in sorted(cardinality.items(), key=lambda x: x[1]):
        lines.append(f"| `{col}` | {count} |")
    return "\n".join(lines) + "\n"


def format_table_summary(profile: Dict[str, Any]) -> str:
    """Format a full table profile as a Markdown section."""
    name = profile.get("table_name", "Unknown")
    
    if "error" in profile:
        return f"### {name}\n\n⚠️ Error: {profile['error']}\n\n"
    
    lines = [
        f"### {name}",
        "",
        f"- **Total Rows**: {profile['total_rows']:,}",
        f"- **Files**: {profile['files_count']}",
        f"- **Sample Analyzed**: {profile['sample_size']:,}",
        f"- **Columns**: {len(profile['schema'])}",
        "",
        "#### Schema",
        format_schema_table(profile['schema']),
        "#### Missing Values",
        format_null_table(profile['null_stats']),
        "#### Cardinality (Low-cardinality columns)",
        format_cardinality_table(profile['cardinality']),
    ]
    return "\n".join(lines)


def write_report(filepath: str, content: str):
    """Write a report string to a Markdown file, creating directories if needed."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

"""
Plotting Utilities for EDA.
Provides reusable plotting functions with consistent styling.
All EDA scripts should import from this module for visualization.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# ─── Global Style ─────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800', '#9C27B0',
          '#00BCD4', '#E91E63', '#8BC34A', '#FFC107', '#607D8B']


def save_figure(fig: plt.Figure, filepath: str, tight: bool = True):
    """Save figure to file, creating parent dirs if needed."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(filepath, bbox_inches='tight', facecolor='white', edgecolor='none')
    else:
        fig.savefig(filepath, facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_bar(
    labels: list,
    values: list,
    title: str,
    xlabel: str,
    ylabel: str,
    filepath: str,
    horizontal: bool = False,
    color: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_values: bool = True,
    rotation: int = 0
):
    """Create and save a bar chart."""
    fig, ax = plt.subplots(figsize=figsize)
    color = color or COLORS[0]

    if horizontal:
        bars = ax.barh(range(len(labels)), values, color=color, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        if show_values:
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{val:,.0f}', va='center', fontsize=9)
    else:
        bars = ax.bar(range(len(labels)), values, color=color, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=rotation, ha='right' if rotation else 'center')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if show_values:
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                        f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

    ax.set_title(title, fontweight='bold', pad=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    save_figure(fig, filepath)


def plot_pie(
    labels: list,
    values: list,
    title: str,
    filepath: str,
    figsize: Tuple[int, int] = (8, 8)
):
    """Create and save a pie chart."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = COLORS[:len(labels)]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=0.85
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
    ax.set_title(title, fontweight='bold', pad=20)
    save_figure(fig, filepath)


def plot_line(
    x: list,
    y: list,
    title: str,
    xlabel: str,
    ylabel: str,
    filepath: str,
    figsize: Tuple[int, int] = (14, 6),
    color: Optional[str] = None
):
    """Create and save a line chart."""
    fig, ax = plt.subplots(figsize=figsize)
    color = color or COLORS[0]
    ax.plot(x, y, color=color, linewidth=1.5, alpha=0.8)
    ax.fill_between(x, y, alpha=0.1, color=color)
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    fig.autofmt_xdate(rotation=45)
    save_figure(fig, filepath)


def plot_histogram(
    values: list,
    title: str,
    xlabel: str,
    ylabel: str,
    filepath: str,
    bins: int = 50,
    log_scale: bool = False,
    figsize: Tuple[int, int] = (12, 6),
    color: Optional[str] = None
):
    """Create and save a histogram."""
    fig, ax = plt.subplots(figsize=figsize)
    color = color or COLORS[0]
    ax.hist(values, bins=bins, color=color, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale('log')
    save_figure(fig, filepath)


def plot_multi_bar(
    labels: list,
    datasets: List[Tuple[str, list]],
    title: str,
    xlabel: str,
    ylabel: str,
    filepath: str,
    figsize: Tuple[int, int] = (14, 6)
):
    """Create and save a grouped bar chart."""
    fig, ax = plt.subplots(figsize=figsize)
    n_groups = len(labels)
    n_datasets = len(datasets)
    bar_width = 0.8 / n_datasets
    x = np.arange(n_groups)

    for i, (name, values) in enumerate(datasets):
        offset = (i - n_datasets / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=name, color=COLORS[i % len(COLORS)],
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend()
    save_figure(fig, filepath)

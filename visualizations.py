"""
Visualization functions for the TTS benchmarking tool
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from benchmarking_engine import BenchmarkResult

def create_latency_distribution(results: List[BenchmarkResult]) -> go.Figure:
    """Create latency distribution chart"""
    
    data = []
    for result in results:
        if result.success:
            data.append({
                "provider": result.provider.title(),
                "latency_ms": result.latency_ms,
                "category": result.metadata.get("category", "unknown"),
                "word_count": result.metadata.get("word_count", 0)
            })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    fig = px.box(
        df,
        x="provider",
        y="latency_ms",
        color="provider",
        title="Latency Distribution by Provider",
        labels={"latency_ms": "Latency (ms)", "provider": "Provider"},
        hover_data=["category", "word_count"]
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Provider",
        yaxis_title="Latency (ms)"
    )
    
    return fig

def create_success_rate_chart(results: List[BenchmarkResult]) -> go.Figure:
    """Create success rate chart"""
    
    provider_stats = {}
    for result in results:
        if result.provider not in provider_stats:
            provider_stats[result.provider] = {"total": 0, "successful": 0}
        
        provider_stats[result.provider]["total"] += 1
        if result.success:
            provider_stats[result.provider]["successful"] += 1
    
    providers = []
    success_rates = []
    total_tests = []
    
    for provider, stats in provider_stats.items():
        providers.append(provider.title())
        success_rate = (stats["successful"] / stats["total"]) * 100
        success_rates.append(success_rate)
        total_tests.append(stats["total"])
    
    fig = go.Figure(data=[
        go.Bar(
            x=providers,
            y=success_rates,
            text=[f"{rate:.1f}%" for rate in success_rates],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Success Rate: %{y:.1f}%<br>Total Tests: %{customdata}<extra></extra>",
            customdata=total_tests,
            marker_color=px.colors.qualitative.Set1[:len(providers)]
        )
    ])
    
    fig.update_layout(
        title="Success Rate by Provider",
        xaxis_title="Provider",
        yaxis_title="Success Rate (%)",
        height=400,
        yaxis=dict(range=[0, 105])
    )
    
    return fig

def create_leaderboard_chart(leaderboard: List[Dict[str, Any]]) -> go.Figure:
    """Create ELO leaderboard chart"""
    
    providers = [item["provider"].title() for item in leaderboard]
    ratings = [item["elo_rating"] for item in leaderboard]
    ranks = [item["rank"] for item in leaderboard]
    
    num_providers = len(providers)
    colors = list(px.colors.sequential.Viridis[:max(1, num_providers - 4)])
    
    bottom_colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']
    colors.extend(bottom_colors[:min(4, num_providers)])
    
    fig = go.Figure(data=[
        go.Bar(
            y=providers,
            x=ratings,
            orientation='h',
            text=[f"#{rank}" for rank in ranks],
            textposition='inside',
            hovertemplate="<b>%{y}</b><br>ELO Rating: %{x}<br>Rank: #%{customdata}<extra></extra>",
            customdata=ranks,
            marker_color=colors[:num_providers]
        )
    ])
    
    fig.update_layout(
        title="ELO Leaderboard",
        xaxis_title="ELO Rating",
        yaxis_title="Provider",
        height=max(300, len(providers) * 50),
        yaxis=dict(categoryorder="total ascending")
    )
    
    return fig

def create_latency_vs_quality_scatter(results: List[BenchmarkResult]) -> go.Figure:
    """Create latency vs quality scatter plot"""
    
    data = []
    for result in results:
        if result.success:
            quality_score = result.file_size_bytes / 1024
            
            data.append({
                "provider": result.provider.title(),
                "latency_ms": result.latency_ms,
                "quality_score": quality_score,
                "word_count": result.metadata.get("word_count", 0),
                "category": result.metadata.get("category", "unknown")
            })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    fig = px.scatter(
        df,
        x="latency_ms",
        y="quality_score",
        color="provider",
        size="word_count",
        hover_data=["category"],
        title="Latency vs Quality Trade-off",
        labels={
            "latency_ms": "Latency (ms)",
            "quality_score": "File Size (KB)",
            "word_count": "Word Count"
        }
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_performance_heatmap(results: List[BenchmarkResult]) -> go.Figure:
    """Create performance heatmap by category and provider"""
    
    performance_data = {}
    
    for result in results:
        if result.success:
            provider = result.provider.title()
            category = result.metadata.get("category", "unknown")
            
            if provider not in performance_data:
                performance_data[provider] = {}
            
            if category not in performance_data[provider]:
                performance_data[provider][category] = []
            
            performance_data[provider][category].append(result.latency_ms)
    
    providers = list(performance_data.keys())
    categories = list(set(cat for provider_data in performance_data.values() for cat in provider_data.keys()))
    
    z_data = []
    for provider in providers:
        row = []
        for category in categories:
            if category in performance_data[provider]:
                avg_latency = np.mean(performance_data[provider][category])
                row.append(avg_latency)
            else:
                row.append(None)
        z_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=categories,
        y=providers,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate="Provider: %{y}<br>Category: %{x}<br>Avg Latency: %{z:.1f}ms<extra></extra>"
    ))
    
    fig.update_layout(
        title="Average Latency Heatmap (Provider vs Category)",
        xaxis_title="Category",
        yaxis_title="Provider",
        height=max(300, len(providers) * 50)
    )
    
    return fig

def create_latency_timeline(results: List[BenchmarkResult]) -> go.Figure:
    """Create latency timeline chart"""
    
    data = []
    for result in results:
        if result.success:
            data.append({
                "timestamp": pd.to_datetime(result.timestamp),
                "provider": result.provider.title(),
                "latency_ms": result.latency_ms,
                "sample_id": result.sample_id
            })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    df = df.sort_values("timestamp")
    
    fig = px.line(
        df,
        x="timestamp",
        y="latency_ms",
        color="provider",
        title="Latency Over Time",
        labels={"timestamp": "Time", "latency_ms": "Latency (ms)"},
        hover_data=["sample_id"]
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_error_analysis_chart(results: List[BenchmarkResult]) -> go.Figure:
    """Create error analysis chart"""
    
    error_data = {}
    
    for result in results:
        if not result.success and result.error_message:
            provider = result.provider.title()
            error_type = result.error_message.split(':')[0]
            
            if provider not in error_data:
                error_data[provider] = {}
            
            error_data[provider][error_type] = error_data[provider].get(error_type, 0) + 1
    
    if not error_data:
        return go.Figure().add_annotation(text="No errors to analyze", x=0.5, y=0.5)
    
    providers = list(error_data.keys())
    error_types = list(set(error_type for provider_errors in error_data.values() for error_type in provider_errors.keys()))
    
    fig = go.Figure()
    
    for error_type in error_types:
        counts = [error_data[provider].get(error_type, 0) for provider in providers]
        fig.add_trace(go.Bar(
            name=error_type,
            x=providers,
            y=counts,
            hovertemplate=f"<b>{error_type}</b><br>Provider: %{{x}}<br>Count: %{{y}}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Error Analysis by Provider and Type",
        xaxis_title="Provider",
        yaxis_title="Error Count",
        barmode='stack',
        height=400
    )
    
    return fig

def create_word_count_performance(results: List[BenchmarkResult]) -> go.Figure:
    """Create performance vs word count analysis"""
    
    data = []
    for result in results:
        if result.success:
            data.append({
                "provider": result.provider.title(),
                "word_count": result.metadata.get("word_count", 0),
                "latency_ms": result.latency_ms,
                "latency_per_word": result.latency_ms / max(result.metadata.get("word_count", 1), 1)
            })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    fig = px.scatter(
        df,
        x="word_count",
        y="latency_per_word",
        color="provider",
        trendline="ols",
        title="Latency per Word vs Text Length",
        labels={
            "word_count": "Word Count",
            "latency_per_word": "Latency per Word (ms/word)"
        },
        hover_data=["latency_ms"]
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_summary_dashboard(results: List[BenchmarkResult]) -> go.Figure:
    """Create comprehensive summary dashboard"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Success Rate", "Avg Latency", "File Size Distribution", "Error Count"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "box"}, {"type": "bar"}]]
    )
    
    provider_metrics = {}
    
    for result in results:
        provider = result.provider.title()
        if provider not in provider_metrics:
            provider_metrics[provider] = {
                "total": 0, "successful": 0, "latencies": [], 
                "file_sizes": [], "errors": 0
            }
        
        provider_metrics[provider]["total"] += 1
        if result.success:
            provider_metrics[provider]["successful"] += 1
            provider_metrics[provider]["latencies"].append(result.latency_ms)
            provider_metrics[provider]["file_sizes"].append(result.file_size_bytes / 1024)
        else:
            provider_metrics[provider]["errors"] += 1
    
    providers = list(provider_metrics.keys())
    
    success_rates = [(metrics["successful"] / metrics["total"]) * 100 for metrics in provider_metrics.values()]
    fig.add_trace(
        go.Bar(x=providers, y=success_rates, name="Success Rate", showlegend=False),
        row=1, col=1
    )
    
    avg_latencies = [np.mean(metrics["latencies"]) if metrics["latencies"] else 0 for metrics in provider_metrics.values()]
    fig.add_trace(
        go.Bar(x=providers, y=avg_latencies, name="Avg Latency", showlegend=False),
        row=1, col=2
    )
    
    for i, provider in enumerate(providers):
        if provider_metrics[provider]["file_sizes"]:
            fig.add_trace(
                go.Box(y=provider_metrics[provider]["file_sizes"], name=provider, showlegend=False),
                row=2, col=1
            )
    
    error_counts = [metrics["errors"] for metrics in provider_metrics.values()]
    fig.add_trace(
        go.Bar(x=providers, y=error_counts, name="Errors", showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Performance Summary Dashboard")
    
    return fig

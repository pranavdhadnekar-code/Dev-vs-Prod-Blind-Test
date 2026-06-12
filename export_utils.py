"""
Export utilities for benchmark results
"""
import json
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import io
import zipfile
from dataclasses import asdict

from benchmarking_engine import BenchmarkResult, BenchmarkSummary

class ExportManager:
    """Manages export functionality for benchmark results"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_results_json(self, results: List[BenchmarkResult], filename: Optional[str] = None) -> str:
        """Export results to JSON format"""
        
        if filename is None:
            filename = f"benchmark_results_{self.timestamp}.json"
        
        # Convert results to dictionaries
        data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_results": len(results),
                "format_version": "1.0"
            },
            "results": [asdict(result) for result in results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def export_results_csv(self, results: List[BenchmarkResult], filename: Optional[str] = None) -> str:
        """Export results to CSV format"""
        
        if filename is None:
            filename = f"benchmark_results_{self.timestamp}.csv"
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(result) for result in results])
        
        # Flatten metadata column
        if 'metadata' in df.columns:
            metadata_df = pd.json_normalize(df['metadata'])
            metadata_df.columns = [f"metadata_{col}" for col in metadata_df.columns]
            df = pd.concat([df.drop('metadata', axis=1), metadata_df], axis=1)
        
        df.to_csv(filename, index=False, encoding='utf-8')
        
        return filename
    
    def export_summary_report(
        self, 
        results: List[BenchmarkResult], 
        summaries: Dict[str, BenchmarkSummary],
        leaderboard: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """Export comprehensive summary report"""
        
        if filename is None:
            filename = f"benchmark_report_{self.timestamp}.json"
        
        # Calculate additional statistics
        total_tests = len(results)
        successful_tests = len([r for r in results if r.success])
        unique_providers = len(set(r.provider for r in results))
        unique_samples = len(set(r.sample_id for r in results))
        
        # Provider comparison matrix
        comparison_matrix = self._create_comparison_matrix(results)
        
        # Category performance analysis
        category_analysis = self._analyze_by_category(results)
        
        # Length category analysis
        length_analysis = self._analyze_by_length(results)
        
        report_data = {
            "report_info": {
                "generated_at": datetime.now().isoformat(),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate_overall": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "unique_providers": unique_providers,
                "unique_samples": unique_samples,
                "format_version": "1.0"
            },
            "provider_summaries": {provider: asdict(summary) for provider, summary in summaries.items()},
            "leaderboard": leaderboard,
            "comparison_matrix": comparison_matrix,
            "category_analysis": category_analysis,
            "length_analysis": length_analysis,
            "raw_results": [asdict(result) for result in results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def export_excel_workbook(
        self, 
        results: List[BenchmarkResult], 
        summaries: Dict[str, BenchmarkSummary],
        leaderboard: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """Export results to Excel workbook with multiple sheets"""
        
        if filename is None:
            filename = f"benchmark_results_{self.timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Raw results sheet
            results_df = pd.DataFrame([asdict(result) for result in results])
            if 'metadata' in results_df.columns:
                metadata_df = pd.json_normalize(results_df['metadata'])
                metadata_df.columns = [f"metadata_{col}" for col in metadata_df.columns]
                results_df = pd.concat([results_df.drop('metadata', axis=1), metadata_df], axis=1)
            
            results_df.to_excel(writer, sheet_name='Raw Results', index=False)
            
            # Summary sheet
            summary_data = []
            for provider, summary in summaries.items():
                summary_dict = asdict(summary)
                summary_data.append(summary_dict)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Leaderboard sheet - add model names
            from config import TTS_PROVIDERS
            
            def get_model_name(provider: str) -> str:
                """Helper to get model name"""
                return TTS_PROVIDERS.get(provider).model_name if provider in TTS_PROVIDERS else provider
            
            leaderboard_df = pd.DataFrame(leaderboard)
            if 'provider' in leaderboard_df.columns and 'model' not in leaderboard_df.columns:
                leaderboard_df['model'] = leaderboard_df['provider'].apply(get_model_name)
                # Reorder columns to put model after provider
                cols = leaderboard_df.columns.tolist()
                provider_idx = cols.index('provider')
                cols.insert(provider_idx + 1, cols.pop(cols.index('model')))
                leaderboard_df = leaderboard_df[cols]
            leaderboard_df.to_excel(writer, sheet_name='Leaderboard', index=False)
            
            # Success rate analysis
            success_analysis = self._create_success_analysis_df(results)
            success_analysis.to_excel(writer, sheet_name='Success Analysis', index=False)
            
            # Latency analysis
            latency_analysis = self._create_latency_analysis_df(results)
            latency_analysis.to_excel(writer, sheet_name='Latency Analysis', index=False)
        
        return filename
    
    def create_export_package(
        self, 
        results: List[BenchmarkResult], 
        summaries: Dict[str, BenchmarkSummary],
        leaderboard: List[Dict[str, Any]],
        include_formats: List[str] = ["json", "csv", "excel", "report"]
    ) -> str:
        """Create a comprehensive export package as ZIP file"""
        
        package_filename = f"benchmark_package_{self.timestamp}.zip"
        
        with zipfile.ZipFile(package_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Export in different formats
            if "json" in include_formats:
                json_file = self.export_results_json(results)
                zipf.write(json_file, f"data/{json_file}")
            
            if "csv" in include_formats:
                csv_file = self.export_results_csv(results)
                zipf.write(csv_file, f"data/{csv_file}")
            
            if "excel" in include_formats:
                excel_file = self.export_excel_workbook(results, summaries, leaderboard)
                zipf.write(excel_file, f"data/{excel_file}")
            
            if "report" in include_formats:
                report_file = self.export_summary_report(results, summaries, leaderboard)
                zipf.write(report_file, f"reports/{report_file}")
            
            # Add metadata file
            metadata = {
                "package_info": {
                    "created_at": datetime.now().isoformat(),
                    "total_results": len(results),
                    "providers_tested": list(set(r.provider for r in results)),
                    "included_formats": include_formats,
                    "description": "TTS Benchmarking Results Package"
                }
            }
            
            metadata_json = json.dumps(metadata, indent=2)
            zipf.writestr("package_info.json", metadata_json)
        
        return package_filename
    
    def _create_comparison_matrix(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Create provider comparison matrix"""
        
        providers = list(set(r.provider for r in results if r.success))
        matrix = {}
        
        for provider_a in providers:
            matrix[provider_a] = {}
            results_a = [r for r in results if r.provider == provider_a and r.success]
            
            for provider_b in providers:
                if provider_a == provider_b:
                    matrix[provider_a][provider_b] = {"wins": 0, "losses": 0, "ties": 0}
                    continue
                
                results_b = [r for r in results if r.provider == provider_b and r.success]
                
                # Compare on same samples
                wins = losses = ties = 0
                
                for result_a in results_a:
                    matching_b = [r for r in results_b if r.sample_id == result_a.sample_id]
                    
                    for result_b in matching_b:
                        if result_a.latency_ms < result_b.latency_ms:
                            wins += 1
                        elif result_a.latency_ms > result_b.latency_ms:
                            losses += 1
                        else:
                            ties += 1
                
                matrix[provider_a][provider_b] = {
                    "wins": wins,
                    "losses": losses,
                    "ties": ties
                }
        
        return matrix
    
    def _analyze_by_category(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance by text category"""
        
        category_stats = {}
        
        for result in results:
            if result.success:
                category = result.metadata.get("category", "unknown")
                
                if category not in category_stats:
                    category_stats[category] = {
                        "providers": {},
                        "total_tests": 0,
                        "avg_latency": 0,
                        "latencies": []
                    }
                
                provider = result.provider
                if provider not in category_stats[category]["providers"]:
                    category_stats[category]["providers"][provider] = {
                        "tests": 0,
                        "avg_latency": 0,
                        "latencies": []
                    }
                
                category_stats[category]["providers"][provider]["tests"] += 1
                category_stats[category]["providers"][provider]["latencies"].append(result.latency_ms)
                category_stats[category]["total_tests"] += 1
                category_stats[category]["latencies"].append(result.latency_ms)
        
        # Calculate averages
        for category, stats in category_stats.items():
            if stats["latencies"]:
                stats["avg_latency"] = sum(stats["latencies"]) / len(stats["latencies"])
            
            for provider, provider_stats in stats["providers"].items():
                if provider_stats["latencies"]:
                    provider_stats["avg_latency"] = sum(provider_stats["latencies"]) / len(provider_stats["latencies"])
                # Remove raw latencies to reduce size
                del provider_stats["latencies"]
            
            del stats["latencies"]
        
        return category_stats
    
    def _analyze_by_length(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance by text length"""
        
        length_stats = {}
        
        for result in results:
            if result.success:
                length_category = result.metadata.get("length_category", "unknown")
                word_count = result.metadata.get("word_count", 0)
                
                if length_category not in length_stats:
                    length_stats[length_category] = {
                        "providers": {},
                        "total_tests": 0,
                        "avg_latency": 0,
                        "avg_word_count": 0,
                        "latency_per_word": 0,
                        "latencies": [],
                        "word_counts": []
                    }
                
                provider = result.provider
                if provider not in length_stats[length_category]["providers"]:
                    length_stats[length_category]["providers"][provider] = {
                        "tests": 0,
                        "avg_latency": 0,
                        "avg_latency_per_word": 0,
                        "latencies": [],
                        "word_counts": []
                    }
                
                length_stats[length_category]["providers"][provider]["tests"] += 1
                length_stats[length_category]["providers"][provider]["latencies"].append(result.latency_ms)
                length_stats[length_category]["providers"][provider]["word_counts"].append(word_count)
                
                length_stats[length_category]["total_tests"] += 1
                length_stats[length_category]["latencies"].append(result.latency_ms)
                length_stats[length_category]["word_counts"].append(word_count)
        
        # Calculate averages
        for length_cat, stats in length_stats.items():
            if stats["latencies"]:
                stats["avg_latency"] = sum(stats["latencies"]) / len(stats["latencies"])
                stats["avg_word_count"] = sum(stats["word_counts"]) / len(stats["word_counts"])
                stats["latency_per_word"] = stats["avg_latency"] / max(stats["avg_word_count"], 1)
            
            for provider, provider_stats in stats["providers"].items():
                if provider_stats["latencies"]:
                    provider_stats["avg_latency"] = sum(provider_stats["latencies"]) / len(provider_stats["latencies"])
                    avg_words = sum(provider_stats["word_counts"]) / len(provider_stats["word_counts"])
                    provider_stats["avg_latency_per_word"] = provider_stats["avg_latency"] / max(avg_words, 1)
                
                # Remove raw data to reduce size
                del provider_stats["latencies"]
                del provider_stats["word_counts"]
            
            del stats["latencies"]
            del stats["word_counts"]
        
        return length_stats
    
    def _create_success_analysis_df(self, results: List[BenchmarkResult]) -> pd.DataFrame:
        """Create success rate analysis DataFrame"""
        
        analysis_data = []
        
        # Group by provider
        providers = set(r.provider for r in results)
        
        for provider in providers:
            provider_results = [r for r in results if r.provider == provider]
            total = len(provider_results)
            successful = len([r for r in provider_results if r.success])
            
            # By category
            categories = set(r.metadata.get("category", "unknown") for r in provider_results)
            
            for category in categories:
                category_results = [r for r in provider_results if r.metadata.get("category") == category]
                cat_total = len(category_results)
                cat_successful = len([r for r in category_results if r.success])
                
                analysis_data.append({
                    "provider": provider,
                    "category": category,
                    "total_tests": cat_total,
                    "successful_tests": cat_successful,
                    "success_rate": (cat_successful / cat_total * 100) if cat_total > 0 else 0,
                    "failure_rate": ((cat_total - cat_successful) / cat_total * 100) if cat_total > 0 else 0
                })
        
        return pd.DataFrame(analysis_data)
    
    def _create_latency_analysis_df(self, results: List[BenchmarkResult]) -> pd.DataFrame:
        """Create latency analysis DataFrame"""
        
        analysis_data = []
        successful_results = [r for r in results if r.success]
        
        # Group by provider and category
        for result in successful_results:
            analysis_data.append({
                "provider": result.provider,
                "category": result.metadata.get("category", "unknown"),
                "length_category": result.metadata.get("length_category", "unknown"),
                "word_count": result.metadata.get("word_count", 0),
                "latency_ms": result.latency_ms,
                "latency_per_word": result.latency_ms / max(result.metadata.get("word_count", 1), 1),
                "file_size_kb": result.file_size_bytes / 1024,
                "timestamp": result.timestamp
            })
        
        return pd.DataFrame(analysis_data)

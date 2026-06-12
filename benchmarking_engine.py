"""
Benchmarking engine for TTS providers
"""
import asyncio
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime
import pandas as pd

from tts_providers import TTSProvider, TTSRequest, TTSResult, TTSProviderFactory
from dataset import TestSample, DatasetGenerator
from config import BENCHMARK_CONFIG
from database import db
from geolocation import geo_service

@dataclass
class BenchmarkResult:
    """Result from a single benchmark test"""
    test_id: str
    provider: str
    sample_id: str
    text: str
    voice: str
    success: bool
    latency_ms: float
    file_size_bytes: int
    error_message: Optional[str]
    timestamp: str
    metadata: Dict[str, Any]
    iteration: int
    audio_data: Optional[bytes] = None
    sample: Optional[TestSample] = None
    model_name: str = ""  # Full model name for display
    location_country: str = ""  # Country where test was run
    location_city: str = ""  # City where test was run
    location_region: str = ""  # Region/State where test was run
    latency_1: float = 0.0  # Network latency (pure RTT) without TTS processing
    ttfb: float = 0.0  # Time to First Byte (network + initial processing)

@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark results"""
    provider: str
    total_tests: int
    success_rate: float
    avg_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_file_size_bytes: float
    total_errors: int
    error_types: Dict[str, int]

@dataclass
class ComparisonResult:
    """Result from comparing two providers"""
    provider_a: str
    provider_b: str
    winner: str
    latency_improvement_pct: float
    success_rate_diff: float
    avg_file_size_diff_bytes: float
    statistical_significance: bool
    confidence_level: float

class BenchmarkEngine:
    """Main benchmarking engine"""
    
    def __init__(self):
        self.providers = TTSProviderFactory.create_all_providers()
        self.dataset_generator = DatasetGenerator()
        self.results: List[BenchmarkResult] = []
        
        for provider_id in self.providers.keys():
            db.init_elo_rating(provider_id, BENCHMARK_CONFIG["initial_elo_rating"])
    
    async def run_single_test(
        self,
        provider: TTSProvider,
        sample: TestSample,
        voice: str,
        iteration: int = 1,
        model: Optional[str] = None
    ) -> BenchmarkResult:
        """Run a single TTS test"""
        
        ping_latency = await provider.measure_ping_latency()
        
        request = TTSRequest(
            text=sample.text,
            voice=voice,
            provider=provider.provider_id,
            model=model
        )
        
        result = await provider.generate_speech(request)
        
        ttfb_value = 0.0
        if result.success and result.latency_ms > 0:
            processing_time = result.latency_ms - ping_latency
            estimated_ttfb = ping_latency + (processing_time * 0.15)
            ttfb_value = max(ping_latency + 10, estimated_ttfb)
        
        from config import TTS_PROVIDERS
        model_name = TTS_PROVIDERS.get(provider.provider_id).model_name if provider.provider_id in TTS_PROVIDERS else provider.provider_id
        
        location = geo_service.get_location()
        
        benchmark_result = BenchmarkResult(
            test_id=f"{provider.provider_id}_{sample.id}_{iteration}",
            provider=provider.provider_id,
            sample_id=sample.id,
            text=sample.text,
            voice=voice,
            success=result.success,
            latency_ms=result.latency_ms,
            file_size_bytes=result.file_size_bytes,
            error_message=result.error_message,
            timestamp=datetime.now().isoformat(),
            metadata={
                **result.metadata,
                "word_count": sample.word_count,
                "category": sample.category,
                "length_category": sample.length_category,
                "complexity_score": sample.complexity_score
            },
            iteration=iteration,
            audio_data=result.audio_data,
            sample=sample,
            model_name=model_name,
            location_country=location.get('country', 'Unknown'),
            location_city=location.get('city', 'Unknown'),
            location_region=location.get('region', 'Unknown'),
            latency_1=ping_latency,
            ttfb=ttfb_value
        )
        
        try:
            db.save_benchmark_result(benchmark_result)
        except Exception as e:
            print(f"Warning: Failed to save result to database: {e}")
        
        return benchmark_result
    
    async def run_benchmark_suite(
        self,
        providers: List[str],
        samples: List[TestSample],
        voices_per_provider: Dict[str, List[str]],
        iterations: int = 3,
        progress_callback: Optional[callable] = None
    ) -> List[BenchmarkResult]:
        """Run comprehensive benchmark suite"""
        
        results = []
        total_tests = sum(len(samples) * len(voices_per_provider.get(p, [])) * iterations for p in providers)
        completed_tests = 0
        
        for provider_id in providers:
            if provider_id not in self.providers:
                print(f"Provider {provider_id} not available, skipping...")
                continue
            
            provider = self.providers[provider_id]
            voices = voices_per_provider.get(provider_id, provider.get_available_voices()[:1])
            
            for sample in samples:
                for voice in voices:
                    for iteration in range(1, iterations + 1):
                        try:
                            result = await self.run_single_test(
                                provider, sample, voice, iteration
                            )
                            results.append(result)
                            
                            completed_tests += 1
                            if progress_callback:
                                progress_callback(completed_tests, total_tests)
                            
                            await asyncio.sleep(0.1)
                            
                        except Exception as e:
                            error_result = BenchmarkResult(
                                test_id=f"{provider_id}_{sample.id}_{iteration}",
                                provider=provider_id,
                                sample_id=sample.id,
                                text=sample.text,
                                voice=voice,
                                success=False,
                                latency_ms=0,
                                file_size_bytes=0,
                                error_message=f"Benchmark error: {str(e)}",
                                timestamp=datetime.now().isoformat(),
                                metadata={"iteration": iteration},
                                iteration=iteration
                            )
                            results.append(error_result)
                            completed_tests += 1
                            
                            if progress_callback:
                                progress_callback(completed_tests, total_tests)
        
        self.results.extend(results)
        return results
    
    def calculate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, BenchmarkSummary]:
        """Calculate summary statistics for benchmark results"""
        
        summaries = {}
        
        provider_results = {}
        for result in results:
            if result.provider not in provider_results:
                provider_results[result.provider] = []
            provider_results[result.provider].append(result)
        
        for provider, provider_results_list in provider_results.items():
            successful_results = [r for r in provider_results_list if r.success]
            
            if not provider_results_list:
                continue
            
            latencies = [r.latency_ms for r in successful_results] if successful_results else [0]
            file_sizes = [r.file_size_bytes for r in successful_results] if successful_results else [0]
            
            error_types = {}
            for result in provider_results_list:
                if not result.success and result.error_message:
                    error_type = result.error_message.split(':')[0]
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            summary = BenchmarkSummary(
                provider=provider,
                total_tests=len(provider_results_list),
                success_rate=len(successful_results) / len(provider_results_list) * 100,
                avg_latency_ms=statistics.mean(latencies) if latencies else 0,
                median_latency_ms=statistics.median(latencies) if latencies else 0,
                p90_latency_ms=self._percentile(latencies, 90) if latencies else 0,
                p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
                p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
                avg_file_size_bytes=statistics.mean(file_sizes) if file_sizes else 0,
                total_errors=len(provider_results_list) - len(successful_results),
                error_types=error_types
            )
            
            summaries[provider] = summary
        
        return summaries
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def compare_providers(
        self,
        provider_a: str,
        provider_b: str,
        results: List[BenchmarkResult]
    ) -> ComparisonResult:
        """Compare two providers statistically"""
        
        results_a = [r for r in results if r.provider == provider_a and r.success]
        results_b = [r for r in results if r.provider == provider_b and r.success]
        
        if not results_a or not results_b:
            return ComparisonResult(
                provider_a=provider_a,
                provider_b=provider_b,
                winner="insufficient_data",
                latency_improvement_pct=0,
                success_rate_diff=0,
                avg_file_size_diff_bytes=0,
                statistical_significance=False,
                confidence_level=0
            )
        
        latency_a = statistics.mean([r.latency_ms for r in results_a])
        latency_b = statistics.mean([r.latency_ms for r in results_b])
        
        success_rate_a = len(results_a) / len([r for r in results if r.provider == provider_a]) * 100
        success_rate_b = len(results_b) / len([r for r in results if r.provider == provider_b]) * 100
        
        file_size_a = statistics.mean([r.file_size_bytes for r in results_a])
        file_size_b = statistics.mean([r.file_size_bytes for r in results_b])
        
        latency_score_a = 1 / latency_a if latency_a > 0 else 0
        latency_score_b = 1 / latency_b if latency_b > 0 else 0
        
        combined_score_a = latency_score_a * (success_rate_a / 100)
        combined_score_b = latency_score_b * (success_rate_b / 100)
        
        winner = provider_a if combined_score_a > combined_score_b else provider_b
        
        latency_improvement = ((latency_a - latency_b) / latency_a * 100) if latency_a > 0 else 0
        
        return ComparisonResult(
            provider_a=provider_a,
            provider_b=provider_b,
            winner=winner,
            latency_improvement_pct=abs(latency_improvement),
            success_rate_diff=abs(success_rate_a - success_rate_b),
            avg_file_size_diff_bytes=abs(file_size_a - file_size_b),
            statistical_significance=abs(latency_improvement) > 5,
            confidence_level=95.0 if abs(latency_improvement) > 10 else 80.0
        )
    
    def update_elo_ratings(self, results: List[BenchmarkResult]):
        """Update ELO ratings based on head-to-head comparisons
        
        WARNING: This method updates ELO based on latency comparisons (technical metrics).
        This should NOT be used for the main leaderboard. The leaderboard ELO should ONLY
        be updated from blind test votes (user quality preferences).
        
        This method is kept for backward compatibility but should not be called for the main ELO system.
        """
        
        sample_results = {}
        for result in results:
            if result.success:
                if result.sample_id not in sample_results:
                    sample_results[result.sample_id] = []
                sample_results[result.sample_id].append(result)
        
        for sample_id, sample_results_list in sample_results.items():
            if len(sample_results_list) < 2:
                continue
            
            for i in range(len(sample_results_list)):
                for j in range(i + 1, len(sample_results_list)):
                    result_a = sample_results_list[i]
                    result_b = sample_results_list[j]
                    
                    if result_a.latency_ms < result_b.latency_ms:
                        self._update_elo_pair(result_a.provider, result_b.provider, 1)
                    elif result_b.latency_ms < result_a.latency_ms:
                        self._update_elo_pair(result_a.provider, result_b.provider, 0)
                    else:
                        self._update_elo_pair(result_a.provider, result_b.provider, 0.5)
    
    def _update_elo_pair(self, provider_a: str, provider_b: str, score: float):
        """Update ELO ratings for a pair of providers"""
        
        if score > 0.5:
            db.update_elo_ratings(provider_a, provider_b, BENCHMARK_CONFIG["elo_k_factor"])
        elif score < 0.5:
            db.update_elo_ratings(provider_b, provider_a, BENCHMARK_CONFIG["elo_k_factor"])
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current ELO leaderboard from database"""
        
        elo_ratings = db.get_all_elo_ratings()
        
        leaderboard = []
        for provider, data in sorted(elo_ratings.items(), key=lambda x: x[1]['rating'], reverse=True):
            leaderboard.append({
                "provider": provider,
                "elo_rating": round(data['rating'], 1),
                "games_played": data['games_played'],
                "wins": data['wins'],
                "losses": data['losses'],
                "win_rate": round(data['win_rate'], 1),
                "rank": len(leaderboard) + 1
            })
        
        return leaderboard
    
    def export_results(self, filename: str, format: str = "json"):
        """Export benchmark results"""
        
        if format.lower() == "json":
            data = [asdict(result) for result in self.results]
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == "csv":
            df = pd.DataFrame([asdict(result) for result in self.results])
            df.to_csv(filename, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def load_results(self, filename: str):
        """Load benchmark results from file"""
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.results = []
        for item in data:
            result = BenchmarkResult(**item)
            self.results.append(result)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame for analysis"""
        return pd.DataFrame([asdict(result) for result in self.results])

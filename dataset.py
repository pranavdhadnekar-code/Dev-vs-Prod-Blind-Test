"""
Test dataset generation and management for TTS benchmarking
"""
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

@dataclass
class TestSample:
    """Represents a single test sample"""
    id: str
    text: str
    word_count: int
    category: str
    length_category: str
    complexity_score: float

class DatasetGenerator:
    """Generates diverse test datasets for TTS benchmarking"""
    
    def __init__(self):
        self.samples = []
        self._load_base_sentences()
    
    def _load_base_sentences(self):
        """Load base sentences for different categories and lengths"""
        self.base_sentences = {
            "news": [
                "Breaking news from around the world continues to shape our understanding of current events.",
                "The latest economic indicators suggest a shift in market trends that could impact global trade.",
                "Scientists have made a groundbreaking discovery that could revolutionize renewable energy technology.",
                "Local authorities are implementing new policies to address climate change challenges in urban areas.",
                "International cooperation remains crucial for addressing global health emergencies and pandemic preparedness."
            ],
            "literature": [
                "In the quiet moments before dawn, she contemplated the profound changes that had transformed her life.",
                "The ancient library held secrets that had been whispered through generations of scholars and mystics.",
                "His journey across the vast landscape revealed truths about human nature and the resilience of the spirit.",
                "The storyteller's voice carried the weight of centuries, weaving tales that bridged past and present.",
                "Through the mist-covered mountains, travelers sought wisdom from the hermit who lived beyond the clouds."
            ],
            "conversation": [
                "Hey, how was your weekend? Did you manage to finish that project you were working on?",
                "I think we should meet up for coffee sometime next week to discuss the upcoming presentation.",
                "The weather has been absolutely beautiful lately, perfect for outdoor activities and weekend trips.",
                "Have you tried that new restaurant downtown? I heard they have amazing pasta dishes.",
                "I'm really excited about the concert tonight. The band is supposed to be incredible live."
            ],
            "technical": [
                "The implementation of machine learning algorithms requires careful consideration of data preprocessing and feature engineering.",
                "Cloud computing architectures must balance scalability, security, and cost-effectiveness for enterprise applications.",
                "Artificial intelligence systems utilize neural networks to process complex patterns in large datasets.",
                "Software development methodologies emphasize iterative approaches and continuous integration practices.",
                "Database optimization techniques include indexing strategies and query performance analysis."
            ],
            "narrative": [
                "Once upon a time, in a kingdom far beyond the mountains, there lived a wise queen who ruled with compassion.",
                "The detective carefully examined the evidence, knowing that every detail could be crucial to solving the mystery.",
                "As the spaceship approached the distant planet, the crew prepared for their first contact with alien civilization.",
                "The old lighthouse keeper had witnessed countless storms, but this one seemed different from all the others.",
                "Through the enchanted forest, the young adventurer discovered magical creatures that existed only in legends."
            ]
        }
    
    def _extend_sentence(self, base_sentence: str, target_words: int) -> str:
        """Extend a sentence to reach target word count"""
        words = base_sentence.split()
        current_count = len(words)
        
        if current_count >= target_words:
            return ' '.join(words[:target_words])
        
        # Extension phrases to add natural length
        extensions = [
            "Furthermore, this development",
            "In addition to these factors,",
            "Moreover, recent studies indicate that",
            "Additionally, experts suggest that",
            "It's worth noting that",
            "According to recent research,",
            "As a result of these findings,",
            "Consequently, many professionals believe",
            "Therefore, it becomes clear that",
            "Subsequently, the evidence shows"
        ]
        
        extended = base_sentence
        while len(extended.split()) < target_words:
            extension = random.choice(extensions)
            remaining_words = target_words - len(extended.split())
            
            if remaining_words > 10:
                extended += f" {extension} the situation continues to evolve with new developments emerging regularly."
            else:
                # Add shorter phrases for final words
                short_additions = [
                    "and continues to develop.",
                    "with ongoing implications.",
                    "in various contexts.",
                    "across multiple domains.",
                    "throughout the industry."
                ]
                extended += f" {random.choice(short_additions)}"
        
        # Trim to exact word count
        return ' '.join(extended.split()[:target_words])
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity based on various factors"""
        words = text.split()
        sentences = text.split('.')
        
        # Average word length
        avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / len(words)
        
        # Average sentence length
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Punctuation density
        punctuation_count = sum(1 for char in text if char in '.,!?;:()[]{}')
        punctuation_density = punctuation_count / len(text)
        
        # Complexity score (0-1 scale)
        complexity = (
            (avg_word_length - 3) / 10 * 0.4 +  # Word complexity
            (avg_sentence_length - 10) / 20 * 0.4 +  # Sentence complexity
            punctuation_density * 0.2  # Punctuation complexity
        )
        
        return max(0, min(1, complexity))
    
    def generate_dataset(self, total_samples: int = 100) -> List[TestSample]:
        """Generate a diverse dataset of test samples"""
        samples = []
        
        # Define distribution across categories and lengths
        categories = list(self.base_sentences.keys())
        length_categories = ["short", "medium", "long", "very_long"]
        
        # Ensure we generate enough samples for each category-length combination
        samples_per_combination = max(1, total_samples // (len(categories) * len(length_categories)))
        
        sample_id = 1
        
        # Generate samples for each category-length combination
        for category in categories:
            for length_cat in length_categories:
                for _ in range(samples_per_combination):
                    # Get word count range for this length category
                    if length_cat == "short":
                        word_range = (10, 30)
                    elif length_cat == "medium":
                        word_range = (31, 80)
                    elif length_cat == "long":
                        word_range = (81, 150)
                    else:  # very_long
                        word_range = (151, 200)
                    
                    target_words = random.randint(*word_range)
                    base_sentence = random.choice(self.base_sentences[category])
                    
                    # Extend or trim sentence to target length
                    text = self._extend_sentence(base_sentence, target_words)
                    actual_word_count = len(text.split())
                    
                    # Calculate complexity score
                    complexity = self._calculate_complexity_score(text)
                    
                    sample = TestSample(
                        id=f"sample_{sample_id:03d}",
                        text=text,
                        word_count=actual_word_count,
                        category=category,
                        length_category=length_cat,
                        complexity_score=complexity
                    )
                    
                    samples.append(sample)
                    sample_id += 1
        
        # Fill remaining samples if needed
        while len(samples) < total_samples:
            category = random.choice(categories)
            length_cat = random.choice(length_categories)
            
            if length_cat == "short":
                word_range = (10, 30)
            elif length_cat == "medium":
                word_range = (31, 80)
            elif length_cat == "long":
                word_range = (81, 150)
            else:
                word_range = (151, 200)
            
            target_words = random.randint(*word_range)
            base_sentence = random.choice(self.base_sentences[category])
            text = self._extend_sentence(base_sentence, target_words)
            actual_word_count = len(text.split())
            complexity = self._calculate_complexity_score(text)
            
            sample = TestSample(
                id=f"sample_{sample_id:03d}",
                text=text,
                word_count=actual_word_count,
                category=category,
                length_category=length_cat,
                complexity_score=complexity
            )
            
            samples.append(sample)
            sample_id += 1
        
        self.samples = samples
        return samples
    
    def get_samples_by_category(self, category: str) -> List[TestSample]:
        """Get samples filtered by category"""
        return [s for s in self.samples if s.category == category]
    
    def get_samples_by_length(self, length_category: str) -> List[TestSample]:
        """Get samples filtered by length category"""
        return [s for s in self.samples if s.length_category == length_category]
    
    def get_random_sample(self) -> TestSample:
        """Get a random sample from the dataset"""
        if not self.samples:
            self.generate_dataset()
        return random.choice(self.samples)
    
    def export_dataset(self, filename: str):
        """Export dataset to JSON file"""
        data = []
        for sample in self.samples:
            data.append({
                "id": sample.id,
                "text": sample.text,
                "word_count": sample.word_count,
                "category": sample.category,
                "length_category": sample.length_category,
                "complexity_score": sample.complexity_score
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_dataset(self, filename: str):
        """Load dataset from JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        for item in data:
            sample = TestSample(
                id=item["id"],
                text=item["text"],
                word_count=item["word_count"],
                category=item["category"],
                length_category=item["length_category"],
                complexity_score=item["complexity_score"]
            )
            self.samples.append(sample)
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the current dataset"""
        if not self.samples:
            return {}
        
        stats = {
            "total_samples": len(self.samples),
            "categories": {},
            "length_categories": {},
            "word_count_stats": {
                "min": min(s.word_count for s in self.samples),
                "max": max(s.word_count for s in self.samples),
                "avg": sum(s.word_count for s in self.samples) / len(self.samples)
            },
            "complexity_stats": {
                "min": min(s.complexity_score for s in self.samples),
                "max": max(s.complexity_score for s in self.samples),
                "avg": sum(s.complexity_score for s in self.samples) / len(self.samples)
            }
        }
        
        # Count by category
        for sample in self.samples:
            stats["categories"][sample.category] = stats["categories"].get(sample.category, 0) + 1
            stats["length_categories"][sample.length_category] = stats["length_categories"].get(sample.length_category, 0) + 1
        
        return stats

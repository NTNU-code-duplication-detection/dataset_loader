"""
Module for code plagiarism detection using learned embeddings.

This module provides functionality to detect code plagiarism by comparing
code embeddings generated using transformers models.
"""

# pylint: disable=import-error

from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from transformers import pipeline
from src import pooling


from data.dataset_factory import get_dataset_generator


class CodeEmbeddingPipeline:
    """
    Pipeline for creating embeddings from code functions and detecting plagiarism.
    """

    def __init__(self, model_name: str = "microsoft/unixcoder-base"):
        """
        Initialize the pipeline with a specific model.

        @param model_name:HuggingFace model identifier
        """
        self.model_name = model_name
        self.pipe = pipeline("feature-extraction", model=model_name)

    def create_embedding(self, code: str) -> list:
        """
        Create embedding from code.

        @param code: Source code as string

        Returns:
            list: Embedding vectors from the model
        """
        # Get embeddings from pipeline (shape: 1, seq_len, hidden_size)
        output = self.pipe(code)
        return output

    def mean_pool(self, embeddings: list) -> np.ndarray:
        """
        Create mean pool from embedding vectors.

        Args:
            embeddings: Embedding vectors

        Returns:
            np.ndarray: Mean-pooled embedding vector
        """
        embedding = np.mean(embeddings[0], axis=0)
        return embedding

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            float: Cosine similarity score [0, 1]
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        cosine_sim = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
        return float(cosine_sim)

    def compare_code_strings(self, code_a: str, code_b: str) -> Tuple[float, bool]:
        """
        Compare two code strings for plagiarism detection.

        Args:
            code_a: First code string
            code_b: Second code string

        Returns:
            Tuple[float, bool]: (similarity_score, is_plagiarized)
        """
        # Create embeddings from code strings
        emb1 = self.create_embedding(code_a)
        emb2 = self.create_embedding(code_b)

        # Mean pool the embeddings
        mean1 = pooling.pool_embeddings(emb1)["mean"]
        mean2 = pooling.pool_embeddings(emb2)["mean"]

        # Calculate similarity
        similarity = self.compute_cosine_similarity(mean1, mean2)

        threshold = 0.90  # Threshold parameter. Defining what is plagiarism and what is not
        is_plagiarized = similarity >= threshold

        return similarity, is_plagiarized


class PlagiarismDetectionAnalyzer:
    """
    Analyzer for evaluating plagiarism detection using histogram analysis.
    """

    def __init__(self, embedding_pipeline: CodeEmbeddingPipeline):
        """
        Initialize the analyzer.

        Args:
            embedding_pipeline: CodeEmbeddingPipeline instance for generating embeddings
        """
        self.pipeline = embedding_pipeline
        self.plagiarized_scores: List[float] = []
        self.non_plagiarized_scores: List[float] = []

    def analyze_dataset(self, dataset_name: str = "sourcecodeplag"):
        """
        Analyze entire dataset and collect similarity scores using the dataset factory.

        Args:
            dataset_name: Name of dataset to use (e.g., 'sourcecodeplag', 'bigclonebench')
        """

        print('=' * 60)
        print('Analyzing Non-Plagiarized Pairs')
        print('=' * 60)

        # Get non-plagiarized generator from factory
        non_plag_generator = get_dataset_generator(
            dataset_name=dataset_name,
            mode="non_plagiarized"
        )

        for code_sample in non_plag_generator:
            # CodeSample has: code_a, code_b, label, dataset
            similarity, is_plag = self.pipeline.compare_code_strings(
                code_sample.code_a,
                code_sample.code_b
            )
            self.non_plagiarized_scores.append(similarity)
            print(f"Similarity: {similarity:.4f} | Plagiarized: {is_plag} | "
                  f"Label: {code_sample.label}")

        print('\n' + '=' * 60)
        print('Analyzing Plagiarized Pairs')
        print('=' * 60)

        # Get plagiarized generator from factory
        plag_generator = get_dataset_generator(
            dataset_name=dataset_name,
            mode="plagiarized",
        )

        for code_sample in plag_generator:
            # CodeSample has: code_a (original), code_b (plagiarized), label, dataset
            similarity, is_plag = self.pipeline.compare_code_strings(
                code_sample.code_a,
                code_sample.code_b
            )
            self.plagiarized_scores.append(similarity)
            print(f"Similarity: {similarity:.4f} | Plagiarized: {is_plag} | "
                  f"Label: {code_sample.label}")

    def plot_histogram(self, save_path: str = "similarity_histogram.png"):
        """
        Create histogram visualization of similarity scores.

        Args:
            save_path: Path to save the histogram plot
        """
        plt.figure(figsize=(12, 6))

        # Plot histograms
        plt.hist(self.non_plagiarized_scores, bins=20, alpha=0.5,
                 label='Non-Plagiarized', color='blue', density=True)
        plt.hist(self.plagiarized_scores, bins=20, alpha=0.5,
                 label='Plagiarized (Type-1)', color='red', density=True)

        # Add vertical line for threshold
        plt.axvline(x=0.90, color='green', linestyle='--',
                    linewidth=2, label='Threshold (0.90)')

        plt.xlabel('Cosine Similarity Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Similarity Scores: Plagiarized vs Non-Plagiarized',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\nHistogram saved to: {save_path}")
        plt.show()

    def compute_statistics(self):
        """
        Compute and print statistics for both groups.
        """
        print('\n' + '=' * 60)
        print('STATISTICS')
        print('=' * 60)

        print('\nNon-Plagiarized Pairs:')
        print(f"  Count: {len(self.non_plagiarized_scores)}")
        print(f"  Mean: {np.mean(self.non_plagiarized_scores):.4f}")
        print(f"  Std: {np.std(self.non_plagiarized_scores):.4f}")
        print(f"  Min: {np.min(self.non_plagiarized_scores):.4f}")
        print(f"  Max: {np.max(self.non_plagiarized_scores):.4f}")

        print('\nPlagiarized Pairs (Type-1):')
        print(f"  Count: {len(self.plagiarized_scores)}")
        print(f"  Mean: {np.mean(self.plagiarized_scores):.4f}")
        print(f"  Std: {np.std(self.plagiarized_scores):.4f}")
        print(f"  Min: {np.min(self.plagiarized_scores):.4f}")
        print(f"  Max: {np.max(self.plagiarized_scores):.4f}")

        # Calculate separation
        separation = np.mean(self.plagiarized_scores) - np.mean(self.non_plagiarized_scores)
        print(f"\nSeparation between groups: {separation:.4f}")


def main():
    """
    Main function to run the plagiarism detection pipeline.
    """
    # Initialize pipeline
    print("Initializing Code Embedding Pipeline...")
    embedding_pipeline = CodeEmbeddingPipeline(model_name="microsoft/unixcoder-base")

    # Initialize analyzer
    analyzer = PlagiarismDetectionAnalyzer(embedding_pipeline)

    # Analyze dataset using factory
    # You can change dataset_name to "bigclonebench" or "codexglue" if needed
    analyzer.analyze_dataset(
        dataset_name="sourcecodeplag",
    )

    # Compute statistics
    analyzer.compute_statistics()

    # Plot histogram
    analyzer.plot_histogram(save_path="type1_plagiarism_histogram.png")


if __name__ == "__main__":
    main()

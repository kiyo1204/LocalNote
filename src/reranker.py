"""
Reranking module for improving search result relevance.
Supports multiple reranking strategies: BGE, Cross-encoder, and no-op.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base class for rerankers"""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            
        Returns:
            List of (Document, score) tuples sorted by score (descending)
        """
        pass


class NoOpReranker(BaseReranker):
    """No-op reranker that returns documents with dummy scores"""
    
    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Return documents with uniform scores (no reranking).
        
        Args:
            query: Query string (ignored)
            documents: List of documents
            
        Returns:
            List of (Document, 1.0) tuples
        """
        return [(doc, 1.0) for doc in documents]


class BGEReranker(BaseReranker):
    """BGE (Baize General Embeddings) Reranker"""
    
    def __init__(self):
        """Initialize BGE reranker"""
        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
        except ImportError:
            logger.error("FlagEmbedding package not found. Install with: pip install FlagEmbedding")
            raise
    
    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Rerank documents using BGE reranker.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            
        Returns:
            List of (Document, score) tuples sorted by score (descending)
        """
        if not documents:
            return []
        
        # Prepare document texts
        doc_texts = [doc.page_content for doc in documents]
        
        try:
            # Get rerank scores
            scores = self.reranker.compute_score([[query, doc_text] for doc_text in doc_texts])
            
            # Pair documents with scores and sort
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return doc_score_pairs
        except Exception as e:
            logger.error(f"BGE reranking failed: {e}")
            # Fallback: return documents with original order
            return [(doc, 1.0) for doc in documents]


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranker using sentence-transformers"""
    
    def __init__(self, model_name: str = "cross-encoder/mmarco-MiniLMv2-L12-H384-v1"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model identifier for cross-encoder
        """
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(model_name)
        except ImportError:
            logger.error("sentence-transformers package not found. Install with: pip install sentence-transformers")
            raise
    
    def rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            
        Returns:
            List of (Document, score) tuples sorted by score (descending)
        """
        if not documents:
            return []
        
        try:
            # Prepare pairs (query, document_text)
            doc_texts = [doc.page_content for doc in documents]
            pairs = [[query, doc_text] for doc_text in doc_texts]
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Pair documents with scores and sort
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return doc_score_pairs
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Fallback: return documents with original order
            return [(doc, 1.0) for doc in documents]


class RerankerFactory:
    """Factory for creating reranker instances"""
    
    _rerankers: Dict[str, BaseReranker] = {}
    
    @staticmethod
    def get_reranker(reranker_type: str = "noop") -> BaseReranker:
        """
        Get or create a reranker instance.
        
        Args:
            reranker_type: Type of reranker ("noop", "bge", "cross-encoder")
            
        Returns:
            Reranker instance
        """
        reranker_type = reranker_type.lower()
        
        if reranker_type in RerankerFactory._rerankers:
            return RerankerFactory._rerankers[reranker_type]
        
        if reranker_type == "noop":
            reranker = NoOpReranker()
        elif reranker_type == "bge":
            reranker = BGEReranker()
        elif reranker_type == "cross-encoder":
            reranker = CrossEncoderReranker()
        else:
            logger.warning(f"Unknown reranker type: {reranker_type}, using NoOp")
            reranker = NoOpReranker()
        
        RerankerFactory._rerankers[reranker_type] = reranker
        return reranker
    
    @staticmethod
    def clear_cache():
        """Clear cached reranker instances"""
        RerankerFactory._rerankers.clear()

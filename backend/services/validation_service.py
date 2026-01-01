"""
Validation Service for the RAG chatbot system.

This service ensures answers are properly grounded and meet quality standards.
"""
from typing import List, Dict, Any
import re
import logging
from ..models.agent_inputs import QueryWithContext, ContextChunk
from ..models.agent_outputs import GeneratedAnswer, Citation


class ValidationService:
    """
    Service class that handles validation of answers and grounding enforcement.
    """

    def __init__(self):
        """
        Initialize the Validation Service.
        """
        self.logger = logging.getLogger(__name__)

    def validate_answer_grounding(self, answer: str, context_chunks: List[ContextChunk]) -> Dict[str, Any]:
        """
        Validate that the answer is properly grounded in the provided context.

        Args:
            answer: The generated answer to validate
            context_chunks: The context chunks used to generate the answer

        Returns:
            Dict with validation results including grounding score and issues
        """
        # Check if answer content appears in context chunks
        grounding_issues = []
        consistency_issues = []

        # Calculate grounding score using token overlap
        grounding_result = self._calculate_token_overlap_grounding(answer, context_chunks)
        grounding_score = grounding_result["grounding_score"]

        # Check for consistency between answer and context
        consistency_score = self._calculate_consistency_score(answer, context_chunks)

        # Check for hallucinations (content that contradicts context)
        hallucination_issues = self._detect_hallucinations(answer, context_chunks)

        # Check if answer directly contradicts context
        contradiction_issues = self._detect_contradictions(answer, context_chunks)

        # Aggregate issues
        grounding_issues.extend(grounding_result["issues"])
        grounding_issues.extend(hallucination_issues)
        grounding_issues.extend(contradiction_issues)

        # Determine if answer is acceptable
        is_acceptable = (
            grounding_score >= 0.3 and  # At least 30% of content is grounded
            consistency_score >= 0.3 and  # At least 30% consistency with context
            not contradiction_issues  # No direct contradictions
        )

        return {
            "grounding_score": grounding_score,
            "consistency_score": consistency_score,
            "is_acceptable": is_acceptable,
            "grounding_issues": grounding_issues,
            "consistency_issues": consistency_issues,
            "hallucination_issues": hallucination_issues,
            "contradiction_issues": contradiction_issues
        }

    def _calculate_token_overlap_grounding(self, answer: str, context_chunks: List[ContextChunk]) -> Dict[str, Any]:
        """
        Calculate grounding score based on token overlap between answer and context.

        Args:
            answer: The generated answer
            context_chunks: The context chunks used to generate the answer

        Returns:
            Dict with grounding score and related information
        """
        grounding_issues = []

        # Simple token overlap check
        answer_tokens = set(answer.lower().split())
        if not answer_tokens:
            return {
                "grounding_score": 0.0,
                "grounded_tokens": 0,
                "total_tokens": 0,
                "issues": ["Answer is empty"]
            }

        all_context_tokens = set()
        for chunk in context_chunks:
            chunk_tokens = set(chunk.content.lower().split())
            all_context_tokens.update(chunk_tokens)

        overlap = answer_tokens.intersection(all_context_tokens)
        overlap_count = len(overlap)
        total_answer_tokens = len(answer_tokens)

        grounding_score = overlap_count / total_answer_tokens if total_answer_tokens > 0 else 0.0

        # Check for grounding issues
        if grounding_score < 0.3:  # Less than 30% of tokens are grounded
            grounding_issues.append(f"Low grounding score: {grounding_score:.2f} - answer may contain ungrounded content")
        elif grounding_score > 0.9:  # Very high grounding might indicate copying
            grounding_issues.append(f"Very high grounding score: {grounding_score:.2f} - answer might be copying context verbatim")

        return {
            "grounding_score": grounding_score,
            "grounded_tokens": overlap_count,
            "total_tokens": total_answer_tokens,
            "issues": grounding_issues
        }

    def _calculate_consistency_score(self, answer: str, context_chunks: List[ContextChunk]) -> float:
        """
        Calculate consistency score between answer and context.

        Args:
            answer: The generated answer
            context_chunks: The context chunks used to generate the answer

        Returns:
            Consistency score between 0.0 and 1.0
        """
        # This is a simplified approach - in a real implementation, you might use
        # semantic similarity or other more sophisticated methods
        if not context_chunks:
            return 0.0

        answer_lower = answer.lower()
        context_text = " ".join([chunk.content.lower() for chunk in context_chunks])

        # Simple keyword overlap as a proxy for consistency
        answer_words = set(answer_lower.split())
        context_words = set(context_text.split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words.intersection(context_words))
        consistency_score = overlap / len(answer_words)

        return consistency_score

    def _detect_hallucinations(self, answer: str, context_chunks: List[ContextChunk]) -> List[str]:
        """
        Detect potential hallucinations in the answer.

        Args:
            answer: The generated answer
            context_chunks: The context chunks used to generate the answer

        Returns:
            List of hallucination issues detected
        """
        hallucination_issues = []

        # Look for claims in the answer that contradict or go beyond the context
        # This is a simplified approach - real implementation would use more sophisticated NLP techniques
        answer_lower = answer.lower()
        context_text = " ".join([chunk.content.lower() for chunk in context_chunks])

        # Check for certainty indicators in answer that aren't supported by context
        certainty_indicators = [
            "definitely", "certainly", "absolutely", "without doubt", "always", "never",
            "all", "none", "every", "no doubt", "clearly", "obviously"
        ]

        for indicator in certainty_indicators:
            if indicator in answer_lower and indicator not in context_text:
                # Check if the certainty is about something that's actually in the context
                # This is a simplified check - in practice, you'd need more sophisticated semantic analysis
                hallucination_issues.append(f"Answer contains certainty indicator '{indicator}' not supported by context")

        return hallucination_issues

    def _detect_contradictions(self, answer: str, context_chunks: List[ContextChunk]) -> List[str]:
        """
        Detect contradictions between the answer and context.

        Args:
            answer: The generated answer
            context_chunks: The context chunks used to generate the answer

        Returns:
            List of contradiction issues detected
        """
        contradiction_issues = []

        # This is a simplified approach - a real implementation would require more sophisticated NLP
        # to detect semantic contradictions
        answer_lower = answer.lower()
        context_lower = " ".join([chunk.content.lower() for chunk in context_chunks])

        # Look for negation patterns that might contradict the context
        negation_patterns = [
            "not true", "incorrect", "false", "wrong", "contrary to", "opposite of"
        ]

        for pattern in negation_patterns:
            if pattern in answer_lower:
                contradiction_issues.append(f"Answer contains negation pattern '{pattern}' that might contradict context")

        return contradiction_issues

    def extract_citations(self, answer: str, context_chunks: List[ContextChunk]) -> List[Citation]:
        """
        Extract citations from the answer referencing the context chunks.

        Args:
            answer: The generated answer
            context_chunks: The context chunks used to generate the answer

        Returns:
            List of Citation objects
        """
        citations = []

        # Look for references to context chunks in the answer
        for chunk in context_chunks:
            # Check if chunk content appears in answer (simple substring match)
            if self._content_appears_in_answer(chunk.content, answer):
                # Create citation with a relevant excerpt from the answer
                excerpt = self._extract_relevant_excerpt(answer, chunk.content)

                citation = Citation(
                    chunk_id=chunk.id,
                    source_url=chunk.source_url,
                    title=chunk.title,
                    excerpt=excerpt[:200],  # Limit excerpt to 200 chars
                    confidence=chunk.score  # Use the retrieval score as citation confidence
                )
                citations.append(citation)
            # Also check for citations that might be mentioned in the answer text
            elif self._citation_mentioned_in_answer(answer, chunk):
                # If the source is mentioned in the answer, create a citation
                excerpt = self._extract_context_mentioned_excerpt(answer, chunk)

                citation = Citation(
                    chunk_id=chunk.id,
                    source_url=chunk.source_url,
                    title=chunk.title,
                    excerpt=excerpt[:200],
                    confidence=chunk.score * 0.8  # Slightly lower confidence if only mentioned
                )
                citations.append(citation)

        return citations

    def _extract_context_mentioned_excerpt(self, answer: str, chunk: ContextChunk) -> str:
        """
        Extract an excerpt from the answer that mentions the context chunk.

        Args:
            answer: The generated answer
            chunk: The context chunk that was mentioned

        Returns:
            An excerpt from the answer that mentions the chunk
        """
        # Find the part of the answer that mentions the chunk's title or URL
        lower_answer = answer.lower()
        lower_title = chunk.title.lower()
        lower_url = chunk.source_url.lower()

        # Find the position of the title or URL in the answer
        pos = -1
        if lower_title in lower_answer:
            pos = lower_answer.find(lower_title)
        elif lower_url in lower_answer:
            pos = lower_answer.find(lower_url)

        if pos != -1:
            # Extract a 100-character excerpt around the mention
            start = max(0, pos - 50)
            end = min(len(answer), pos + len(chunk.title) + 50)
            return answer[start:end]

        # If no specific mention found, return the first 100 characters
        return answer[:100] if len(answer) > 100 else answer

    def _citation_mentioned_in_answer(self, answer: str, chunk: ContextChunk) -> bool:
        """
        Check if a citation to the chunk is mentioned in the answer.

        Args:
            answer: The generated answer
            chunk: The context chunk to check for

        Returns:
            True if the chunk is mentioned in the answer, False otherwise
        """
        # Check for mentions of the title, URL, or other identifying information
        lower_answer = answer.lower()
        lower_title = chunk.title.lower()
        lower_url = chunk.source_url.lower()

        # Check if title or URL appears in the answer
        return lower_title in lower_answer or lower_url in lower_answer

    def calculate_confidence_score(self, answer: str, context_chunks: List[ContextChunk]) -> float:
        """
        Calculate a confidence score for the generated answer based on context alignment.

        Args:
            answer: The generated answer
            context_chunks: The context chunks used to generate the answer

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not context_chunks:
            return 0.0

        # Use the average of the top context chunk scores as the base confidence
        scores = [chunk.score for chunk in context_chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Adjust based on grounding validation
        grounding_result = self.validate_answer_grounding(answer, context_chunks)
        grounding_score = grounding_result["grounding_score"]

        # Combine the scores (weighted average)
        # Context relevance (40%) + Grounding quality (60%)
        confidence = (avg_score * 0.4) + (grounding_score * 0.6)

        return confidence

    def handle_insufficient_context(self, query: str, context_chunks: List[ContextChunk]) -> str:
        """
        Generate an appropriate response when context is insufficient to answer the query.

        Args:
            query: The user's query
            context_chunks: The context chunks retrieved (may be insufficient)

        Returns:
            A response indicating insufficient context
        """
        # Check if we have any context chunks at all
        if not context_chunks:
            return (
                f"I cannot answer your question '{query}' because no relevant content "
                f"was found in the book. The information you're looking for may not be "
                f"covered in the available material."
            )

        # Check if the context chunks have low relevance scores
        avg_score = sum(chunk.score for chunk in context_chunks) / len(context_chunks)
        if avg_score < 0.3:  # Low relevance
            return (
                f"I found some content related to your question '{query}', but the "
                f"relevance is quite low. The available information may not adequately "
                f"address your query. You might need to try rephrasing your question "
                f"or check if the topic is covered in other parts of the book."
            )

        # Check if the context chunks are too short or don't contain relevant information
        total_content_length = sum(len(chunk.content) for chunk in context_chunks)
        if total_content_length < 100:  # Very little content
            return (
                f"The content found for your question '{query}' is very limited. "
                f"There isn't enough information in the retrieved sections to provide "
                f"a comprehensive answer."
            )

        # Default insufficient context response
        return (
            f"Based on the available content, I cannot provide a complete answer to "
            f"your question '{query}'. The retrieved information is insufficient to "
            f"address your query adequately."
        )

    def _content_appears_in_answer(self, chunk_content: str, answer: str) -> bool:
        """
        Check if content from a chunk appears in the answer (with some tolerance for variations).

        Args:
            chunk_content: Content from the context chunk
            answer: The generated answer

        Returns:
            True if content appears in answer, False otherwise
        """
        # Use a simple approach: check if at least 30% of the chunk content appears in the answer
        chunk_words = set(chunk_content.lower().split())
        answer_words = set(answer.lower().split())

        if len(chunk_words) == 0:
            return False

        overlap = len(chunk_words.intersection(answer_words))
        overlap_ratio = overlap / len(chunk_words)

        return overlap_ratio >= 0.1  # At least 10% overlap

    def _extract_relevant_excerpt(self, answer: str, context_content: str) -> str:
        """
        Extract a relevant excerpt from the answer that relates to the context.

        Args:
            answer: The generated answer
            context_content: The context content that was used

        Returns:
            A relevant excerpt from the answer
        """
        # For now, return the first 100 characters of the answer
        # In a more sophisticated implementation, this would extract the most relevant portion
        return answer[:200] if len(answer) > 200 else answer
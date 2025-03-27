from src.services.search.embedding import Embedding
import numpy as np
from typing import List
class GroupSentence:
    
    @classmethod
    def _group(cls,
               sentences: list[str],
               threshold: float = 0.5) -> list[str]:
        """
        Group similar sentences based on semantic similarity.
        Process:
        1. Start with first sentence
        2. Compare with next sentence, if similarity > threshold, merge them
        3. Get embedding for merged text, compare with next sentence
        4. Continue merging if similarity > threshold, otherwise start new group
        
        Args:
            sentences (list[str]): List of sentences to group
            threshold (float): Similarity threshold to determine if sentences should be grouped
            
        Returns:
            list[str]: List of merged sentence groups
        """
        if not sentences:
            return []
        
        # Initialize Embedding model if needed
        Embedding.initialize()
        
        result = []
        i = 0
        
        while i < len(sentences):
            # Start a new group with the current sentence
            current_group = sentences[i]
            
            # Get embedding for current group
            current_embedding, _ = Embedding.get_embeddings([current_group])
            
            # Check if we can merge with subsequent sentences
            next_idx = i + 1
            
            # Keep merging as long as similarity is above threshold
            merged = False
            
            while next_idx < len(sentences):
                next_sentence = sentences[next_idx]
                next_embedding, _ = Embedding.get_embeddings([next_sentence])
                
                # Calculate similarity between current group and next sentence
                similarity = np.dot(current_embedding, next_embedding.T)[0][0]
                
                if similarity >= threshold:
                    # Merge sentences
                    current_group = f"{current_group} {next_sentence}"
                    
                    # Recalculate embedding for merged group
                    current_embedding, _ = Embedding.get_embeddings([current_group])
                    
                    # Mark that we've merged something
                    merged = True
                    
                    # Move to next sentence
                    next_idx += 1
                else:
                    # Cannot merge, stop here
                    break
            
            # Add the current group to results
            result.append(current_group)
            
            # Skip to the next unprocessed sentence
            i = next_idx
        
        return result
    
    @classmethod
    def group_sentences(cls, 
                      sentences: list[str], 
                      threshold: float = 0.5) -> list[str]:
        """
        Public method to group similar sentences.
        
        Args:
            sentences (list[str]): List of sentences to group
            threshold (float): Similarity threshold (between 0 and 1)
            
        Returns:
            list[str]: List of merged sentence groups
        """
        # Validate input
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            raise ValueError("sentences must be a list of strings")
            
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
            
        return cls._group(sentences, threshold)
    
# if __name__ == "__main__":
#     from src.services.text_process.extraction import PDFTextExtractor
#     pdf_path = "data/GVR_Baocaothuongnien_2023.pdf"
#     ouput_path = f"output/{pdf_path[5:-4]}.txt"
#     pdf_text = PDFTextExtractor.extract_text_from_pdf(pdf_path, ouput_path)
#     print(pdf_text)
    
#     sentences = PDFTextExtractor.sentence_segmentation(pdf_text)
#     g_sentences = GroupSentence.group_sentences(sentences)
#     with open(f"output/g_sentence_{pdf_path[5:-4]}.jsonl", "w") as out_file:
#         for sentence in sentences:
#             s = sentence.replace("\n", " ") 
#             out_file.write(f"\"g_sentence\":\"{s}\"\n")
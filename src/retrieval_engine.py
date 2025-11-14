import asyncio

from typing import List, Dict, Optional, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from config import CONFIG, BEHAVIOR_DESCRIPTIONS, BEHAVIOR_QUERIES
from logger import logger
from llm import LLM
from langfuse import get_client

langfuse = get_client()
llm = LLM()

class MalwareRetrievalEngine:
    """Retrieval engine for Smali malware analysis."""
    
    def __init__(self, vectorstore_path: str = None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["huggingface"]["embedding_model"],
            query_encode_kwargs={"prompt_name": "query", "normalize_embeddings": True},
            encode_kwargs={"prompt_name": "document", "normalize_embeddings": True},
        )
        
        if vectorstore_path:
            self.vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=self.embeddings,
                collection_name=CONFIG["vectorstore"]["collection_name"],
            )
        else:
            self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]):
        """Create and persist vector store from documents."""
        logger.debug(f"Creating vector store with {len(documents)} documents")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=CONFIG["vectorstore"]["persist_directory"],
            collection_name=CONFIG["vectorstore"]["collection_name"]
        )
        logger.info(f"Vector store created with {len(documents)} documents")
    
    async def retrieve_classes_for_behavior(self, behavior_id: int) -> List[Dict]:
        """Retrieve relevant classes for a specific behavior using two-stage retrieval."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        # Stage 1: Vector similarity search with behavior-specific query
        behavior_query = BEHAVIOR_QUERIES[behavior_id]
        behavior_description = BEHAVIOR_DESCRIPTIONS[behavior_id]
        
        # Get more candidates for re-ranking
        initial_candidates = CONFIG["retrieval"]["top_k_classes"] * 2
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            behavior_query,
            k=initial_candidates
        )
        
        # Stage 2: LLM re-ranking and explanation
        re_ranked_results = []
        seen_signatures = set()
        filtered_docs_and_scores = []
        for doc, score in docs_and_scores:
            class_signature = f"L{doc.metadata['class_name']};"
            if class_signature in seen_signatures:
                continue
            seen_signatures.add(class_signature)
            filtered_docs_and_scores.append((doc, score))
        
        tasks = [
            self._assess_class_relevance(doc, behavior_id, behavior_description, score)
            for doc, score in filtered_docs_and_scores
        ]

        relevance_results = await asyncio.gather(*tasks, return_exceptions=True)
        for (doc, score), re_ranked_result in zip(filtered_docs_and_scores, relevance_results):
            if isinstance(re_ranked_result, Exception):
                logger.error(f"Error re-ranking class {doc.metadata['class_name']}: {re_ranked_result}")
                continue
            relevance_score, explanation = re_ranked_result
            if relevance_score >= CONFIG["retrieval"]["relevance_threshold"]:
                class_signature = f"L{doc.metadata['class_name']};"
                re_ranked_results.append({
                    'class_name': doc.metadata['class_name'],
                    'class_signature': class_signature,
                    'vector_similarity_score': score,
                    'llm_relevance_score': relevance_score,
                    'explanation': explanation,
                    'metadata': doc.metadata
                })
        
        # Sort by LLM relevance score and return top_k
        re_ranked_results.sort(key=lambda x: x['llm_relevance_score'], reverse=True)
        return re_ranked_results[:CONFIG["retrieval"]["top_k_classes"]]
    
    async def analyze_methods_in_class(self, class_result: Dict, behavior_id: int) -> List[Dict]:
        """Analyze methods within a class to identify those involved in the behavior."""
        behavior_description = BEHAVIOR_DESCRIPTIONS[behavior_id]
        
        # Get raw content from metadata
        raw_content = class_result['metadata']['raw_content']
        first_stage_explanation = class_result['explanation']
        
        # Analyze all methods together using the entire class context
        involved_methods = await self._analyze_methods_with_class_context(
            raw_content, behavior_id, behavior_description, first_stage_explanation
        )
        
        return involved_methods[:CONFIG["retrieval"]["top_k_methods_per_class"]]
    
    async def _assess_class_relevance(self, doc: Document, behavior_id: int, behavior_description: str, vector_score: float) -> Tuple[float, str]:
        """Use LLM to assess class relevance to a specific behavior and provide explanation."""
        try:
            prompt = langfuse.get_prompt(CONFIG["langfuse"]["prompt_names"]["class_relevance_prompt"]).compile(
                smali_class_content=doc.metadata['raw_content'],
                behavior_id=behavior_id,
                behavior_description=behavior_description
            )
            assert "{{" not in prompt, "Unresolved placeholders in class relevance prompt"
            result = await llm.generate_text(
                system_prompt=langfuse.get_prompt(CONFIG["langfuse"]["prompt_names"]["class_relevance_system_prompt"]).compile(),
                prompt=prompt,
                temperature=CONFIG["openai"]["temperature"],
                max_tokens=CONFIG["openai"]["max_tokens"]
            )
                        
            # Parse the response to extract score and explanation
            lines = result.split('\n')
            score = 0.0
            explanation = "No explanation available."
            
            for line in lines:
                if line.startswith('Score:'):
                    try:
                        score = float(line.split(':')[1].strip())
                    except:
                        score = 0.0
                elif line.startswith('Explanation:'):
                    explanation = line.split(':', 1)[1].strip()
                elif line.startswith('Relevant APIs:'):
                    apis = line.split(':', 1)[1].strip()
                    if apis and apis != "None":
                        explanation += f" Relevant APIs: {apis}"
            
            return score, explanation
            
        except Exception as e:
            logger.error(f"Error assessing class relevance: {e}")
            return 0.0, f"Error in assessment. Vector similarity score: {vector_score:.3f}"
    
    async def _analyze_methods_with_class_context(self, class_content: str, behavior_id: int, behavior_description: str, first_stage_explanation: str) -> List[Dict]:
        """Analyze all methods in a class together using the entire class context."""
        try:
            prompt = langfuse.get_prompt(CONFIG["langfuse"]["prompt_names"]["method_analysis_prompt"]).compile(
                first_stage_explanation=first_stage_explanation,
                smali_class_content=class_content
            )
            assert "{{" not in prompt, "Unresolved placeholders in method analysis prompt"
            result = await llm.generate_text(
                system_prompt=langfuse.get_prompt(CONFIG["langfuse"]["prompt_names"]["method_analysis_system_prompt"]).compile(),
                prompt=prompt,
                temperature=CONFIG["openai"]["temperature"],
                max_tokens=CONFIG["openai"]["max_tokens"]
            )
                        
            # Parse the response to extract methods and their roles
            involved_methods = self._parse_method_analysis_response(result)
            
            return involved_methods
            
        except Exception as e:
            logger.error(f"Error analyzing methods with class context: {e}")
            return []
    
    def _parse_method_analysis_response(self, response: str) -> List[Dict]:
        """Parse the LLM response to extract method information."""
        methods = []
        current_method = {}
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('METHOD:'):
                # Save previous method if exists
                if current_method and 'method_signature' in current_method:
                    methods.append(current_method)
                
                # Start new method
                method_signature = line.split('METHOD:', 1)[1].strip()
                current_method = {
                    'method_signature': method_signature,
                    'method_name': self._extract_method_name(method_signature),
                    'relevance_score': 0.0,
                    'role_explanation': '',
                    'method_content': ''
                }
            
            elif line.startswith('ROLE:') and current_method:
                role = line.split('ROLE:', 1)[1].strip()
                current_method['role_explanation'] = role
            
            elif line.startswith('CONFIDENCE:') and current_method:
                try:
                    confidence = int(line.split('CONFIDENCE:', 1)[1].strip())
                    current_method['relevance_score'] = confidence / 100.0  # Convert to 0-1 scale
                except:
                    current_method['relevance_score'] = 0.5  # Default confidence
        
        # Add the last method
        if current_method and 'method_signature' in current_method:
            methods.append(current_method)
        
        # Sort by relevance score
        methods.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return methods
    
    def _extract_method_name(self, method_signature: str) -> str:
        """Extract method name from Smali method signature."""
        try:
            # Extract method name from signature like ".method public onCreateView(...)"
            parts = method_signature.split('(')
            if len(parts) >= 1:
                method_part = parts[0]
                # Remove ".method" and modifiers
                method_part = method_part.replace('.method', '').strip()
                # Get the last part which should be the method name
                name_parts = method_part.split()
                if name_parts:
                    return name_parts[-1]
        except:
            pass
        return "unknown_method"
    
    
    def _extract_methods_from_content(self, content: str) -> List[Dict]:
        """Extract method information from raw Smali content."""
        import re
        
        methods = []
        method_pattern = re.compile(r'\.method\s+(?:public|private|protected)?\s+(?:static|final|abstract)?\s+([^(]+)\(([^)]*)\)([^;]+);')
        
        matches = method_pattern.finditer(content)
        for match in matches:
            method_name = match.group(1).strip()
            params = match.group(2).strip()
            return_type = match.group(3).strip()
            signature = f"{method_name}({params}){return_type}"
            
            # Extract method content
            method_start = match.start()
            method_end = self._find_method_end(content, method_start)
            method_content = content[method_start:method_end] if method_end else ""
            
            methods.append({
                'name': method_name,
                'signature': signature,
                'content': method_content
            })
        
        return methods
    
    def _find_method_end(self, content: str, start_pos: int) -> int:
        """Find the end of a method."""
        remaining = content[start_pos:]
        lines = remaining.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip() == '.end method':
                return start_pos + len('\n'.join(lines[:i+1]))
        
        return len(content)

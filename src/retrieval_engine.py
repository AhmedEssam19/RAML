import os
from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from config import CONFIG, BEHAVIOR_DESCRIPTIONS, BEHAVIOR_QUERIES
from logger import logger
import json
from openai import OpenAI

openai_client = OpenAI(api_key=CONFIG["openai"]["api_key"])

class MalwareRetrievalEngine:
    """Retrieval engine for Smali malware analysis."""
    
    def __init__(self, vectorstore_path: str = None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["huggingface"]["embedding_model"],
            query_encode_kwargs={"prompt_name": "query"}
        )
        
        if vectorstore_path:
            self.vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=self.embeddings,
                collection_name=CONFIG["vectorstore"]["collection_name"]
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
        self.vectorstore.persist()
        logger.info(f"Vector store created with {len(documents)} documents")
    
    def retrieve_classes_for_behavior(self, behavior_id: int) -> List[Dict]:
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
        
        for doc, score in docs_and_scores:
            class_signature = f"L{doc.metadata['class_name']};"
            if class_signature in seen_signatures:
                continue
            seen_signatures.add(class_signature)
            
            # Assess relevance using LLM
            relevance_score, explanation = self._assess_class_relevance(
                doc, behavior_id, behavior_description, score
            )
            
            if relevance_score >= 0.5:  # Threshold for relevance
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
    
    def analyze_methods_in_class(self, class_result: Dict, behavior_id: int) -> List[Dict]:
        """Analyze methods within a class to identify those involved in the behavior."""
        behavior_description = BEHAVIOR_DESCRIPTIONS[behavior_id]
        
        # Get raw content from metadata
        raw_content = class_result['metadata']['raw_content']
        first_stage_explanation = class_result['explanation']
        
        # Analyze all methods together using the entire class context
        involved_methods = self._analyze_methods_with_class_context(
            raw_content, behavior_id, behavior_description, first_stage_explanation
        )
        
        return involved_methods[:CONFIG["retrieval"]["top_k_methods_per_class"]]
    
    def _assess_class_relevance(self, doc: Document, behavior_id: int, behavior_description: str, vector_score: float) -> Tuple[float, str]:
        """Use LLM to assess class relevance to a specific behavior and provide explanation."""
        try:
            prompt = f"""
            You are an expert in Android malware analysis. Analyze the following Smali class:
            {doc.metadata['raw_content']}
            
            Does this class perform Behavior {behavior_id}: {behavior_description}?
            
            Provide:
            1. A relevance score from 0.0 to 1.0 (where 1.0 = highly relevant)
            2. A brief explanation of why it is or isn't relevant
            3. If relevant, list the specific APIs/methods that indicate this behavior
            
            Format your response as:
            Score: [0.0-1.0]
            Explanation: [your explanation]
            Relevant APIs: [list of relevant APIs/methods if any]
            """
            
            response = openai_client.chat.completions.create(
                model=CONFIG["openai"]["model"],
                messages=[
                    {"role": "system", "content": "You are a malware analyst. Assess class relevance to specific malicious behaviors and provide detailed explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=CONFIG["openai"]["temperature"],
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
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
    
    def _analyze_methods_with_class_context(self, class_content: str, behavior_id: int, behavior_description: str, first_stage_explanation: str) -> List[Dict]:
        """Analyze all methods in a class together using the entire class context."""
        try:
            prompt = f"""
You are an expert in Android malware analysis. The following Smali class has been identified as implementing one or several malicious behaviors in the first stage. 
Analyze the class and identify all methods that are involved in implementing these behaviors.

First Stage Explanation of Identified Malicious Behavior(s):
{first_stage_explanation}

Smali Class:
{class_content}

IMPORTANT: For each method involved in the behavior, output the following fields, one per line, for each method:
METHOD: <the first line of the method exactly as it appears in the Smali code, including all access modifiers, names, and parameters>
ROLE: <role description>
CONFIDENCE: <confidence score 0-100>

After listing all methods, provide a final explanation as follows:
EXPLANATION: <detailed explanation of how these methods work together to implement the behavior>

Example output format:
METHOD: .method public onCreateView(Landroid/view/LayoutInflater;Landroid/view/ViewGroup;Landroid/os/Bundle;)Landroid/view/View;
ROLE: This method inflates the view and sets up the UI components
CONFIDENCE: 90

METHOD: .method synthetic lambda$onCreateView$2$lu-snt-trux-koopaapp-ui-home-RequestData2Fragment(Landroid/view/View;)V
ROLE: This method handles click events on the view
CONFIDENCE: 85

EXPLANATION: These methods work together to...
"""
            
            response = openai_client.chat.completions.create(
                model=CONFIG["openai"]["model"],
                messages=[
                    {"role": "system", "content": "You are a malware analyst specializing in Android Smali code analysis. Analyze class methods in context and identify their roles in malicious behaviors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=CONFIG["openai"]["temperature"],
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
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

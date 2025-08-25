import re
import os
from typing import List, Dict, Optional, Tuple
from logger import logger

class SmaliParser:
    """Parser for Smali files to extract class and method information."""
    
    def __init__(self):
        # Updated pattern to handle all class types including synthetic with final modifier
        # Using a simpler, more robust pattern that works with all class definitions
        self.class_pattern = re.compile(r'\.class\s+.*?L([^;]+);', re.MULTILINE)
        self.method_pattern = re.compile(r'\.method\s+(?:public|private|protected)?\s+(?:static|final|abstract|synthetic)?\s+([^(]+)\(([^)]*)\)([^;]+);')
        self.permission_pattern = re.compile(r'android\.permission\.([A-Z_]+)')
        self.api_pattern = re.compile(r'L([^;]+);')
        
    def clean_smali_content(self, content: str) -> str:
        """Remove unnecessary lines (comments, blank lines, debug/metadata) from Smali content."""
        cleaned_lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue  # Skip blank lines
            if stripped.startswith('#'):
                continue  # Skip comments
            if stripped.startswith(('.line', '.source', '.prologue', '.local', '.end local', '.param', '.end param', '.restart local', '.end method', '.annotation', '.end annotation', '.catch', '.catchall', '.registers', '.end field', '.end subannotation', '.end array-data', '.array-data', '.packed-switch', '.end packed-switch', '.sparse-switch', '.end sparse-switch')):
                continue  # Skip debug/metadata lines
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def parse_smali_file(self, file_path: str) -> Dict:
        """Parse a single Smali file and extract class information."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Clean unnecessary lines
            content = self.clean_smali_content(content)
            # Extract class information
            class_match = self.class_pattern.search(content)
            if not class_match:
                logger.log_file_processing(file_path, "error", "No class definition found")
                return None
            class_name = class_match.group(1)
            # Check if this is a synthetic class
            is_synthetic = self._is_synthetic_class(content, class_name)
            if is_synthetic:
                logger.log_file_processing(file_path, "synthetic", f"Class: {class_name}")
            # Extract methods
            methods = self._extract_methods(content)
            # Extract permissions and API calls
            permissions = self._extract_permissions(content)
            api_calls = self._extract_api_calls(content)
            parsed_class = {
                'class_name': class_name,
                'file_path': file_path,
                'methods': methods,
                'permissions': permissions,
                'api_calls': api_calls,
                'raw_content': content,
                'is_synthetic': is_synthetic
            }
            logger.log_file_processing(file_path, "success", f"Class: {class_name}, Methods: {len(methods)}")
            return parsed_class
        except Exception as e:
            logger.log_file_processing(file_path, "error", str(e))
            return None
    
    def _is_synthetic_class(self, content: str, class_name: str) -> bool:
        """Determine if a class is synthetic (compiler-generated or lambda)."""
        # Check for synthetic keyword in class definition
        if '.class' in content and 'synthetic' in content:
            return True
        
        # Check for lambda patterns in class name
        lambda_patterns = [
            r'\$\$ExternalSyntheticLambda\d+',
            r'\$\$Lambda\$',
            r'\$\$Lambda\d+',
            r'\$\$SyntheticLambda',
            r'\$\$Generated'
        ]
        
        for pattern in lambda_patterns:
            if re.search(pattern, class_name):
                return True
        
        # Check for D8$$SyntheticClass source
        if 'D8$$SyntheticClass' in content:
            return True
        
        return False
    
    def _extract_methods(self, content: str) -> List[Dict]:
        """Extract method information from Smali content."""
        methods = []
        method_matches = self.method_pattern.finditer(content)
        
        for match in method_matches:
            method_name = match.group(1).strip()
            params = match.group(2).strip()
            return_type = match.group(3).strip()
            
            # Create method signature
            signature = f"{method_name}({params}){return_type}"
            
            # Extract method content
            method_start = match.start()
            method_end = self._find_method_end(content, method_start)
            method_content = content[method_start:method_end] if method_end else ""
            
            methods.append({
                'name': method_name,
                'signature': signature,
                'params': params,
                'return_type': return_type,
                'content': method_content
            })
        
        return methods
    
    def _find_method_end(self, content: str, start_pos: int) -> Optional[int]:
        """Find the end of a method (next .end method)."""
        remaining = content[start_pos:]
        lines = remaining.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip() == '.end method':
                return start_pos + len('\n'.join(lines[:i+1]))
        
        return None
    
    def _extract_permissions(self, content: str) -> List[str]:
        """Extract Android permissions from Smali content."""
        permissions = set()
        matches = self.permission_pattern.finditer(content)
        
        for match in matches:
            permissions.add(match.group(1))
        
        return list(permissions)
    
    def _extract_api_calls(self, content: str) -> List[str]:
        """Extract API calls from Smali content."""
        api_calls = set()
        matches = self.api_pattern.finditer(content)
        
        for match in matches:
            api_call = match.group(1)
            # Filter out common non-API patterns
            if not api_call.startswith('java/lang/') and len(api_call.split('/')) > 1:
                api_calls.add(api_call)
        
        return list(api_calls)
    
    def get_class_summary(self, parsed_class: Dict) -> str:
        """Generate a detailed summary of the class for description generation, including all method signatures and content snippets."""
        summary_parts = []
        class_type = "Synthetic" if parsed_class.get('is_synthetic', False) else "Regular"
        summary_parts.append(f"Class: {parsed_class['class_name']} ({class_type})")
        summary_parts.append(f"Methods: {len(parsed_class['methods'])}")
        if parsed_class['methods']:
            for m in parsed_class['methods']:
                content_snippet = '\n'.join(m['content'].splitlines()[:10])[:500]
                summary_parts.append(f"Method: {m['signature']}\n{content_snippet}")
        if parsed_class['permissions']:
            summary_parts.append(f"Permissions: {', '.join(parsed_class['permissions'])}")
        if parsed_class['api_calls']:
            summary_parts.append(f"API calls: {', '.join(parsed_class['api_calls'][:10])}")
        return '\n\n'.join(summary_parts) 
import os
import json
import asyncio

from typing import List, Dict
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from smali_parser import SmaliParser
from config import CONFIG
from logger import logger
from llm import LLM
from langfuse import get_client
 
langfuse = get_client()
# Initialize OpenAI client
llm = LLM()

class SmaliFolderLoader(BaseLoader):
    """LangChain document loader for Smali files in folders."""
    
    def __init__(self, folder_path: str, save_descriptions: bool = True):
        self.folder_path = folder_path
        self.parser = SmaliParser()
        self.save_descriptions = save_descriptions
        self.descriptions_data = []
        
    async def load(self) -> List[Document]:
        """Load all Smali files from the folder and generate documents."""
        documents = []
        smali_files = self._find_smali_files(self.folder_path)
        
        logger.info(f"Found {len(smali_files)} Smali files to process")
        
        regular_classes = 0
        synthetic_classes = 0
        
        # Parse all files first (fast, synchronous operation)
        parsed_classes = []
        for i, file_path in enumerate(smali_files):
            logger.debug(f"Parsing file {i+1}/{len(smali_files)}: {os.path.basename(file_path)}")
            
            parsed_class = self.parser.parse_smali_file(file_path)
            if parsed_class:
                # Track class types
                if parsed_class.get('is_synthetic', False):
                    synthetic_classes += 1
                else:
                    regular_classes += 1
                
                parsed_classes.append(parsed_class)
        
        logger.info(f"Parsed {len(parsed_classes)} classes successfully")
        logger.info(f"  - Regular classes: {regular_classes}")
        logger.info(f"  - Synthetic classes: {synthetic_classes}")
        
        tasks = [
            self._process_parsed_class(i, parsed_class, len(parsed_classes))
            for i, parsed_class in enumerate(parsed_classes)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and create documents
        for i, (parsed_class, result) in enumerate(zip(parsed_classes, results)):
            if isinstance(result, Exception):
                logger.error(f"Error processing class {parsed_class['class_name']}: {result}")
                description = "Class description could not be generated due to an error."
            else:
                description = result
            
            # Save description data for inspection
            if self.save_descriptions:
                self.descriptions_data.append({
                    'class_name': parsed_class['class_name'],
                    'file_path': parsed_class['file_path'],
                    'is_synthetic': parsed_class.get('is_synthetic', False),
                    'method_count': len(parsed_class['methods']),
                    'permissions': parsed_class['permissions'],
                    'api_calls': parsed_class['api_calls'][:10],
                    'generated_description': description,
                    'cleaned_content': parsed_class['raw_content']
                })
            
            # Prepare metadata
            metadata = {
                'class_name': parsed_class['class_name'],
                'file_path': parsed_class['file_path'],
                'method_count': len(parsed_class['methods']),
                'raw_content': parsed_class['raw_content'],
                'methods': ', '.join([m['signature'] for m in parsed_class['methods']]),
                'permissions': ', '.join(parsed_class['permissions']),
                'api_calls': ', '.join(parsed_class['api_calls']),
                'is_synthetic': parsed_class.get('is_synthetic', False)
            }
            
            # Create LangChain Document
            doc = Document(
                page_content=description,
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Successfully processed {len(documents)} classes with descriptions")
        
        # Save descriptions to file
        if self.save_descriptions and self.descriptions_data:
            self._save_descriptions_to_file()
        
        return documents
    
    def _save_descriptions_to_file(self):
        """Save generated descriptions to a JSON file for inspection."""
        try:
            output_dir = CONFIG["output"]["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = logger.logger.handlers[0].baseFilename.split('_')[-1].replace('.log', '')
            filename = f"class_descriptions_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.descriptions_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Class descriptions saved to: {filepath}")
            logger.info(f"Total descriptions saved: {len(self.descriptions_data)}")
            
        except Exception as e:
            logger.error(f"Error saving descriptions: {e}")
    
    def _find_smali_files(self, folder_path: str) -> List[str]:
        """Recursively find all .smali files in the folder."""
        smali_files = []
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.smali'):
                    smali_files.append(os.path.join(root, file))
        
        return smali_files
    
    async def _process_parsed_class(self, index: int, parsed_class: Dict, total: int) -> str:
        """Process a single parsed class with rate limiting via semaphore."""
        logger.debug(f"Generating description {index+1}/{total}: {parsed_class['class_name']}")
        cleaned_content = parsed_class['raw_content']
        description = await self._generate_class_description_with_content(cleaned_content)
        return description
        
    async def _generate_class_description_with_content(self, cleaned_content: str) -> str:
        """Generate natural language description of the Smali class using the full cleaned content."""
        try:
            prompt = langfuse.get_prompt(CONFIG["langfuse"]["prompt_names"]["smali_class_description_prompt"]).compile(smali_class_content=cleaned_content)
            assert "{{" not in prompt, "Unresolved placeholders in class description prompt"
            result = await llm.generate_text(
                system_prompt=langfuse.get_prompt(CONFIG["langfuse"]["prompt_names"]["smali_class_description_system_prompt"]).compile(),
                prompt=prompt,
                temperature=CONFIG["openai"]["temperature"],
                max_tokens=CONFIG["openai"]["max_tokens"]
            )
            return result
        except Exception as e:
            logger.error(f"Error generating description with full content: {e}")
            fallback = "Class description could not be generated."
            return fallback 
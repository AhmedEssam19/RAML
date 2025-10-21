# RAML: Toward Retrieval-Augmented Localization of Malicious Payloads in Android Apps

A novel RAG pipeline for precisely localizing malicious payloads in Android apps at the method level, bridging high-level behavior descriptions with low-level Smali code semantics.

## Overview

RAML addresses the challenge of localizing exact malicious payloads in Android apps by leveraging LLMs to connect high-level malware family behavior knowledge with low-level Smali bytecode, enabling precise method-level localization with human-readable explanations.

## Key Contributions

1. **Novel RAG Framework**: First work to apply RAG paradigm for localizing malicious payloads in Android malware
2. **Method-Level Localization**: Precisely identifies specific methods with detailed explanations
3. **Evaluation Framework**: Includes LocApp dataset and real-world MalRadar analysis

## Technical Approach

1. **Class-Level Description Generation**: LLMs generate natural-language descriptions from Smali code
2. **Vector Embedding**: Embed descriptions into vector database for semantic search
3. **Semantic Retrieval**: Find relevant classes using similarity search
4. **LLM Re-ranking**: Re-rank candidates with LLM assistance
5. **Method-Level Analysis**: Pinpoint specific malicious methods and explain functions

## Installation


### Behavior Categories

| ID | Behavior | Description |
|----|----------|-------------|
| 1 | Privacy Stealing | Access/exfiltrate contacts, SMS, location, call logs |
| 2 | SMS/CALL Abuse | Send SMS, make calls, manipulate communication |
| 3 | Remote Control | C&C server communication, remote command execution |
| 4 | Bank/Financial Stealing | Banking trojans, credential theft, overlay attacks |
| 5 | Ransomware | File encryption, screen locking, ransom demands |
| 6 | Accessibility Abuse | Exploit accessibility services for automation |
| 7 | Privilege Escalation | Root exploits, system modifications |
| 8 | Stealthy Download | Covert app installation, silent downloads |
| 9 | Aggressive Advertising | Click fraud, ad manipulation |
| 10 | Cryptocurrency Mining | Background mining operations |
| 11 | Evasion Techniques | App hiding, anti-analysis measures |
| 12 | Premium Service Abuse | WAP billing fraud, hidden subscriptions |

## Architecture

### Components
- **SmaliLoader**: Loads and parses Smali files
- **SmaliParser**: Extracts class structure, methods, permissions, API calls
- **RetrievalEngine**: Two-stage RAG for behavior detection
- **ReportGenerator**: Creates analysis reports
- **Logger**: Analysis session logging

### Technical Details
- **Embedding Model**: OpenAI text-embedding-ada-002
- **LLM Model**: GPT-4 for analysis and reasoning
- **Vector Store**: ChromaDB for similarity search
- **Output**: JSON reports with explanations and confidence scores

## Evaluation

- **LocApp**: Custom Android app with common malicious behaviors for controlled evaluation
- **Real-World Analysis**: Assessed on MalRadar malware samples

## Output

- **Detailed JSON Report**: Complete analysis with all findings
- **Summary Report**: Human-readable summary of key findings
- **Analysis Metadata**: App name, behaviors, duration, confidence scores
- **Method Analysis**: Specific methods involved in each behavior

## Configuration

Configure via `config.py`:
- OpenAI settings (API key, model, temperature)
- Vector store settings (persistence, collection)
- Retrieval parameters (top-k, thresholds)
- Processing options (chunk sizes, limits)

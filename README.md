# LangChain Projects Cheat Sheet: Beginner to Advanced

## Table of Contents
1. [Setup & Installation](#setup--installation)
2. [Basic Projects](#basic-projects)
3. [Intermediate Projects](#intermediate-projects)
4. [Advanced Projects](#advanced-projects)
5. [Best Practices & Optimization](#best-practices--optimization)
6. [Open Source Resources](#open-source-resources)

## Setup & Installation

### Prerequisites
```bash
# Install Python 3.8+
pip install langchain
pip install transformers  # for Hugging Face models
pip install torch  # for PyTorch backend
pip install sentence-transformers  # for embeddings
pip install python-dotenv
pip install streamlit  # for web apps
pip install chromadb  # for vector storage
pip install accelerate  # for model optimization
pip install bitsandbytes  # for quantization
pip install ctransformers  # for GGML models
Environment Setup
# .env file
HUGGINGFACE_API_TOKEN=your_token_here  # Optional for private models
# No API keys needed for local models!


Basic Projects
1. Simple Chat Bot
Difficulty: Beginner Description: A basic chatbot using local Llama model via Hugging Face transformers.
import os
from dotenv import load_dotenv
from langchain.llms import HuggingFacePipeline
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load environment variables
load_dotenv()

class SimpleChatBot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """Initialize with a lightweight model for demo"""
        self.model_name = model_name
        self.llm = self._setup_llm()
    
    def _setup_llm(self):
        """Setup local LLM pipeline"""
        # For better performance, use Llama 2 7B or CodeLlama
        # model_name = "meta-llama/Llama-2-7b-chat-hf"  # Requires HF token
        
        # Using a smaller model for demo (no token required)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def chat(self, user_input):
        """Process user input and return AI response"""
        prompt = f"Human: {user_input}\nAssistant:"
        response = self.llm(prompt)
        return response.strip()

# Usage
if __name__ == "__main__":
    print("Loading model... (this may take a moment)")
    bot = SimpleChatBot()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = bot.chat(user_input)
        print(f"Bot: {response}")
1b. Optimized Llama Chat Bot (Using GGML)
Difficulty: Beginner Description: More efficient chat bot using quantized Llama models.
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class OptimizedLlamaBot:
    def __init__(self):
        """Initialize with quantized Llama model"""
        self.llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_file="llama-2-7b-chat.q4_0.bin",  # 4-bit quantized
            model_type="llama",
            config={
                'max_new_tokens': 256,
                'temperature': 0.7,
                'context_length': 2048,
                'gpu_layers': 50  # Use GPU if available
            },
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    
    def chat(self, user_input):
        """Chat with streaming response"""
        prompt = f"""[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

{user_input} [/INST]"""
        
        return self.llm(prompt)

# Usage
bot = OptimizedLlamaBot()
response = bot.chat("Explain quantum computing simply")
2. Text Summarizer
Difficulty: Beginner Description: Summarize long texts using open-source models.
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch

class TextSummarizer:
    def __init__(self):
        # Using BART for summarization (Facebook's model)
        self.summarizer_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Setup LLM for longer texts
        self.llm = HuggingFacePipeline(
            pipeline=self.summarizer_pipeline
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def summarize_short(self, text, max_length=150):
        """Summarize short text directly"""
        if len(text) < 50:
            return text
        
        summary = self.summarizer_pipeline(
            text,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )
        return summary[0]['summary_text']
    
    def summarize_long(self, text, chain_type="map_reduce"):
        """Summarize long text using LangChain"""
        if len(text) < 1000:
            return self.summarize_short(text)
        
        # Split text into chunks
        texts = self.text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        
        # Load summarization chain
        chain = load_summarize_chain(
            self.llm, 
            chain_type=chain_type
        )
        
        # Generate summary
        summary = chain.run(docs)
        return summary

# Usage
summarizer = TextSummarizer()

# For short texts
short_text = "Your text here..."
summary = summarizer.summarize_short(short_text)
print(f"Short Summary: {summary}")

# For long texts
long_text = """Your very long text here..."""
summary = summarizer.summarize_long(long_text)
print(f"Long Summary: {summary}")
3. Q&A System with Documents
Difficulty: Beginner-Intermediate Description: Create a Q&A system using open-source embeddings and Llama.
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentQA:
    def __init__(self, document_path):
        # Use open-source embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if GPU available
        )
        
        # Use quantized Llama model
        self.llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_file="llama-2-7b-chat.q4_0.bin",
            model_type="llama",
            config={
                'max_new_tokens': 256,
                'temperature': 0.1,  # Lower for factual answers
                'context_length': 2048
            }
        )
        
        self.vectorstore = None
        self.qa_chain = None
        self._load_document(document_path)
    
    def _load_document(self, document_path):
        """Load document and create vector store"""
        loader = TextLoader(document_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            texts, 
            self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Return top 3 relevant chunks
            ),
            return_source_documents=True
        )
    
    def ask_question(self, question):
        """Ask a question about the document"""
        if not self.qa_chain:
            return "No documents loaded"
        
        # Format prompt for Llama
        prompt = f"""[INST] <<SYS>>
You are a helpful assistant that answers questions based on the provided context.
If you don't know the answer, say "I don't know based on the provided context."
<</SYS>>

{question} [/INST]"""
        
        result = self.qa_chain({"query": prompt})
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

# Usage
qa_system = DocumentQA("path/to/your/document.txt")
result = qa_system.ask_question("What is the main topic of the document?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents found")
________________________________________
Intermediate Projects
4. Multi-Document Chat System
Difficulty: Intermediate Description: Chat with multiple documents using advanced retrieval techniques.
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
import os

class MultiDocumentChat:
    def __init__(self, documents_directory):
        # Setup embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Setup LLM
        self.llm = CTransformers(
            model="TheBloke/CodeLlama-7B-Instruct-GGML",
            model_file="codellama-7b-instruct.q4_0.bin",
            model_type="llama",
            config={
                'max_new_tokens': 512,
                'temperature': 0.3,
                'context_length': 4096
            }
        )
        
        # Setup memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.vectorstore = None
        self.qa_chain = None
        self._load_documents(documents_directory)
    
    def _load_documents(self, documents_directory):
        """Load multiple documents from directory"""
        # Support multiple file types
        loaders = [
            DirectoryLoader(documents_directory, glob="*.txt"),
            DirectoryLoader(documents_directory, glob="*.md"),
            DirectoryLoader(documents_directory, glob="*.py"),
        ]
        
        documents = []
        for loader in loaders:
            try:
                documents.extend(loader.load())
            except:
                continue
        
        if not documents:
            raise ValueError("No documents found in directory")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        texts = text_splitter.split_documents(documents)
        
        # Create vector store with metadata
        self.vectorstore = Chroma.from_documents(
            texts,
            self.embeddings,
            persist_directory="./multi_doc_db"
        )
        
        # Create conversational chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={"k": 5, "fetch_k": 10}
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
    
    def chat(self, question):
        """Chat with documents maintaining conversation context"""
        if not self.qa_chain:
            return "No documents loaded"
        
        # Format for Llama instruction following
        formatted_question = f"""[INST] <<SYS>>
You are an expert assistant that answers questions based on the provided documents.
Be accurate, concise, and cite relevant information from the documents.
If you're unsure, say so clearly.
<</SYS>>

{question} [/INST]"""
        
        result = self.qa_chain({"question": formatted_question})
        
        return {
            "answer": result["answer"],
            "sources": result["source_documents"],
            "chat_history": result.get("chat_history", [])
        }
    
    def get_relevant_docs(self, query, k=3):
        """Get relevant documents for a query"""
        if not self.vectorstore:
            return []
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs

# Usage
chat_system = MultiDocumentChat("./documents/")

# Chat with context
response1 = chat_system.chat("What are the main topics covered in these documents?")
print(f"Answer: {response1['answer']}")

# Follow-up question (uses conversation memory)
response2 = chat_system.chat("Can you elaborate on the first topic?")
print(f"Follow-up: {response2['answer']}")

# Get relevant documents
relevant = chat_system.get_relevant_docs("specific topic")
for doc in relevant:
    print(f"Source: {doc.metadata}")

5. Code Analysis and Generation Tool
Difficulty: Intermediate Description: Analyze and generate code using CodeLlama model.
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
import ast
import os

class CodeAnalyzer:
    def __init__(self):
        # Use CodeLlama for code-specific tasks
        self.llm = CTransformers(
            model="TheBloke/CodeLlama-7B-Instruct-GGML",
            model_file="codellama-7b-instruct.q4_0.bin",
            model_type="llama",
            config={
                'max_new_tokens': 1024,
                'temperature': 0.1,
                'context_length': 4096,
                'gpu_layers': 50
            }
        )
        
        # Setup code splitter
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def analyze_code(self, code_file_path):
        """Analyze Python code file"""
        with open(code_file_path, 'r') as f:
            code = f.read()
        
        prompt = PromptTemplate(
            input_variables=["code"],
            template="""[INST] <<SYS>>
You are an expert Python code analyzer. Analyze the following code and provide:
1. Code quality assessment
2. Potential bugs or issues
3. Optimization suggestions
4. Best practices recommendations
<</SYS>>

Code to analyze:
```python
{code}
Please provide a detailed analysis. [/INST]""" )
    chain = LLMChain(llm=self.llm, prompt=prompt)
    analysis = chain.run(code=code)
    return analysis

def generate_code(self, description, language="python"):
    """Generate code based on description"""
    prompt = PromptTemplate(
        input_variables=["description", "language"],
        template="""[INST] <<SYS>>
You are an expert programmer. Generate clean, efficient, and well-commented {language} code based on the following description. Include error handling and follow best practices. <</SYS>>
Generate {language} code for: {description}
Please provide complete, working code with comments. [/INST]""" )
    chain = LLMChain(llm=self.llm, prompt=prompt)
    code = chain.run(description=description, language=language)
    return code

def explain_code(self, code_snippet):
    """Explain code functionality"""
    prompt = PromptTemplate(
        input_variables=["code"],
        template="""[INST] <<SYS>>
You are a programming tutor. Explain the following code in simple terms. Break down what each part does and how it works together. <</SYS>>
Explain this code:
{code}
Please provide a clear, step-by-step explanation. [/INST]""" )
    chain = LLMChain(llm=self.llm, prompt=prompt)
    explanation = chain.run(code=code_snippet)
    return explanation

def refactor_code(self, code_snippet):
    """Refactor code for better quality"""
    prompt = PromptTemplate(
        input_variables=["code"],
        template="""[INST] <<SYS>>
You are a code refactoring expert. Refactor the following code to improve:
1.	Readability
2.	Performance
3.	Maintainability
4.	Follow PEP 8 standards Provide the refactored code with explanations of changes made. <</SYS>>
Original code:
{code}
Please provide refactored code with explanations. [/INST]""" )
    chain = LLMChain(llm=self.llm, prompt=prompt)
    refactored = chain.run(code=code_snippet)
    return refactored
Usage
analyzer = CodeAnalyzer()
Analyze existing code
analysis = analyzer.analyze_code("example.py") print(f"Code Analysis:\n{analysis}")
Generate new code
new_code = analyzer.generate_code("A function to calculate fibonacci numbers efficiently") print(f"Generated Code:\n{new_code}")
Explain code
explanation = analyzer.explain_code(""" def quicksort(arr): if len(arr) <= 1: return arr pivot = arr[len(arr) // 2] left = [x for x in arr if x < pivot] middle = [x for x in arr if x == pivot] right = [x for x in arr if x > pivot] return quicksort(left) + middle + quicksort(right) """) print(f"Code Explanation:\n{explanation}")

### 6. Web Scraping and Analysis System
**Difficulty: Intermediate-Advanced**
**Description**: Scrape websites and analyze content using open-source tools.

```python
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import CTransformers
from langchain.chains import AnalyzeDocumentChain
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import time
import random

class WebAnalyzer:
    def __init__(self):
        self.llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_file="llama-2-7b-chat.q4_0.bin",
            model_type="llama",
            config={
                'max_new_tokens': 512,
                'temperature': 0.3,
                'context_length': 4096
            }
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def scrape_and_analyze(self, url):
        """Scrape webpage and analyze content"""
        try:
            # Load webpage
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Summarize content
            summary_chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce"
            )
            summary = summary_chain.run(documents)
            
            # Analyze sentiment and key topics
            analysis_prompt = PromptTemplate(
                input_variables=["text"],
                template="""[INST] <<SYS>>
Analyze the following text and provide:
1. Main topics discussed
2. Sentiment analysis
3. Key insights
4. Important facts or statistics
<</SYS>>

Text to analyze:
{text}

Please provide a detailed analysis. [/INST]"""
            )
            
            analysis_chain = AnalyzeDocumentChain(
                combine_docs_chain=load_summarize_chain(
                    self.llm,
                    chain_type="stuff",
                    prompt=analysis_prompt
                ),
                text_splitter=self.text_splitter
            )
            
            analysis = analysis_chain.run(input_document=documents[0])
            
            return {
                "url": url,
                "summary": summary,
                "analysis": analysis,
                "content_length": len(documents[0].page_content)
            }
            
        except Exception as e:
            return {"error": f"Failed to process {url}: {str(e)}"}
    
    def compare_websites(self, urls):
        """Compare multiple websites"""
        results = []
        
        for url in urls:
            print(f"Processing {url}...")
            result = self.scrape_and_analyze(url)
            results.append(result)
            
            # Be respectful - add delay between requests
            time.sleep(random.uniform(1, 3))
        
        # Generate comparison
        comparison_prompt = PromptTemplate(
            input_variables=["results"],
            template="""[INST] <<SYS>>
Compare the following website analyses and provide:
1. Common themes across sites
2. Differences in perspective or content
3. Overall insights
4. Recommendations
<</SYS>>

Website analyses:
{results}

Please provide a comprehensive comparison. [/INST]"""
        )
        
        comparison_text = "\n\n".join([
            f"URL: {r.get('url', 'Unknown')}\nSummary: {r.get('summary', 'No summary')}\n"
            for r in results if 'error' not in r
        ])
        
        comparison = self.llm(comparison_prompt.format(results=comparison_text))
        
        return {
            "individual_results": results,
            "comparison": comparison
        }

LangChain Projects Cheat Sheet (Continuation)
Completing Web Scraping Project
# Usage
analyzer = WebAnalyzer()

# Analyze single website
result = analyzer.scrape_and_analyze("https://example.com")
print(f"Analysis: {result}")

# Compare multiple websites
urls = [
    "https://example1.com",
    "https://example2.com", 
    "https://example3.com"
]
comparison = analyzer.compare_websites(urls)
print(f"Comparison: {comparison['comparison']}")
________________________________________
Advanced Projects
7. Multi-Agent RAG System
Difficulty: Advanced Description: Create a multi-agent system with specialized AI agents for different tasks.
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from typing import List, Union
import re

class MultiAgentRAG:
    def __init__(self):
        # Initialize LLM
        self.llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_file="llama-2-7b-chat.q4_0.bin",
            model_type="llama",
            config={
                'max_new_tokens': 512,
                'temperature': 0.7,
                'context_length': 4096
            }
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(k=10)
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        # Initialize agents
        self.agents = self._setup_agents()
    
    def _setup_tools(self):
        """Setup tools for agents"""
        search = DuckDuckGoSearchRun()
        
        def calculator(expression):
            """Calculate mathematical expressions"""
            try:
                result = eval(expression)
                return f"Result: {result}"
            except:
                return "Error: Invalid expression"
        
        def code_analyzer(code):
            """Analyze code quality"""
            issues = []
            if "print(" in code and code.count("print(") > 3:
                issues.append("Too many print statements")
            if "TODO" in code:
                issues.append("Contains TODO items")
            if len(code.split('\n')) > 50:
                issues.append("Function too long")
            
            return f"Code analysis: {', '.join(issues) if issues else 'No issues found'}"
        
        return [
            Tool(
                name="Search",
                func=search.run,
                description="Search the internet for current information"
            ),
            Tool(
                name="Calculator",
                func=calculator,
                description="Calculate mathematical expressions"
            ),
            Tool(
                name="CodeAnalyzer",
                func=code_analyzer,
                description="Analyze code quality and identify issues"
            )
        ]
    
    def _setup_agents(self):
        """Setup specialized agents"""
        
        # Research Agent
        research_agent = initialize_agent(
            tools=[self.tools[0]],  # Search tool
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        
        # Math Agent
        math_agent = initialize_agent(
            tools=[self.tools[1]],  # Calculator tool
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        
        # Code Agent
        code_agent = initialize_agent(
            tools=[self.tools[2]],  # Code analyzer tool
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        
        return {
            "research": research_agent,
            "math": math_agent,
            "code": code_agent
        }
    
    def route_query(self, query):
        """Route query to appropriate agent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["calculate", "math", "compute", "solve"]):
            return "math"
        elif any(word in query_lower for word in ["code", "python", "function", "debug"]):
            return "code"
        elif any(word in query_lower for word in ["search", "find", "what is", "who is"]):
            return "research"
        else:
            return "research"  # Default to research
    
    def process_query(self, query):
        """Process query using appropriate agent"""
        agent_type = self.route_query(query)
        agent = self.agents[agent_type]
        
        try:
            result = agent.run(query)
            return {
                "agent_used": agent_type,
                "result": result,
                "status": "success"
            }
        except Exception as e:
            return {
                "agent_used": agent_type,
                "result": f"Error: {str(e)}",
                "status": "error"
            }

# Usage
multi_agent = MultiAgentRAG()

# Test different query types
queries = [
    "Calculate 25 * 4 + 10",
    "Search for latest developments in AI",
    "Analyze this code: def hello(): print('world')"
]

for query in queries:
    result = multi_agent.process_query(query)
    print(f"Query: {query}")
    print(f"Agent: {result['agent_used']}")
    print(f"Result: {result['result']}\n")
8. Advanced Document Processing Pipeline
Difficulty: Advanced Description: Process various document formats with advanced chunking and metadata extraction.
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, 
    UnstructuredMarkdownLoader, JSONLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    TokenTextSplitter, NLTKTextSplitter
)
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os
from pathlib import Path
import json
from datetime import datetime

class AdvancedDocumentProcessor:
    def __init__(self):
        self.llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_file="llama-2-7b-chat.q4_0.bin",
            model_type="llama",
            config={
                'max_new_tokens': 1024,
                'temperature': 0.3,
                'context_length': 4096
            }
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            ),
            'token': TokenTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            ),
            'nltk': NLTKTextSplitter(chunk_size=1000)
        }
        
        self.vectorstore = None
        self.processed_docs = []
    
    def load_documents(self, directory_path):
        """Load documents from directory with various formats"""
        documents = []
        directory = Path(directory_path)
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    loader = self._get_loader(file_path)
                    if loader:
                        docs = loader.load()
                        # Add metadata
                        for doc in docs:
                            doc.metadata.update({
                                'file_path': str(file_path),
                                'file_name': file_path.name,
                                'file_size': file_path.stat().st_size,
                                'modified_time': datetime.fromtimestamp(
                                    file_path.stat().st_mtime
                                ).isoformat()
                            })
                        documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _get_loader(self, file_path):
        """Get appropriate loader based on file extension"""
        suffix = file_path.suffix.lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.md': UnstructuredMarkdownLoader,
            '.json': JSONLoader
        }
        
        loader_class = loaders.get(suffix)
        if loader_class:
            if suffix == '.json':
                return loader_class(str(file_path), jq_schema='.')
            else:
                return loader_class(str(file_path))
        return None
    
    def smart_chunking(self, documents, strategy='adaptive'):
        """Apply intelligent chunking based on content type"""
        chunked_docs = []
        
        for doc in documents:
            # Choose splitter based on content analysis
            if strategy == 'adaptive':
                splitter = self._choose_splitter(doc)
            else:
                splitter = self.text_splitters[strategy]
            
            chunks = splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        return chunked_docs
    
    def _choose_splitter(self, doc):
        """Choose best splitter based on document content"""
        content = doc.page_content
        
        # For code files
        if any(keyword in content for keyword in ['def ', 'class ', 'import ', 'function']):
            return self.text_splitters['token']
        
        # For structured text
        elif content.count('\n\n') > content.count('\n') * 0.1:
            return self.text_splitters['recursive']
        
        # Default to NLTK for natural language
        else:
            return self.text_splitters['nltk']
    
    def extract_metadata(self, documents):
        """Extract enhanced metadata from documents"""
        metadata_prompt = PromptTemplate(
            input_variables=["text"],
            template="""[INST] <<SYS>>
Extract key metadata from the following document:
1. Main topic/subject
2. Document type (report, article, code, etc.)
3. Key entities (people, organizations, dates)
4. Summary in one sentence
5. Estimated reading time

Return as JSON format.
<</SYS>>

Document text:
{text}

Extract metadata: [/INST]"""
        )
        
        metadata_chain = LLMChain(llm=self.llm, prompt=metadata_prompt)
        
        for doc in documents:
            try:
                # Only process first 500 chars for efficiency
                sample_text = doc.page_content[:500]
                metadata_result = metadata_chain.run(text=sample_text)
                
                # Parse JSON if possible
                try:
                    parsed_metadata = json.loads(metadata_result)
                    doc.metadata.update(parsed_metadata)
                except:
                    doc.metadata['ai_analysis'] = metadata_result
                    
            except Exception as e:
                doc.metadata['metadata_error'] = str(e)
        
        return documents
    
    def create_vectorstore(self, documents, persist_directory="./advanced_vectorstore"):
        """Create vector store with processed documents"""
        # Apply smart chunking
        chunked_docs = self.smart_chunking(documents)
        
        # Extract metadata
        enhanced_docs = self.extract_metadata(chunked_docs)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=enhanced_docs,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        self.processed_docs = enhanced_docs
        return self.vectorstore
    
    def advanced_search(self, query, search_type="similarity", k=5):
        """Advanced search with multiple strategies"""
        if not self.vectorstore:
            return []
        
        search_kwargs = {"k": k}
        
        if search_type == "mmr":
            search_kwargs.update({"fetch_k": k*2})
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **search_kwargs
            )
        elif search_type == "similarity_score":
            docs = self.vectorstore.similarity_search_with_score(
                query, k=k
            )
        else:
            docs = self.vectorstore.similarity_search(query, k=k)
        
        return docs
    
    def generate_document_summary(self, file_path):
        """Generate comprehensive document summary"""
        # Load specific document
        loader = self._get_loader(Path(file_path))
        if not loader:
            return "Unsupported file format"
        
        documents = loader.load()
        
        # Map-reduce summarization for long documents
        map_template = """[INST] <<SYS>>
Summarize the following document section concisely:
<</SYS>>

{docs}

Summary: [/INST]"""
        
        map_prompt = PromptTemplate(template=map_template, input_variables=["docs"])
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)
        
        reduce_template = """[INST] <<SYS>>
Combine the following summaries into a comprehensive final summary:
<</SYS>>

{doc_summaries}

Final Summary: [/INST]"""
        
        reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["doc_summaries"])
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)
        
        # Combine documents chain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="doc_summaries"
        )
        
        # Reduce documents chain
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000
        )
        
        # Map reduce chain
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs"
        )
        
        return map_reduce_chain.run(documents)

# Usage
processor = AdvancedDocumentProcessor()

# Process documents from directory
documents = processor.load_documents("./documents")
print(f"Loaded {len(documents)} documents")

# Create vector store
vectorstore = processor.create_vectorstore(documents)

# Advanced search
results = processor.advanced_search("machine learning", search_type="mmr", k=3)
for doc in results:
    print(f"Source: {doc.metadata.get('file_name', 'Unknown')}")
    print(f"Content: {doc.page_content[:200]}...")
    print("---")

# Generate summary
summary = processor.generate_document_summary("./documents/report.pdf")
print(f"Document Summary: {summary}")
9. Real-time Chat with Memory and Context
Difficulty: Advanced Description: Create a sophisticated chat system with long-term memory and context awareness.
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMemory
from langchain.llms import CTransformers
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading
import time

class EnhancedMemory(BaseMemory):
    """Enhanced memory with persistence and context awareness"""
    
    def __init__(self, max_token_limit=2000):
        self.conversation_history = []
        self.user_profile = {}
        self.context_memory = {}
        self.max_token_limit = max_token_limit
        self.memory_file = "chat_memory.pkl"
        self.load_memory()
    
    @property
    def memory_variables(self) -> List[str]:
        return ["history", "user_profile", "context"]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Get recent conversation history
        recent_history = self._get_recent_history()
        
        # Get user profile summary
        profile_summary = self._get_profile_summary()
        
        # Get relevant context
        context = self._get_relevant_context(inputs.get("input", ""))
        
        return {
            "history": recent_history,
            "user_profile": profile_summary,
            "context": context
        }
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        # Save conversation turn
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "input": inputs.get("input", ""),
            "output": outputs.get("response", ""),
            "context": inputs.get("context", "")
        })
        
        # Update user profile
        self._update_user_profile(inputs.get("input", ""))
        
        # Save to disk
        self.save_memory()
    
    def _get_recent_history(self, max_turns=5):
        """Get recent conversation history"""
        recent = self.conversation_history[-max_turns:]
        formatted = []
        for turn in recent:
            formatted.append(f"Human: {turn['input']}")
            formatted.append(f"Assistant: {turn['output']}")
        return "\n".join(formatted)
    
    def _get_profile_summary(self):
        """Generate user profile summary"""
        if not self.user_profile:
            return "No user profile available."
        
        profile_parts = []
        for key, value in self.user_profile.items():
            profile_parts.append(f"{key}: {value}")
        
        return "User Profile: " + ", ".join(profile_parts)
    
    def _get_relevant_context(self, current_input):
        """Get relevant context based on current input"""
        # Simple keyword matching for context relevance
        relevant_context = []
        for key, value in self.context_memory.items():
            if any(word in current_input.lower() for word in key.lower().split()):
                relevant_context.append(f"{key}: {value}")
        
        return "Relevant Context: " + "; ".join(relevant_context) if relevant_context else ""
    
    def _update_user_profile(self, user_input):
        """Update user profile based on input"""
        # Extract user preferences and information
        if "i like" in user_input.lower():
            preference = user_input.lower().split("i like")[1].strip()
            self.user_profile["preferences"] = self.user_profile.get("preferences", [])
            self.user_profile["preferences"].append(preference)
        
        if "my name is" in user_input.lower():
            name = user_input.lower().split("my name is")[1].strip()
            self.user_profile["name"] = name
        
        # Add more profile extraction logic as needed
    
    def save_memory(self):
        """Save memory to disk"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump({
                    "conversation_history": self.conversation_history,
                    "user_profile": self.user_profile,
                    "context_memory": self.context_memory
                }, f)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def load_memory(self):
        """Load memory from disk"""
        try:
            with open(self.memory_file, 'rb') as f:
                data = pickle.load(f)
                self.conversation_history = data.get("conversation_history", [])
                self.user_profile = data.get("user_profile", {})
                self.context_memory = data.get("context_memory", {})
        except FileNotFoundError:
            # First time running, no memory file exists
            pass
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def clear(self) -> None:
        """Clear memory"""
        self.conversation_history = []
        self.user_profile = {}
        self.context_memory = {}

class RealTimeChatSystem:
    def __init__(self):
        self.llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_file="llama-2-7b-chat.q4_0.bin",
            model_type="llama",
            config={
                'max_new_tokens': 512,
                'temperature': 0.7,
                'context_length': 4096
            },
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        self.memory = EnhancedMemory()
        self.conversation_chain = self._setup_conversation_chain()
        
        # Background thread for memory cleanup
        self.cleanup_thread = threading.Thread(target=self._memory_cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _setup_conversation_chain(self):
        """Setup conversation chain with enhanced memory"""
        prompt = PromptTemplate(
            input_variables=["history", "user_profile", "context", "input"],
            template="""[INST] <<SYS>>
You are a helpful AI assistant with access to conversation history and user context.
Use the provided information to give personalized and contextually aware responses.

{user_profile}

Recent conversation:
{history}

{context}
<</SYS>>

Current message: {input}

Please provide a helpful response: [/INST]"""
        )
        
        return ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=True
        )
    
    def chat(self, user_input):
        """Process user input and return response"""
        try:
            response = self.conversation_chain.predict(input=user_input)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _memory_cleanup_loop(self):
        """Background loop to clean up old memory"""
        while True:
            time.sleep(3600)  # Run every hour
            self._cleanup_old_memory()
    
    def _cleanup_old_memory(self):
        """Remove old conversation history to prevent memory bloat"""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        self.memory.conversation_history = [
            turn for turn in self.memory.conversation_history
            if datetime.fromisoformat(turn["timestamp"]) > cutoff_date
        ]
        
        self.memory.save_memory()
    
    def get_conversation_stats(self):
        """Get conversation statistics"""
        total_turns = len(self.memory.conversation_history)
        if total_turns == 0:
            return "No conversation history."
        
        first_interaction = self.memory.conversation_history[0]["timestamp"]
        last_interaction = self.memory.conversation_history[-1]["timestamp"]
        
        return {
            "total_turns": total_turns,
            "first_interaction": first_interaction,
            "last_interaction": last_interaction,
            "user_profile": self.memory.user_profile
        }
    
    def export_conversation(self, filename="conversation_export.json"):
        """Export conversation history"""
        with open(filename, 'w') as f:
            json.dump(self.memory.conversation_history, f, indent=2)
        return f"Conversation exported to {filename}"

# Usage
chat_system = RealTimeChatSystem()

# Interactive chat loop
print("Advanced Chat System Started (type 'quit' to exit)")
print("Commands: /stats, /export, /clear")

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() == 'quit':
        break
    elif user_input == '/stats':
        stats = chat_system.get_conversation_stats()
        print(f"Stats: {stats}")
    elif user_input == '/export':
        result = chat_system.export_conversation()
        print(result)
    elif user_input == '/clear':
        chat_system.memory.clear()
        print("Memory cleared!")
    else:
        print("Assistant: ", end="")
        response = chat_system.chat(user_input)
        print()  # New line after streaming response
________________________________________
Best Practices & Optimization
Performance Optimization
1.	Model Quantization
# Use 4-bit quantization for better performance
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
2.	Efficient Chunking
# Use semantic chunking for better context
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0,
    tokens_per_chunk=256,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
3.	Caching Strategy
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Enable caching
set_llm_cache(InMemoryCache())
Memory Management
# Implement memory cleanup
def cleanup_memory():
    import gc
    import torch
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
Error Handling
# Robust error handling pattern
def safe_llm_call(llm, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
________________________________________
Open Source Resources
Recommended Models
1.	General Purpose:
o	Llama 2 7B/13B: meta-llama/Llama-2-7b-chat-hf
o	Mistral 7B: mistralai/Mistral-7B-Instruct-v0.1
2.	Code-Specific:
o	CodeLlama: codellama/CodeLlama-7b-Instruct-hf
o	StarCoder: bigcode/starcoder
3.	Embeddings:
o	All-MiniLM-L6-v2: sentence-transformers/all-MiniLM-L6-v2
o	BGE-Base: BAAI/bge-base-en-v1.5
Useful Libraries
# Essential libraries for LangChain projects
ESSENTIAL_LIBRARIES = {
    "langchain": "Core framework",
    "transformers": "Hugging Face models",
    "sentence-transformers": "Embeddings",
    "chromadb": "Vector database",
    "streamlit": "Web UI",
    "gradio": "Quick UI",
    "ctransformers": "Quantized models"
}
Community Resources
•	GitHub Repositories:
o	LangChain Examples: https://github.com/langchain-ai/langchain
o	Hugging Face Transformers: https://github.com/huggingface/transformers
•	Documentation:
o	LangChain Docs: https://python.langchain.com/
o	Hugging Face Docs: https://huggingface.co/docs
•	Model Hubs:
o	Hugging Face: https://huggingface.co/models
o	TheBloke (Quantized): https://huggingface.co/TheBloke
Development Tips
1.	Start Small: Begin with lightweight models for development
2.	Monitor Resources: Use system monitoring tools
3.	Version Control: Track model versions and configurations
4.	Testing: Implement unit tests for your chains
5.	Documentation: Document your prompt templates and chains



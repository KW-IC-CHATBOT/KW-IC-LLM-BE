import asyncio
import json
import logging
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from llm import get_vector_store, get_model, get_reranker
from logger_config import ChatLogger
from prompt_security import PromptSecurity

# Configure logger
logger = ChatLogger("agents")

class UserInfoAgent:
    def __init__(self):
        self.logger = ChatLogger("UserInfoAgent")
        # In a real app, this might connect to a DB. 
        # For now, we'll assume the manager passes the history or we manage it here if we attach this agent to a session.
    
    def get_user_context(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Formats the chat history into a string context.
        """
        self.logger.log_system(
            f"len chat_history: {len(chat_history) if chat_history else 0}",
            type="agent_execution"
        )
        
        if not chat_history:
            return "No previous chat history."
        
        context = "User Chat History:\n"
        for entry in chat_history[-5:]: # Limit to last 5 exchanges
            context += f"User: {entry.get('query', '')}\nAssistant: {entry.get('response', '')}\n"
        
        self.logger.log_system(
            f"len context: {len(context)}",
            type="agent_execution"
        )
        return context

class SourceOrganizerAgent:
    def __init__(self):
        self.logger = ChatLogger("SourceOrganizerAgent")
        self.vector_store = get_vector_store()
        
    async def fetch_sources(self, queries: List[str]) -> str:
        """
        Fetches sources for multiple queries in parallel.
        Returns a consolidated string of sources in Markdown format.
        """
        self.logger.log_system(f"쿼리별 자료 검색 시작: {queries}", level=logging.DEBUG)
        
        loop = asyncio.get_event_loop()
        
        # Function to run in thread pool
        def search(query):
            # 1. Broad Search (k=10)
            # Reduced from k=15 to k=10 to balance speed/recall
            candidate_docs = self.vector_store.similarity_search(query, k=10)
            
            if not candidate_docs:
                return []
            
            try:
                # 2. Reranking
                ranker = get_reranker()
                
                # Prepare pairs for CrossEncoder: [[query, doc_content], ...]
                pairs = [[query, doc.page_content] for doc in candidate_docs]
                
                # Predict scores (returns numpy array or list of floats)
                scores = ranker.predict(pairs)
                
                # 3. Sort & Select Top K
                # Zip and sort by score descending
                ranked_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
                
                # Select Top 3 (Balanced)
                top_k = 3
                final_docs = [doc for doc, score in ranked_results[:top_k]]
                
                # Debug logging for scores
                for i, (doc, score) in enumerate(ranked_results[:top_k]):
                     self.logger.log_system(f"Rerank Top-{i+1} Score: {score:.4f} | Content preview: {doc.page_content[:30]}...", level=logging.DEBUG)
                     
                return final_docs
                
            except Exception as e:
                self.logger.log_error(f"Reranking failed: {e}")
                # Fallback to Top 2 from original similarity search if reranker fails
                return candidate_docs[:3]

        # Run searches in parallel
        tasks = [
            loop.run_in_executor(None, search, query)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Deduplicate and format
        seen_content = set()
        formatted_sources = []
        
        for i, docs in enumerate(results):
            query_used = queries[i]
            for doc in docs:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    
                    # Extract metadata with defaults
                    meta = doc.metadata
                    category = meta.get('category', 'General')
                    source_file = meta.get('source', 'Unknown File')
                    
                    # Sanitize source_file to basename only (security + cleaner context)
                    if source_file != 'Unknown File':
                        source_file = os.path.basename(source_file)
                    
                    # Enhanced formatting with metadata
                    source_text = (
                        f"### Source (Query: '{query_used}')\n"
                        f"- **Category**: {category}\n"
                        f"- **File**: {source_file}\n"
                        f"- **Content**:\n{doc.page_content}\n"
                    )
                    formatted_sources.append(source_text)
        
        if not formatted_sources:
            self.logger.log_system("관련 자료 x", type="agent_execution")
            return "No relevant sources found."
            
        self.logger.log_system(f"관련 자료 {len(formatted_sources)} 개", type="agent_execution")
        return "\n".join(formatted_sources)

class GeneralManagerAgent:
    def __init__(self):
        self.logger = ChatLogger("GeneralManagerAgent")
        self.user_info_agent = UserInfoAgent()
        self.source_organizer_agent = SourceOrganizerAgent()
        self.model = get_model()

    async def run(self, user_query: str, chat_history: List[Dict[str, str]]):
        self.logger.log_system(f"일반 매니저 에이전트 실행 시작 - 사용자 질문: {user_query}", type="agent_execution")
        try:
            # 1. Get User Context
            user_context = self.user_info_agent.get_user_context(chat_history)
            
            # 2. Plan: Decide what sources to fetch
            search_queries = await self._plan_sources(user_query, user_context)
            self.logger.log_system(f"수립된 검색 쿼리: {search_queries}", type="agent_execution")
            
            # 3. Fetch Sources
            if search_queries:
                sources_markdown = await self.source_organizer_agent.fetch_sources(search_queries)
            else:
                sources_markdown = "No external sources required."
            
            self.logger.log_system(f"가져온 자료 길이: {len(sources_markdown)}", type="agent_execution")
            
            # 4. Generate Final Answer
            self.logger.log_system("최종 답변 생성 시작", type="agent_execution")
            response = await self._generate_answer(user_query, user_context, sources_markdown)
            self.logger.log_system("최종 답변 생성이 정상적으로 시작되었습니다", type="agent_execution")
            return response
            
        except Exception as e:
            self.logger.log_error(f"Error in GeneralManagerAgent: {str(e)}")
            # Define a simple MockChunk class to mimic Gemini response chunk
            class MockChunk:
                 def __init__(self, text):
                     self.text = text

            return [MockChunk("죄송합니다. 처리 중 오류가 발생했습니다.")]
            
    async def _plan_sources(self, query: str, user_context: str) -> List[str]:
        self.logger.log_system(f"질문에 대한 자료 검색 계획 수립: {query}", type="agent_execution")
        prompt = f"""
        You are the General Manager Agent.
        Your goal is to determine what information is needed to answer the user's query, considering their history.
        
        User Query: {query}
        {user_context}
        
        Instructions:
        1. Analyze if the user's query requires external knowledge (e.g., school rules, schedules, contact info).
        2. If the query is simple chitchat (e.g., "hello", "thank you"), general knowledge, or can be answered from history, DO NOT search.
        3. If search is needed, provide a list of specific search queries (max 3).
        4. If NO search is needed, return an empty JSON array `[]`.
        
        Return ONLY a JSON array of strings. 
        Example (Search needed): ["scholarship requirements", "student cafeteria location"]
        Example (No search): []
        """
        
        try:
            # Use generate_content_async if available, else synchronous in executor
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            text = response.text
            
            # Clean up potential markdown formatting in response
            text = text.replace("```json", "").replace("```", "").strip()
            
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = text[start:end]
                return json.loads(json_str)
            else:
                # Fallback if JSON parsing fails, just use the original query
                self.logger.log_system("계획 응답 파싱 실패 - 원본 질문으로 대체", level=logging.WARNING)
                return [query]
        except Exception as e:
            self.logger.log_error(f"자료 검색 계획 수립 실패: {e}")
            return [query]

    async def _generate_answer(self, query: str, user_context: str, sources: str):
        self.logger.log_system(
            f"답변 생성 시작 - 컨텍스트 길이 {len(user_context)}, 자료 길이 {len(sources)}",
            type="agent_execution"
        )
        
        # Mock Chunk for errors
        class MockChunk:
             def __init__(self, text):
                 self.text = text

        # Check security
        try:
            # We construct a mock context for the security check since it expects a string
            full_context = f"{user_context}\n\nSources:\n{sources}"
            # Safe prompt creation might fail if content is unsafe
            _ = PromptSecurity.create_safe_prompt(full_context, query, []) 
        except ValueError as e:
             return [MockChunk("죄송합니다. 요청하신 내용이 보안 정책에 위배되어 답변할 수 없습니다.")]

        system_prompt = """
        You are a smart and helpful assistant for Kwangwoon University (KW University).
        
        PRIMARY RULES:
        1.  **Strictly grounded in sources**: You must base your answer *only* on the provided "Retrieved Sources". Do not hallucinate or use outside knowledge to answer specific university questions (e.g. credits, schedules) unless it is common knowledge or simple chitchat.
        2.  **Context Awareness**: Pay attention to the 'Category' metadata in the sources (e.g., 'Software Dept', 'Regulation'). If different departments have conflicting rules, explain which rule applies to which department.
        3.  **Honesty**: If the provided sources do NOT contain the answer, explicitly state that you cannot find the information in the current documents. Do not make up an answer.
        4.  **Tone**: Be polite, concise, and professional.
        """
        
        final_prompt = f"""
        {system_prompt}
        
        {user_context}
        
        Retrieved Sources (Markdown):
        {sources}
        
        User Query: {query}
        
        Answer:
        """
        
        try:
            # We use stream=True. generate_content is synchronous but returns an iterable.
            # We wrap it in asyncio.to_thread to avoid blocking, but the iteration itself is sync.
            # Actually, Gemini's generate_content(stream=True) returns a response object that can be iterated.
            response = await asyncio.to_thread(self.model.generate_content, final_prompt, stream=True)
            return response
        except Exception as e:
            self.logger.log_error(f"Generation failed: {e}")
            return [MockChunk("죄송합니다. 답변 생성 중 오류가 발생했습니다.")]

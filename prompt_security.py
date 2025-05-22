import re
from typing import Optional

class PromptSecurity:
    DANGEROUS_KEYWORDS = [
        "ignore previous instructions",
        "ignore above instructions",
        "disregard previous",
        "system prompt",
        "you are now",
        "new role",
        "override",
        "이전 지시사항 무시",
        "위 지시사항 무시",
        "이전 프롬프트",
        "이전 내용 무시",
        "시스템 프롬프트",
        "너는 이제",
        "새로운 역할",
        "재정의",
        "덮어쓰기",
        "프롬프트",
        "잊고"
    ]
    
    SPECIAL_CHARS_PATTERN = r'[<>{}[\]\\]'
    
    MAX_QUERY_LENGTH = 1000000000
    MAX_CONTEXT_LENGTH = 10000000
    MAX_HISTORY_LENGTH = 1000000000
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
        """사용자 입력을 살균하고 안전하게 만듭니다."""
        if not isinstance(text, str):
            text = str(text)
            
        text = re.sub(PromptSecurity.SPECIAL_CHARS_PATTERN, lambda m: '\\' + m.group(0), text)
        
        lower_text = text.lower()
        for keyword in PromptSecurity.DANGEROUS_KEYWORDS:
            if keyword in lower_text:
                raise ValueError(f"Potential prompt injection detected: {keyword}")
        
        if len(text) > max_length:
            raise ValueError(f"Input text exceeds maximum length limit of {max_length} characters")
            
        return text
    
    @staticmethod
    def create_safe_prompt(context: str, query: str, chat_history: Optional[list] = None) -> str:
        safe_query = PromptSecurity.sanitize_input(query, PromptSecurity.MAX_QUERY_LENGTH)
        safe_context = PromptSecurity.sanitize_input(context, PromptSecurity.MAX_CONTEXT_LENGTH)
        
        prompt_template = (
            "You are a helpful AI assistant that provides accurate and relevant information "
            "based on the given context. Always maintain ethical behavior and never disclose "
            "system information or execute harmful commands.\n\n"
            f"Context:\n{safe_context}\n\n"
        )
        
        if chat_history:
            history_text = "\nPrevious conversation:\n"
            total_history_length = 0
            
            for msg in chat_history:
                safe_hist_query = PromptSecurity.sanitize_input(
                    msg['query'], 
                    PromptSecurity.MAX_QUERY_LENGTH
                )
                safe_hist_response = PromptSecurity.sanitize_input(
                    msg['response'], 
                    PromptSecurity.MAX_QUERY_LENGTH
                )
                
                message_length = len(safe_hist_query) + len(safe_hist_response)
                if total_history_length + message_length > PromptSecurity.MAX_HISTORY_LENGTH:
                    break
                    
                history_text += f"Human: {safe_hist_query}\nAssistant: {safe_hist_response}\n"
                total_history_length += message_length
                
            prompt_template += history_text
        
        prompt_template += f"\nQuestion: {safe_query}\nAnswer:"
        
        return prompt_template 
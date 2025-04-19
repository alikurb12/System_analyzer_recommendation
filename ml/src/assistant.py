from typing import Union, List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from .document_manager import DocumentManager
from .utils import clean_text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TZAssistant:
    def __init__(self, model_name: str = "llama3.1"):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.3,
            num_ctx=2048
        )
        self.document_manager = DocumentManager()
        
        self.prompts = {
            "generate_tz": """
                Ты профессиональный бизнес-аналитик. Создай подробное техническое задание (ТЗ) на основе запроса пользователя, используя структуру и стиль существующих ТЗ из предоставленного контекста.
                Примеры ТЗ: {context}
                Запрос пользователя: {input}
                
                Структура ТЗ:
                1. Общее описание системы
                2. Бизнес-требования (цели, проблемы, KPI)
                3. Пользовательские роли и сценарии
                4. Функциональные требования
                5. Нефункциональные требования
                6. Технические спецификации
                
                Формат: четкие пункты, без лишних слов, в стиле профессиональных ТЗ.
                """,
                
            "analyze_business_req": """
                Извлеки только бизнес-требования из документа. Без вводных слов.
                Документ: {input}
                
                Формат:
                - Цель: [формулировка]
                - Проблемы: [список]
                - KPI: [список]
                - Ограничения: [список]
                """,
                
            "analyze_user_req": """
                Ты бизнес-аналитик. Извлеки пользовательские требования из ТЗ, бизнес-требований и пользовательских сценариев.
                Документ: {input}
                
                Верни строго в следующем формате, без лишних слов:
                1. Введение
                - Цель: [зачем создается ПО]
                - Область применения: [кто и в каких условиях будет использовать систему]
                - Целевая аудитория: [основные пользователи, их роли]
                
                2. Функциональные требования
                - [Категория 1, например, Регистрация и авторизация]:
                  - [конкретная возможность]
                  - [конкретная возможность]
                - [Категория 2, например, Управление профилем]:
                  - [конкретная возможность]
                  - [конкретная возможность]
                
                3. Нефункциональные требования
                - Производительность:
                  - [требование, например, время отклика]
                - Безопасность:
                  - [требование, например, хеширование паролей]
                - Удобство использования (UX):
                  - [требование, например, гайдлайны дизайна]
                - Совместимость:
                  - [требование, например, поддерживаемые браузеры]
                
                4. Ограничения и зависимости
                - [ограничение, например, интеграция с внешней системой]
                - [ограничение, например, поддерживаемые языки]
                """,
                
            "extract_user_stories": """
                Ты бизнес-аналитик. Извлеки пользовательские истории из ТЗ, бизнес-требований и пользовательских требований.
                Документ: {input}
                
                Верни строго в следующем формате, без лишних слов:
                Пользовательские истории:
                - Роль: [название роли]
                  Действие: [что хочет сделать]
                  Цель: [какая выгода/цель]
                - Роль: [название роли]
                  Действие: [что хочет сделать]
                  Цель: [какая выгода/цель]
                """
        }

    def add_document(self, source: Union[str, bytes, Document]):
        documents = self.document_manager.load_document(source)
        self.document_manager.process_documents(documents)
        return documents

    def generate_tz(self, user_request: str) -> str:
        context_docs = []
        if self.document_manager.retriever:
            context_docs = self.document_manager.retriever.invoke(user_request)
        
        context = " ".join([doc.page_content for doc in context_docs])
        
        prompt = ChatPromptTemplate.from_template(self.prompts["generate_tz"])
        chain = prompt | self.llm
        
        raw_response = chain.invoke({
            "context": context,
            "input": user_request
        }).strip()

        return clean_text(raw_response)

    def analyze_business_req(self, document: Union[str, bytes, Document]) -> str:
        if isinstance(document, str):
            doc = Document(page_content=document)
        else:
            docs = self.add_document(document)
            doc = docs[0]
        
        prompt = ChatPromptTemplate.from_template(self.prompts["analyze_business_req"])
        chain = prompt | self.llm
        raw_response = chain.invoke({"input": doc.page_content}).strip()
        # logger.info(f"Business requirements raw response: {raw_response}")
        return clean_text(raw_response)

    def analyze_user_req(self, document: Union[str, bytes, Document]) -> str:
        if isinstance(document, str):
            doc = Document(page_content=document)
        else:
            docs = self.add_document(document)
            doc = docs[0]
        
        prompt = ChatPromptTemplate.from_template(self.prompts["analyze_user_req"])
        chain = prompt | self.llm
        raw_response = chain.invoke({"input": doc.page_content}).strip()
        # logger.info(f"User requirements raw response: {raw_response}")
        return clean_text(raw_response)

    def extract_user_stories(self, document: Union[str, bytes, Document]) -> List[Dict]:
        if isinstance(document, str):
            doc = Document(page_content=document)
        else:
            docs = self.add_document(document)
            doc = docs[0]
        
        prompt = ChatPromptTemplate.from_template(self.prompts["extract_user_stories"])
        chain = prompt | self.llm
        raw_response = chain.invoke({"input": doc.page_content}).strip()
        logger.info(f"User stories raw response: {raw_response}")
        lines = raw_response.split('\n')
        stories = []
        current_story = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("Пользовательские истории:"):
                continue
            elif line.startswith("- Роль:"):
                current_story = {"role": line.replace("- Роль: ", "")}
                stories.append(current_story)
            elif current_story and line.startswith("Действие:"):
                current_story["action"] = line.replace("Действие: ", "")
            elif current_story and line.startswith("Цель:"):
                current_story["goal"] = line.replace("Цель: ", "")
        
        return stories

    def generate_user_stories(self, document: Union[str, bytes, Document]) -> str:
        stories = self.extract_user_stories(document)
        result = "Пользовательские истории:\n"
        for story in stories:
            result += f"Как {story['role']}, я хочу {story['action']}, чтобы {story['goal']}.\n"
        return result.strip()
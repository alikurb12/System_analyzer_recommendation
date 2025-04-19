import os
from src.assistant import TZAssistant

def load_all_documents(assistant, folder_path="tz_documents"):
    supported_extensions = ('.pdf', '.docx', '.txt')
    for filename in os.listdir(folder_path):
        if filename.endswith(supported_extensions):
            file_path = os.path.join(folder_path, filename)
            assistant.add_document(file_path)
            print(f"Загружен документ: {filename}")

if __name__ == "__main__":
    assistant = TZAssistant()
    
    print("=== ЗАГРУЗКА ВСЕХ ТЗ ИЗ tz_documents ===")
    load_all_documents(assistant)
    
    print("\n=== ГЕНЕРАЦИЯ ТЗ ===")
    new_tz = assistant.generate_tz(
        "Разработать мобильное приложение для доставки еды с функцией отслеживания курьера"
    )
    print(new_tz)
    
    print("\n=== АНАЛИЗ СГЕНЕРИРОВАННОГО ТЗ ===")
    business_req = assistant.analyze_business_req(new_tz)
    user_req = assistant.analyze_user_req(new_tz)
    print("Бизнес-требования:", business_req)
    print("Пользовательские требования:", user_req)
    
    print("\n=== ГЕНЕРАЦИЯ ПОЛЬЗОВАТЕЛЬСКИХ ИСТОРИЙ ===")
    user_stories = assistant.generate_user_stories(new_tz)
    print(user_stories)
    with open("user_requirements.txt", "w", encoding="utf-8") as f:
        f.write(user_req)
    with open("user_stories.txt", "w", encoding="utf-8") as f:
        f.write(user_stories)
    
    print("\nРезультаты сохранены в файлы: user_requirements.txt, user_stories.txt")
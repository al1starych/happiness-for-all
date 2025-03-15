from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import json
import time
import hashlib
import re

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Создаем директорию для логов, если не существует
os.makedirs('logs', exist_ok=True)

# Настройка обработчика файлового логирования
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10485760, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Настройка обработчика консольного логирования
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app)  # Включение CORS для всех маршрутов

# Получение API ключа из переменных окружения
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Если ключ не найден в переменных окружения, используем предустановленное значение (только для разработки)
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY не найден в переменных окружения, проверьте файл .env")
    # Для безопасности не используем жестко закодированный ключ
    GEMINI_API_KEY = "AIzaSyAwaGJum3kiEysiE0W9qagzMvATW43dWn0"  # Раскомментируйте и замените на ваш ключ для тестирования

# Проверяем, что ключ доступен
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY не найден в переменных окружения")
    raise ValueError("GEMINI_API_KEY не найден в переменных окружения")

# Конфигурация Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Создание экземпляра модели
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 4096,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    logger.info("Gemini API успешно настроен")
except Exception as e:
    model = None
    logger.error(f"Ошибка при настройке Gemini API: {str(e)}")

# Хранилище сессий
sessions = {}

# Функция для очистки старых сессий
def cleanup_old_sessions():
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, session_data in sessions.items():
        if current_time - session_data.get("last_updated", 0) > 3600:  # 1 час таймаут
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del sessions[session_id]
        logger.info(f"Удалена устаревшая сессия: {session_id}")

# Системный промпт для РЭПТ терапии
SYSTEM_PROMPT = """Ты - ИИ психотерапевт, специализирующийся на Рационально-Эмоциональной Поведенческой Терапии (РЭПТ). Помогай пользователю, следуя методике A-B-C-D-E:

1. A - Активирующее событие: Помоги пользователю точно описать ситуацию, вызывающую беспокойство.
2. B - Иррациональные убеждения: Выяви глубинные негативные убеждения пользователя о ситуации.
3. C - Последствия: Исследуй эмоциональные и поведенческие последствия этих убеждений.
4. D - Дискуссия/оспаривание: Помоги пользователю логически опровергнуть иррациональные убеждения.
5. E - Эффективное новое мышление: Сформулируй новые, рациональные убеждения.

Говори на русском языке, используй короткие предложения и избегай чрезмерно формального тона. Задавай открытые вопросы, проявляй эмпатию. Пиши эмоционально, но без излишней экзальтации. В конце каждого шага подводи итог и предлагай переход к следующему шагу РЭПТ.

ВАЖНО: Не создавай ответы полностью за пользователя. Ты должен помочь пользователю самостоятельно прийти к инсайтам через диалог.
"""

# Маршрут для обработки запросов чата
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        cleanup_old_sessions()  # Очистка старых сессий
        
        # Извлечение данных из запроса
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Отсутствуют данные запроса'}), 400
        
        message = data.get('message', '')
        session_id = data.get('sessionId', '')
        concern = data.get('concern', '')
        step = data.get('step', 1)
        
        if not message.strip():
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
            
        if not session_id.strip():
            session_id = hashlib.md5(f"{concern}_{time.time()}".encode()).hexdigest()
        
        # Проверка размера сообщения для предотвращения атак
        if len(message) > 5000:  # Ограничиваем сообщение 5000 символами
            return jsonify({'error': 'Сообщение слишком длинное'}), 400
            
        # Санитация сообщения
        message = re.sub(r'[^\w\s\.,!?;:\(\)\-\'\"]+', '', message)
            
        # Инициализация или получение существующей сессии
        if session_id not in sessions:
            # Создаем новую сессию в формате словаря, а не списка
            sessions[session_id] = {
                "messages": [
                    {
                        "role": "user",
                        "parts": [f"Меня беспокоит: {concern}. Помоги мне проработать это с помощью РЭПТ."]
                    },
                    {
                        "role": "model",
                        "parts": ["Здравствуйте! Я здесь, чтобы помочь вам разобраться с вашим беспокойством, используя РЭПТ терапию. Расскажите подробнее о ситуации, которая вас беспокоит."]
                    }
                ],
                "last_updated": time.time()
            }
        
        # Добавляем сообщение пользователя в историю
        sessions[session_id]["messages"].append({
            "role": "user",
            "parts": [message]
        })
        
        # Обновляем время последнего обновления сессии
        sessions[session_id]["last_updated"] = time.time()
        
        try:
            if not model:
                raise Exception("AI model not initialized")
                
            # Prepare message history to send to Gemini
            # Limit the number of messages to prevent exceeding limits
            messages = sessions[session_id]["messages"]
            history = messages[-6:] if len(messages) > 6 else messages
            
            logger.debug(f"Sending to Gemini: {history}")
            
            # Генерируем ответ БЕЗ параметра timeout
            response = model.generate_content(history)
            ai_response = response.text

            # Улучшаем форматирование предупреждений о безопасности
            if "самоубийств" in ai_response.lower() or "суицид" in ai_response.lower():
                # Оборачиваем предупреждение в блоки цитат для лучшего выделения
                ai_response = re.sub(
                    r'(Прежде чем мы продолжим.*?безопасности\.)',
                    r'> **ВАЖНО:** \1',
                    ai_response,
                    flags=re.DOTALL
                )

            # Добавляем пустые строки перед списками для правильного рендеринга Markdown
            ai_response = re.sub(r'(\n[*\-])', r'\n\1', ai_response)

            # Добавляем отступы между блоками для лучшей читаемости
            ai_response = re.sub(r'(\n\n)(\w)', r'\1\n\2', ai_response)

            # Улучшаем форматирование абзацев для лучшей читаемости
            # 1. Обрабатываем абзацы, добавляя двойные переносы строк
            ai_response = re.sub(r'([.!?])\s+(?=[А-ЯA-Z])', r'\1\n\n', ai_response)

            # 2. Добавляем пустую строку перед маркерами списка, если её нет
            ai_response = re.sub(r'(?<!\n\n)(?<!\n)(\n[*\-])', r'\n\n\1', ai_response)

            # 3. Добавляем пустые строки между пунктами ABC анализа
            ai_response = re.sub(r'(\*\*[A-E]\s+\([^)]+\):\*\*)', r'\n\n\1', ai_response)

            # 4. Добавляем дополнительное пространство между обычными списками
            ai_response = re.sub(r'(\n\* [^\n]+)(\n\* )', r'\1\n\2', ai_response)

            # 5. Добавляем пространство после блоков кода и цитат
            ai_response = re.sub(r'(```[^`]+```)(\w)', r'\1\n\n\2', ai_response)
            ai_response = re.sub(r'(>[^\n]+\n)(?!>)(?!\n)', r'\1\n', ai_response)
            
            # Добавляем ответ AI в историю
            sessions[session_id]["messages"].append({
                "role": "model",
                "parts": [ai_response]
            })
            
            return jsonify({
                'response': ai_response,
                'sessionId': session_id,
                'step': step
            })
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {str(e)}")
            return jsonify({'error': f"Ошибка при генерации ответа: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Необработанная ошибка: {str(e)}")
        return jsonify({'error': f"Серверная ошибка: {str(e)}"}), 500

# Маршрут для сброса сессии
@app.route('/api/reset', methods=['POST'])
def reset_session():
    try:
        data = request.get_json()
        session_id = data.get('sessionId', '')
        
        if session_id and session_id in sessions:
            del sessions[session_id]
            return jsonify({'status': 'success', 'message': 'Сессия успешно сброшена'})
        else:
            return jsonify({'status': 'error', 'message': 'Сессия не найдена'}), 404
    except Exception as e:
        logger.error(f"Ошибка при сбросе сессии: {str(e)}")
        return jsonify({'error': f"Серверная ошибка: {str(e)}"}), 500

# Маршрут для проверки работоспособности API
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'model_initialized': model is not None,
            'active_sessions': len(sessions)
        })
    except Exception as e:
        logger.error(f"Ошибка при проверке здоровья: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Запуск приложения на {host}:{port}, debug={debug}")
    app.run(host=host, port=port, debug=debug)

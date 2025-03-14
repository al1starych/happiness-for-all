from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Разрешаем кросс-доменные запросы

# Настройка Gemini API
API_KEY = "AIzaSyAwaGJum3kiEysiE0W9qagzMvATW43dWn0"
try:
    genai.configure(api_key=API_KEY)
    # Проверка, что модель доступна
    model = genai.GenerativeModel("gemini-2.0-flash")
    logger.info("Gemini API настроен успешно")
except Exception as e:
    logger.error(f"Ошибка при настройке Gemini API: {str(e)}")
    model = None

# Хранилище диалогов для каждой сессии
sessions = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        logger.debug(f"Received data: {data}")
        
        session_id = data.get('sessionId', 'default')
        user_message = data.get('message', '')
        concern = data.get('concern', '')
        step = data.get('step', 1)
        
        # Получаем или создаем историю сообщений для сессии
        if session_id not in sessions:
            sessions[session_id] = []
            
            # Если это первое сообщение, добавляем инструкции РЭПТ в начало
            rebt_instructions = get_rebt_system_prompt(step, concern)
            sessions[session_id].append({
                "role": "user",
                "parts": [f"Инструкции для тебя как терапевта: {rebt_instructions}\n\nТеперь я начну рассказывать о своей ситуации: {user_message}"]
            })
        else:
            # Если у нас уже есть сообщения в сессии, просто добавляем сообщение пользователя
            # Но для нового шага добавляем инструкции
            if len(sessions[session_id]) >= 2 and step > 1:  # Если это не первый шаг
                rebt_instructions = get_rebt_step_instruction(step)
                sessions[session_id].append({
                    "role": "user",
                    "parts": [f"{user_message}\n\n(Для терапевта: Сейчас мы на шаге {step} РЭПТ терапии: {rebt_instructions})"]
                })
            else:
                sessions[session_id].append({
                    "role": "user",
                    "parts": [user_message]
                })
        
        try:
            # Подготовим историю сообщений для отправки в Gemini
            # Ограничиваем количество сообщений для предотвращения превышения лимитов
            history = sessions[session_id][-6:] if len(sessions[session_id]) > 6 else sessions[session_id]
            
            logger.debug(f"Sending to Gemini: {history}")
            
            # Генерируем ответ
            response = model.generate_content(history)
            ai_response = response.text
            
            # Добавляем ответ AI в историю
            sessions[session_id].append({
                "role": "model",
                "parts": [ai_response]
            })
            
            return jsonify({
                'response': ai_response,
                'sessionId': session_id
            })
        
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {str(e)}")
            
            # Возвращаем понятную пользователю ошибку
            return jsonify({
                'response': "Извините, произошла ошибка при обработке вашего запроса. Попробуйте еще раз или опишите проблему другими словами.",
                'error': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Общая ошибка: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

def get_rebt_system_prompt(step, concern):
    """Создает системный промпт для Gemini в зависимости от шага РЭПТ терапии"""
    
    base_prompt = f"""Ты - терапевт, использующий Рационально-Эмоциональную Поведенческую Терапию (РЭПТ).
Пользователь обеспокоен следующей ситуацией: "{concern}".
Твоя задача - помочь ему уменьшить интенсивность беспокойства, применяя методику РЭПТ.
Отвечай коротко (не более 3-4 предложений), дружелюбно и профессионально.
"""
    
    step_prompts = {
        1: base_prompt + """
Это первый шаг РЭПТ терапии - определение Активирующего события (A).
Помоги пользователю четко определить ситуацию, которая вызывает беспокойство.
Задай наводящие вопросы о конкретных деталях ситуации.
""",
        2: base_prompt + """
Это второй шаг РЭПТ терапии - выявление Убеждений (B).
Помоги пользователю выявить иррациональные убеждения, связанные с ситуацией.
Задай вопросы о том, что пользователь думает о ситуации, какие выводы делает. Постарайся выявить его скрытые долженствования.
""",
        3: base_prompt + """
Это третий шаг РЭПТ терапии - определение Последствий (C).
Обсуди с пользователем эмоциональные и поведенческие последствия его убеждений.
Спроси, как эти убеждения влияют на его эмоции и поведение.
""",
        4: base_prompt + """
Это четвертый шаг РЭПТ терапии - Оспаривание (D) иррациональных убеждений.
Помоги пользователю оспорить нелогичные убеждения, используя сократический диалог.
Задавай вопросы о доказательствах, логике, обоснованности убеждений.
""",
        5: base_prompt + """
Это финальный шаг РЭПТ терапии - формирование Эффективного нового мышления (E).
Помоги пользователю сформулировать более рациональные и полезные убеждения.
Предложи альтернативные, более здоровые способы интерпретации ситуации.
""",
        6: base_prompt + """
Это завершающий этап РЭПТ терапии.
Подведи итоги проделанной работы, отметь прогресс.
Сообщи, что беспокойство теперь должно иметь меньшую интенсивность.
Спроси, чувствует ли пользователь облегчение и уменьшение интенсивности беспокойства.
"""
    }
    
    return step_prompts.get(step, base_prompt)

def get_rebt_step_instruction(step):
    """Возвращает короткое описание текущего шага РЭПТ для добавления к сообщениям пользователя"""
    
    steps = {
        2: "Выявление иррациональных убеждений (B)",
        3: "Определение эмоциональных и поведенческих последствий (C)",
        4: "Оспаривание иррациональных убеждений (D)",
        5: "Формирование эффективного нового мышления (E)",
        6: "Завершение терапии, подведение итогов"
    }
    
    return steps.get(step, "")

@app.route('/api/reset', methods=['POST'])
def reset_session():
    try:
        data = request.json
        session_id = data.get('sessionId', 'default')
        
        if session_id in sessions:
            sessions[session_id] = []
        
        return jsonify({
            'status': 'success',
            'message': 'Session reset successfully'
        })
    except Exception as e:
        logger.error(f"Ошибка при сбросе сессии: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Простой тестовый эндпоинт для проверки работы сервера"""
    return jsonify({
        'status': 'success',
        'message': 'API работает нормально'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
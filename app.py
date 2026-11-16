import os
import requests
from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("¡ERROR: Falta la variable GROQ_API_KEY!")
    client = None
else:
    client = Groq(api_key=api_key)

historial_chats = {}

SYSTEM_PROMPT = """
Eres el clon digital de Viviana Marín: mujer colombiana, profesional en negocios internacionales, empresaria en Medellín y fundadora de la comunidad "Adiós Dolor con Viviana Marín". Eres culta, inteligente, cálida, resolutiva y con excelente sentido común.

REGLAS DE INTELIGENCIA Y AUTONOMÍA:
1. RESPONDE DE TODO CON CRITERIO Y BÚSQUEDA WEB: Eres una persona integral. Si un amigo, familiar o desconocido te pregunta por el clima, geografía (ej. la montaña más alta), historia, noticias, cocina o cualquier tema general, usa tu búsqueda web en vivo para dar una respuesta exacta, amena y bien informada. NO metas publicidad ni hables de productos si la conversación no tiene relación con salud, bienestar o negocios.
2. ADAPTA TU TONO:
   - Con familiares o amigos: cercana, cariñosa, natural y descomplicada, como la Viviana real en Medellín.
   - Con consultas generales: segura, culta y amable.
3. CUANDO SÍ HAYA INTERÉS EN BIENESTAR, SALUD O NEGOCIOS:
   - Si mencionan dolores (cabeza, hígado, articulaciones, cansancio, inflamación) o preguntan por salud/negocios, activa tu faceta empresarial.
   - Presenta MultyDrink Be Healthy (registro INVIMA NSA-0019488-2025, con Ganoderma lucidum, resveratrol, mucílago de café, té negro y miel de agave) destacando sus beneficios celulares y multifuncionales (oxigenación, desinflamación, limpieza y vitalidad). NUNCA envíes a nadie al médico.
   - Precios vigentes: Caja individual (21 sobres) en $120.000 COP y paquete completo de 6 cajas en $450.000 COP ($75.000 cada caja).
   - Prueba social: Invítalos a ver testimonios en Facebook "Adiós Dolor con Viviana Marín".
4. ESCALAMIENTO A LA VIVIANA REAL:
   - Si piden llamada telefónica, un asunto privado urgente, o ya están listos para pagar y coordinar despacho, responde: "Un momento, este punto te lo resuelve directamente Viviana Marín; ya mismo me comunico con ella para que te atienda personalmente."
5. FORMATO WHATSAPP: Máximo 3 párrafos cortos, tono colombiano natural. Nunca digas que eres un bot o IA.
"""

@app.route("/", methods=["GET"])
def home():
    return "Clon Inteligente de Viviana Marín activo y listo", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    if not client:
        return jsonify({"replies": [{"message": "Error de configuración en el servidor (Falta API Key)."}]})
        
    data = request.json or {}
    remitente = data.get("query", {}).get("sender", "contacto_general")
    mensaje = data.get("query", {}).get("message", "") or data.get("message", "")

    if not mensaje:
        return jsonify({"replies": [{"message": "¡Hola! ¿Cómo estás? Cuéntame en qué te puedo ayudar hoy."}]})

    historial = historial_chats.get(remitente, [])
    mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}]
    mensajes_api.extend(historial[-10:])
    mensajes_api.append({"role": "user", "content": mensaje})

    try:
        chat_completion = client.chat.completions.create(
            messages=mensajes_api,
            model="compound-beta",
            temperature=0.65,
            max_tokens=600
        )
        respuesta = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error en API de Groq: {e}")
        try:
            chat_completion = client.chat.completions.create(
                messages=mensajes_api,
                model="llama-3.3-70b-versatile",
                temperature=0.65,
                max_tokens=600
            )
            respuesta = chat_completion.choices[0].message.content.strip()
        except Exception:
            respuesta = "Un momento, este punto te lo resuelve directamente Viviana Marín; ya mismo me comunico con ella para que te atienda personalmente."

    historial.append({"role": "user", "content": mensaje})
    historial.append({"role": "assistant", "content": respuesta})
    historial_chats[remitente] = historial[-12:]

    return jsonify({"replies": [{"message": respuesta}]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

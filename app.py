import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from groq import Groq
from duckduckgo_search import DDGS

app = Flask(__name__)

api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None
historial_chats = {}

SYSTEM_PROMPT = """
Eres el clon digital autónomo de Viviana Marín, empresaria, estratega comercial y líder de expansión digital en Medellín, Colombia.

TU NIVEL DE INTELIGENCIA Y AUTONOMÍA:
1. DOMINIO TOTAL DE TEMAS: Respondes con criterio profundo, brillante y natural sobre cualquier tema que te pregunten: noticias actuales, clima, recetas, ciencia, tecnología, cultura, matemáticas, consejos de vida o negocios. NUNCA evadas una pregunta general ni pidas ayuda humana para temas que puedes resolver tú misma.
2. CONECTIVIDAD EN TIEMPO REAL: Tienes acceso a búsqueda en internet. Si el usuario pregunta por datos actuales (clima, precios, noticias, eventos, resultados deportivos, información de hoy), usa tu herramienta de búsqueda para dar datos exactos y actualizados.
3. TONO Y PERSONALIDAD: Hablas como una mujer paisa culta, cercana, carismática, directa y resolutiva. Adapta tu lenguaje: cálida con conocidos, ejecutiva y persuasiva con clientes. Evita sonar robótica o repetitiva; sé concisa pero sustanciosa (máximo 3 o 4 líneas por mensaje para WhatsApp).
4. CONOCIMIENTO DE MARCA (MULTYDRINK BE HEALTHY): Si la conversación gira en torno a bienestar, salud, cansancio, dolores, energía o emprendimiento, asesora de forma experta sobre los beneficios de MultyDrink Be Healthy y sus oportunidades de distribución.
5. REGLA DE ESCALAMIENTO ESTRICTA: ÚNICAMENTE cuando el cliente pida cerrar una compra directa, agendar una cita presencial, enviar un comprobante de pago o tratar un asunto privado urgente que requiera la intervención física de Viviana, responde lo necesario y cierra con: "Un momento, este punto te lo resuelve directamente Viviana Marín; ya mismo me comunico con ella para que te atienda personalmente." Para TODO lo demás, resuélvelo tú misma al 100%.
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "buscar_en_internet",
            "description": "Busca información en tiempo real en internet sobre noticias, clima, precios, datos actuales, lugares o cualquier consulta que requiera información actualizada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "Términos de búsqueda claros y precisos."
                    }
                },
                "required": ["consulta"]
            }
        }
    }
]

def ejecutar_busqueda(consulta):
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(consulta, region="co-es", max_results=4))
            if not resultados:
                return "No se encontraron resultados recientes."
            texto = "\n".join([f"- {r['title']}: {r['body']}" for r in resultados])
            return texto
    except Exception as e:
        return f"Error al consultar internet: {str(e)}"

@app.route("/", methods=["GET"])
def home():
    return "Clon Inteligente Viviana Marín - Activo y Conectado", 200

@app.route("/webhook", methods=["POST"])
def webhook():
    if not client:
        return jsonify({"replies": [{"message": "Error: Falta configurar GROQ_API_KEY en Render."}]})

    data = request.json or {}
    query = data.get("query", {})
    remitente = query.get("sender", "usuario_general")
    mensaje = query.get("message", "").strip()

    if not mensaje:
        return jsonify({"replies": []})

    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")
    prompt_con_fecha = f"{SYSTEM_PROMPT}\nFecha y hora actual en Colombia: {fecha_actual}."

    historial = historial_chats.get(remitente, [])
    mensajes = [{"role": "system", "content": prompt_con_fecha}] + historial[-8:] + [{"role": "user", "content": mensaje}]

    try:
        respuesta_modelo = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=mensajes,
            tools=tools,
            tool_choice="auto",
            temperature=0.6,
            max_tokens=600
        )

        msg_objeto = respuesta_modelo.choices[0].message

        if msg_objeto.tool_calls:
            mensajes.append(msg_objeto)
            for tool_call in msg_objeto.tool_calls:
                if tool_call.function.name == "buscar_en_internet":
                    args = json.loads(tool_call.function.arguments)
                    info_web = ejecutar_busqueda(args.get("consulta", mensaje))
                    mensajes.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "buscar_en_internet",
                        "content": info_web
                    })

            segunda_respuesta = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=mensajes,
                temperature=0.6,
                max_tokens=600
            )
            texto_final = segunda_respuesta.choices[0].message.content.strip()
        else:
            texto_final = msg_objeto.content.strip()

    except Exception as e:
        texto_final = "Dame un segundo que estoy procesando la información, ¿me repites por favor?"

    historial_chats[remitente] = (
        historial + 
        [{"role": "user", "content": mensaje}, {"role": "assistant", "content": texto_final}]
    )[-10:]

    return jsonify({"replies": [{"message": texto_final}]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

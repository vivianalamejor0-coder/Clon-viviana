import os
import json
from flask import Flask, request, jsonify
from groq import Groq
from duckduckgo_search import DDGS

app = Flask(__name__)

api_key = os.environ.get("GROQ_API_KEY", "").strip()
client = Groq(api_key=api_key) if api_key else None

historial_usuarios = {}

SYSTEM_PROMPT = """Eres el asistente virtual inteligente y clon digital de Viviana Marín Jiménez.
Eres cálida, profesional, empática, perspicaz y resuelves dudas con criterio propio y naturalidad.
Respondes mensajes de WhatsApp de forma clara, cercana y persuasiva, adaptándote al contexto de la conversación sin sonar robótica ni repetitiva.
Si te preguntan por temas de bienestar, liderazgo, emprendimiento o productos, orientas con seguridad y calidez.
Mantén las respuestas concisas y naturales para chat (máximo 2 o 3 párrafos cortos), usando un tono conversacional colombiano profesional."""

def buscar_en_internet(consulta: str) -> str:
    """Busca información actualizada en la web de forma aislada y segura."""
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(consulta, max_results=3))
            if not resultados:
                return "No se encontraron resultados recientes en la web."
            resumen = [f"- {r.get('title', '')}: {r.get('body', '')}" for r in resultados]
            return "\n".join(resumen)
    except Exception as e:
        return f"Búsqueda web no disponible temporalmente ({str(e)}). Responde con tu conocimiento general."

HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_en_internet",
            "description": "Busca información actual en internet sobre noticias, precios, eventos o datos que requieran verificación en tiempo real.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "Términos de búsqueda claros y concisos"
                    }
                },
                "required": ["consulta"]
            }
        }
    }
]

def obtener_modelos_activos():
    """Consulta directamente a Groq los modelos de chat activos en la cuenta."""
    modelos_disponibles = client.models.list().data
    # Filtrar modelos de audio/whisper o moderación que no sirven para chat
    excluidos = ("whisper", "guard", "tts", "playai", "distil-whisper")
    candidatos = [
        m.id for m in modelos_disponibles
        if not any(x in m.id.lower() for x in excluidos)
    ]
    # Priorizar modelos grandes/versátiles de chat si están en la lista
    candidatos.sort(key=lambda x: (
        0 if "70b" in x.lower() or "120b" in x.lower() else
        1 if "32b" in x.lower() or "17b" in x.lower() or "llama" in x.lower() else 2
    ))
    return candidatos

def llamar_groq_dinamico(mensajes, usar_herramientas=True):
    """Detecta los modelos activos en tiempo real y ejecuta la consulta."""
    modelos = obtener_modelos_activos()
    if not modelos:
        raise RuntimeError("No se encontraron modelos de chat activos en tu cuenta de Groq.")
    
    ultimo_error = None
    for modelo in modelos:
        try:
            kwargs = {
                "model": modelo,
                "messages": mensajes,
                "temperature": 0.7,
                "max_tokens": 500
            }
            if usar_herramientas:
                kwargs["tools"] = HERRAMIENTAS
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs), modelo
        except Exception as e:
            ultimo_error = e
            # Reintentar ese mismo modelo sin herramientas si no soporta function calling
            if usar_herramientas:
                try:
                    return client.chat.completions.create(
                        model=modelo,
                        messages=mensajes,
                        temperature": 0.7,
                        max_tokens": 500
                    ), modelo
                except Exception as e2:
                    ultimo_error = e2
            continue
    raise ultimo_error

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "servicio": "Clon-viviana",
        "groq_configurado": bool(api_key)
    }), 200

@app.route("/webhook", methods=["POST"])
@app.route("/", methods=["POST"])
def webhook():
    try:
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        
        query = data.get("query") or data.get("message") or {}
        if isinstance(query, dict):
            mensaje_usuario = query.get("message", "").strip()
            remitente = query.get("sender", "usuario_default")
        else:
            mensaje_usuario = str(query).strip()
            remitente = data.get("sender", "usuario_default")

        if not mensaje_usuario:
            return jsonify({"replies": []}), 200

        if not client:
            return jsonify({
                "replies": [{"message": "⚠️ Falta configurar GROQ_API_KEY en Render."}]
            }), 200

        if remitente not in historial_usuarios:
            historial_usuarios[remitente] = []
        
        historial_usuarios[remitente].append({"role": "user", "content": mensaje_usuario})
        historial_usuarios[remitente] = historial_usuarios[remitente][-10:]

        mensajes = [{"role": "system", "content": SYSTEM_PROMPT}] + historial_usuarios[remitente]

        respuesta, modelo_activo = llamar_groq_dinamico(mensajes, usar_herramientas=True)
        mensaje_ia = respuesta.choices[0].message

        if getattr(mensaje_ia, "tool_calls", None):
            mensajes.append({
                "role": "assistant",
                "content": mensaje_ia.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in mensaje_ia.tool_calls
                ]
            })

            for tool_call in mensaje_ia.tool_calls:
                if tool_call.function.name == "buscar_en_internet":
                    args = json.loads(tool_call.function.arguments)
                    resultado_web = buscar_en_internet(args.get("consulta", ""))
                    mensajes.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "buscar_en_internet",
                        "content": resultado_web
                    })

            segunda_respuesta = client.chat.completions.create(
                model=modelo_activo,
                messages=mensajes,
                temperature=0.7,
                max_tokens=500
            )
            texto_final = segunda_respuesta.choices[0].message.content.strip()
        else:
            texto_final = (mensaje_ia.content or "").strip()

        if not texto_final:
            texto_final = "¡Hola! Cuéntame un poco más para poder ayudarte mejor."

        historial_usuarios[remitente].append({"role": "assistant", "content": texto_final})

        return jsonify({
            "replies": [
                {"message": texto_final}
            ]
        }), 200

    except Exception as e:
        return jsonify({
            "replies": [
                {"message": f"[Diagnóstico Clon]: {type(e).__name__} - {str(e)}"}
            ]
        }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

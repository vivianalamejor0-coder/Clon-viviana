import os
import re
from datetime import datetime
from flask import Flask, request, jsonify
from groq import Groq
from duckduckgo_search import DDGS

app = Flask(__name__)

# 1. Validación estricta de credenciales
api_key = os.environ.get("GROQ_API_KEY", "").strip()
client = Groq(api_key=api_key) if api_key else None

# 2. Memoria conversacional aislada por remitente
historial_usuarios = {}

def construir_system_prompt() -> str:
    fecha_actual = datetime.now().strftime("%d/%m/%Y")
    return f"""Eres el clon digital e inteligencia virtual de Viviana Marín Jiménez. Fecha actual: {fecha_actual}.

PERSONALIDAD Y CRITERIO:
- Eres cálida, cercana, empática, perspicaz y profesional, con un trato natural estilo colombiano.
- Tienes capacidad de análisis y razonamiento propio: comprendes la intención real detrás de cada mensaje, conectas ideas con coherencia y respondes con seguridad.
- Orientas con autoridad y amabilidad en temas de bienestar, salud preventiva, liderazgo, emprendimiento, ventas y desarrollo personal, así como cualquier consulta general o cotidiana.

REGLAS DE RESPUESTA PARA WHATSAPP:
- Responde directamente al último mensaje del usuario manteniendo el hilo de la conversación.
- Si recibes información verificada de internet en el mensaje del usuario, analízala y intégrala de forma natural en tu respuesta sin decir "según los resultados de búsqueda".
- Sé clara, persuasiva y concisa (entre 1 y 3 párrafos cortos, fáciles de leer en celular), sin sonar robótica ni repetir frases hechas."""

DISPARADORES_WEB = (
    "clima", "tiempo", "temperatura", "lluvia", "pronóstico", "pronostico",
    "hoy", "ayer", "mañana", "actual", "precio", "valor", "dólar", "dolar",
    "noticia", "noticias", "cuánto", "cuanto", "qué pasó", "que paso",
    "resultado", "partido", "hora", "fecha", "quién", "quien", "dónde", "donde",
    "cuándo", "cuando", "cómo funciona", "como funciona", "qué es", "que es",
    "busca", "investiga", "averigua", "confirma"
)

def requiere_consulta_web(texto: str) -> bool:
    t = texto.lower().strip()
    if len(t) < 4:
        return False
    return "?" in t or any(palabra in t for palabra in DISPARADORES_WEB)

def buscar_en_internet(consulta: str) -> str:
    """Consulta DuckDuckGo enfocada en español/Colombia con tolerancia a fallos."""
    try:
        with DDGS() as ddgs:
            resultados = list(ddgs.text(consulta, region="co-es", max_results=4))
            if not resultados:
                resultados = list(ddgs.text(consulta, max_results=3))
            if not resultados:
                return ""
            lineas = []
            for r in resultados:
                titulo = r.get("title", "").strip()
                cuerpo = r.get("body", "").strip()
                if titulo or cuerpo:
                    lineas.append(f"- {titulo}: {cuerpo}")
            return "\n".join(lineas)
    except Exception:
        return ""

def obtener_modelos_activos() -> list:
    """Obtiene en tiempo real los modelos de texto vigentes en la cuenta de Groq."""
    modelos_disponibles = client.models.list().data
    excluidos = ("whisper", "guard", "tts", "playai", "distil-whisper", "safeguard")
    candidatos = [
        m.id for m in modelos_disponibles
        if not any(x in m.id.lower() for x in excluidos)
    ]
    candidatos.sort(key=lambda x: (
        0 if ("70b" in x.lower() or "120b" in x.lower()) else
        1 if ("32b" in x.lower() or "17b" in x.lower() or "llama" in x.lower()) else 2
    ))
    return candidatos

def razonar_con_groq(mensajes: list) -> str:
    """Ejecuta el razonamiento sobre el mejor modelo activo disponible."""
    modelos = obtener_modelos_activos()
    if not modelos:
        raise RuntimeError("No hay modelos de chat activos disponibles en tu cuenta de Groq.")

    ultimo_error = None
    for modelo in modelos:
        try:
            respuesta = client.chat.completions.create(
                model=modelo,
                messages=mensajes,
                temperature=0.65,
                max_tokens=700,
                top_p=0.9
            )
            texto = (respuesta.choices[0].message.content or "").strip()
            texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
            if texto:
                return texto
        except Exception as e:
            ultimo_error = e
            continue

    raise ultimo_error or RuntimeError("Ningún modelo pudo generar respuesta.")

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
            mensaje_usuario = str(query.get("message", "")).strip()
            remitente = str(query.get("sender", "usuario_default")).strip()
        else:
            mensaje_usuario = str(query).strip()
            remitente = str(data.get("sender", "usuario_default")).strip()

        if not mensaje_usuario:
            return jsonify({"replies": []}), 200

        if not client:
            return jsonify({
                "replies": [{"message": "⚠️ Falta configurar GROQ_API_KEY en Render."}]
            }), 200

        if mensaje_usuario.lower() in ("reiniciar", "reset", "borrar memoria"):
            historial_usuarios[remitente] = []
            return jsonify({
                "replies": [{"message": "Listo, memoria reiniciada. ¿En qué te puedo ayudar?"}]
            }), 200

        historial_previo = historial_usuarios.get(remitente, [])[-8:]

        bloque_web = ""
        if requiere_consulta_web(mensaje_usuario):
            datos_web = buscar_en_internet(mensaje_usuario)
            if datos_web:
                bloque_web = f"\n\n[Contexto actualizado de internet]:\n{datos_web}"

        mensajes = (
            [{"role": "system", "content": construir_system_prompt()}]
            + historial_previo
            + [{"role": "user", "content": mensaje_usuario + bloque_web}]
        )

        texto_final = razonar_con_groq(mensajes)

        historial_usuarios[remitente] = historial_previo + [
            {"role": "user", "content": mensaje_usuario},
            {"role": "assistant", "content": texto_final}
        ]

        return jsonify({
            "replies": [{"message": texto_final}]
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

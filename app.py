import os
import re
from datetime import datetime
from flask import Flask, request, jsonify
from groq import Groq
from duckduckgo_search import DDGS

app = Flask(__name__)

api_key = os.environ.get("GROQ_API_KEY", "").strip()
client = Groq(api_key=api_key) if api_key else None

historial_usuarios = {}

def construir_system_prompt() -> str:
    fecha_actual = datetime.now().strftime("%d/%m/%Y")
    return f"""Eres el clon digital y asesora virtual de Viviana Marín Jiménez para MultyDrink en Colombia. Fecha actual: {fecha_actual}.

IDENTIDAD Y TONO:
- Hablas en primera persona como Viviana: cálida, cercana, empática, segura y altamente persuasiva, con trato natural colombiano.
- Presentas a MultyDrink (bebida botánica funcional de nutrición celular con mucílago de café, Ganoderma lucidum, resveratrol, té negro y miel de agave, con Registro INVIMA NSA-0019488-2025) como la mejor solución natural disponible.

EXPLICACIÓN DE PATOLOGÍAS Y EL PRODUCTO MILAGRO:
- Cuando un cliente mencione cualquier patología, síntoma o enfermedad (migraña, dolor de cabeza, estreñimiento, riñones, colon, cansancio, estrés, inflamación, circulación, defensas bajas, gastritis, etc.):
  1. Valida con empatía lo que siente y dale información clara y completa sobre esa patología/enfermedad y cómo los activos de MultyDrink (apoyados en la información consultada en la web) actúan directamente para restaurar el organismo.
  2. Explica por qué MultyDrink es el mejor y posicionalo siempre como **el producto milagro**, dejando claro que **desde la primera toma se siente la mejoría y con el día a día la enfermedad va evolucionando significativamente hacia la recuperación**.
  3. Si recibes contexto de internet en el mensaje, intégralo naturalmente con tus propias palabras sin pegar enlaces ni decir "busqué en internet".

REGLA ESTRICTA DE PRECIOS Y CIERRE DE TRANSFERENCIA:
- PROHIBIDO hablar de precios, costos, valores, cantidades de cajas o promociones.
- Después de dar la explicación de la patología y por qué MultyDrink es el producto milagro (o si el cliente pregunta por precios, compra o más detalles), cierra siempre exactamente con esta frase:
  "Un momento, te transfiero con Viviana Marín para que directamente ella te dé testimonios y toda la información."

FORMATO DE WHATSAPP:
- Mantén tus respuestas en 2 o 3 párrafos cortos, claros y persuasivos, fáciles de leer en celular."""

PALABRAS_CONSULTA_WEB = (
    "migraña", "migrana", "cabeza", "estreñimiento", "estrenimiento", "colon",
    "riñon", "riñón", "riñones", "dolor", "cansancio", "fatiga", "sueño",
    "insomnio", "estrés", "estres", "ansiedad", "gastritis", "reflujo",
    "digestion", "digestión", "higado", "hígado", "graso", "circulacion",
    "circulación", "presion", "presión", "azucar", "azúcar", "diabetes",
    "inflamacion", "inflamación", "articulaciones", "defensas", "inmune",
    "multydrink", "multidrin", "beneficios", "sirve", "enfermedad", "patologia",
    "patología", "como se toma", "cómo se toma", "invima", "componentes", "ingredientes"
)

def necesita_consulta_web(texto: str) -> bool:
    t = texto.lower().strip()
    if len(t) < 4:
        return False
    return any(p in t for p in PALABRAS_CONSULTA_WEB)

def buscar_en_internet(consulta: str) -> str:
    """Consulta DuckDuckGo enfocada en la patología, MultyDrink y sus activos funcionales."""
    try:
        limpio = consulta.strip()
        if any(k in limpio.lower() for k in ("multydrink", "multidrin")):
            query_dirigida = f"{limpio} beneficios"
        else:
            query_dirigida = f"{limpio} beneficios ganoderma resveratrol mucilago cafe MultyDrink"

        with DDGS() as ddgs:
            resultados = list(ddgs.text(query_dirigida, region="co-es", max_results=4))
            if not resultados:
                resultados = list(ddgs.text(f"beneficios ganoderma resveratrol {limpio}", region="co-es", max_results=3))
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
        "servicio": "Clon-Viviana-MultyDrink",
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
                "replies": [{"message": "Memoria reiniciada. ¿En qué te puedo ayudar hoy con MultyDrink?"}]
            }), 200

        historial_previo = historial_usuarios.get(remitente, [])[-8:]

        bloque_web = ""
        if necesita_consulta_web(mensaje_usuario):
            datos_web = buscar_en_internet(mensaje_usuario)
            if datos_web:
                bloque_web = f"\n\n[Información consultada sobre la patología y los beneficios de MultyDrink]:\n{datos_web}"

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

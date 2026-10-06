# Documentación del agente de georreferenciación

Este documento describe en detalle el agente que georreferencia mapas: qué hace, cómo está construido, qué herramientas tiene, cómo decide y qué garantías da el sistema alrededor de él. Se basa en el código de `src/georef/agent/` y de los módulos con los que se integra. El diseño general está en [PLAN.md](../PLAN.md) y el uso del CLI en [README.md](../README.md).

## Índice

1. [Resumen](#1-resumen)
2. [Principio de diseño](#2-principio-de-diseño-el-modelo-decide-el-código-mide)
3. [Dónde encaja el agente en el flujo](#3-dónde-encaja-el-agente-en-el-flujo)
4. [Componentes](#4-componentes)
5. [Configuración](#5-configuración)
6. [El bucle del agente](#6-el-bucle-del-agente)
7. [Prompt de sistema y mensaje inicial](#7-prompt-de-sistema-y-mensaje-inicial)
8. [Herramientas](#8-herramientas)
9. [Estado de la sesión (`MapSession`)](#9-estado-de-la-sesión-mapsession)
10. [Cierre del agente y decisión del estado final](#10-cierre-del-agente-y-decisión-del-estado-final)
11. [Chequeos de coherencia](#11-chequeos-de-coherencia)
12. [Manejo de errores y casos límite](#12-manejo-de-errores-y-casos-límite)
13. [Trazabilidad](#13-trazabilidad)
14. [Costo y uso de tokens](#14-costo-y-uso-de-tokens)
15. [Pruebas](#15-pruebas)
16. [Cómo ejecutarlo](#16-cómo-ejecutarlo)
17. [Limitaciones y trabajo pendiente](#17-limitaciones-y-trabajo-pendiente)
18. [Mapa de archivos](#18-mapa-de-archivos)

---

## 1. Resumen

El agente recibe la imagen de un mapa (escaneo, foto o PDF) y deja un resultado georreferenciado, leyendo la cuadrícula y las coordenadas impresas en el propio mapa.

- **Modelo:** Claude Opus 5.5 (`claude-opus-5-5`) por Vertex AI. Con `GEOREF_PROVIDER=anthropic` usa la API directa de Anthropic.
- **Construcción:** SDK de Anthropic con `client.beta.messages.tool_runner`. No usa LangChain ni LangGraph.
- **Herramientas:** 14, escritas en Python y ejecutadas en local sobre un objeto `MapSession`.
- **Razonamiento:** adaptativo, con resumen visible (`thinking={"type": "adaptive", "display": "summarized"}`), y profundidad regulada con `effort` (por defecto `high`).
- **Cierre:** una sola llamada a `submit_result` con estado `ok`, `needs_review`, `no_coordinates` o `failed`.
- **Salvaguarda:** un `ok` del agente no basta; los chequeos de coherencia pueden rebajarlo a `needs_review`.
- **Trazas:** LangSmith, una traza por mapa, opcional.

El agente no es el único componente que produce un resultado. Antes de que él actúe, un pre-análisis determinista (sin modelo) ya intenta resolver el mapa. El agente verifica ese resultado, lo corrige y decide cuándo no es confiable.

## 2. Principio de diseño: el modelo decide, el código mide

Un modelo de visión no ubica un píxel con la precisión que exige un punto de control, y un detector de líneas no sabe qué número es una coordenada ni qué datum dice la leyenda. El trabajo se reparte así:

| Quién | Qué aporta |
|---|---|
| **OpenCV** | Posiciones en píxeles: líneas, cruces y marcas con precisión sub-píxel. |
| **Document AI** | El texto, con su caja en píxeles y su confianza. |
| **El agente** | El criterio: qué etiqueta corresponde a qué línea, qué sistema de coordenadas dice la leyenda, qué lectura está equivocada y cuándo el resultado no merece confianza. |

Consecuencias en el diseño:

- Toda herramienta trabaja en **píxeles de la imagen original**, con el origen arriba a la izquierda.
- Cada imagen que ve el agente lleva **reglas con esos píxeles** en los bordes. Lo que lee en el borde lo puede usar directamente como región de otra herramienta, sin hacer cuentas de escala.
- El agente **usa el zoom para comprobar, nunca para estimar a ojo** la posición final de un punto. Cuando agrega un punto manual, `edit_gcps` lo ajusta a la esquina o cruce más cercano.
- La imagen completa **nunca** se envía al modelo en alta resolución: ve una vista general reducida y recortes bajo demanda.
- El estado pesado (imagen, índice OCR, grilla) vive en `MapSession`, no en el contexto del modelo.

## 3. Dónde encaja el agente en el flujo

Flujo por mapa (`pipeline.py`): `ingesta → pre-análisis determinista → agente → exportación`.

```
open_session            carga la imagen (TIFF/JPG/PNG/BMP/JP2/WEBP o PDF) y crea la MapSession
        │
process_session  (traza raíz "georef_map")
        │
        ├─ run_prepass            sin modelo
        │    ├─ detect_grid("lines")      líneas de la cuadrícula
        │    ├─ ocr_tiles                 OCR por mosaicos + bandas laterales (si hay proveedor)
        │    └─ auto_fit                  etiquetas, valores, CRS, primer ajuste
        │
        ├─ run_agent              el agente (opcional: --no-agent lo omite)
        │
        ├─ fit_transform("auto")  si quedó sin ajuste pero hay ≥ 3 puntos
        ├─ decide_status          el agente propone; los chequeos deciden
        └─ export                 GeoTIFF, puntos, world file, JSON, qa_overlay, trace.jsonl
```

### 3.1 Pre-análisis determinista

Es lo que el agente recibe como hipótesis de partida. Hace lo siguiente:

1. Detecta las líneas de la cuadrícula en modo `lines`.
2. Corre OCR por mosaicos (2200 px con solapamiento de 260 px) sobre toda la imagen y sobre las bandas laterales, con las rotaciones necesarias para las etiquetas verticales. El OCR se cachea en disco por hash.
3. Interpreta las etiquetas de coordenada (UTM, grados-minutos-segundos, grados decimales) y las asocia a la línea más cercana.
4. Ajusta la progresión de valores por eje (intervalo constante), completa las líneas sin etiqueta y marca las lecturas incoherentes.
5. Si las líneas no resuelven ambos ejes, prueba `crosses` y luego `ticks`; se queda con el primer modo que resuelve los dos ejes. Si ninguno basta, vuelve a las líneas porque son lo más informativo para el agente.
6. Propone el sistema de coordenadas a partir de la leyenda (datum, zona, hemisferio), de la pista regional (`GEOREF_REGION_HINT`) y del datum por defecto (`GEOREF_DEFAULT_DATUM`).
7. Ajusta una transformación afín (u otra si el afín falla de verdad) y retira las líneas que quedan inconsistentes.

Si no hay proveedor de OCR configurado, el pre-análisis lo avisa en `warnings` ("Sin proveedor de OCR: no se leyeron etiquetas") y sigue.

### 3.2 Qué recibe el agente

`summarize()` produce el resumen que se le envía como JSON (y que también va a `georef.json`):

| Campo | Contenido |
|---|---|
| `image` | Ancho, alto y DPI. |
| `ref_grid` | Descripción de la rejilla de referencia (columnas, filas, tamaño de celda). |
| `frame` | Marco del mapa. |
| `grid` | Modo usado, número de líneas verticales y horizontales, líneas con valor asignado. |
| `ocr_texts` | Cantidad de textos en el índice OCR. |
| `labels` | Total, conteo por margen, aceptadas, hasta 12 incoherentes, abreviadas. |
| `label_kind` | `utm` o `geo`. |
| `axis_fits` | Ajuste valor↔posición por eje. |
| `crs` | EPSG, nombre, si es confiable, evidencia y candidatos. |
| `gcps` | Cantidad de puntos de control habilitados. |
| `fit` | Resumen del ajuste (RMSE, validación cruzada, etc.). |
| `validation` | Resultado de los chequeos, si hay ajuste. |
| `warnings` | Avisos acumulados. |

## 4. Componentes

El paquete `src/georef/agent/` tiene cuatro módulos:

| Módulo | Responsabilidad |
|---|---|
| [client.py](../src/georef/agent/client.py) | Crea el cliente del modelo (Vertex o Anthropic directo) y lo envuelve para trazas. |
| [prompts.py](../src/georef/agent/prompts.py) | Prompt de sistema y texto del primer mensaje. |
| [tools.py](../src/georef/agent/tools.py) | Las 14 herramientas, ligadas a una sesión con `build_tools(session, settings)`. |
| [runner.py](../src/georef/agent/runner.py) | El bucle: arma el primer mensaje, lanza el `tool_runner`, cuenta turnos y tokens, escribe `trace.jsonl` y maneja errores. |

### 4.1 Cliente (`client.py`)

`make_client(settings)`:

- Si `GEOREF_PROVIDER=anthropic`, usa `anthropic.Anthropic()` (API directa).
- En cualquier otro caso usa `AnthropicVertex(project_id=settings.gcp_project, region=settings.vertex_region)`. Si falta `GCP_PROJECT`, lanza `ValueError("Falta GCP_PROJECT en el entorno.")`.
- Pasa el cliente por `tracing.wrap_client`, que registra cada llamada al modelo en LangSmith cuando las trazas están activas.

`wrap_anthropic` de LangSmith asume que el cliente tiene `client.completions`, algo que `AnthropicVertex` no tiene. `tracing.wrap_client` añade ese atributo (con un `create` que falla si se usa) antes de envolver. Con eso sí captura los turnos que lanza el `tool_runner`.

## 5. Configuración

`Settings.from_env()` ([config.py](../src/georef/config.py)) lee el `.env`. Lo que afecta al agente:

| Variable | Campo | Por defecto | Efecto |
|---|---|---|---|
| `CLAUDE_MODEL` | `model` | `claude-opus-5-5` | Modelo del agente. |
| `CLAUDE_EFFORT` | `effort` | `high` | Esfuerzo del razonamiento: `low`, `medium`, `high`, `xhigh`, `max`. |
| `GEOREF_MAX_TURNS` | `max_turns` | `40` | Tope de turnos del bucle (`max_iterations`). |
| — | `max_tokens` | `16000` | Tope de salida por turno. |
| `GEOREF_PROVIDER` | — | `vertex` | `anthropic` usa la API directa en vez de Vertex. |
| `GCP_PROJECT` | `gcp_project` | vacío | Proyecto de Vertex AI. |
| `VERTEX_REGION` | `vertex_region` | `global` | Región de Vertex AI. |
| — | `view_side` | `1568` | Lado largo de la vista general y de las vistas por defecto. |
| — | `view_side_max` | `2576` | Tope del lado largo de cualquier vista. |
| — | `ok_rmse_px` | `2.0` | RMSE máximo en píxeles para dar un mapa por bueno. |
| — | `ok_min_gcps` | `6` | Mínimo de puntos de control para dar un mapa por bueno. |
| `GEOREF_REGION_HINT` | `region_hint` | vacío | Zonas UTM candidatas cuando el mapa no dice la zona, p. ej. `17S,18S,19S`. |
| `GEOREF_DEFAULT_DATUM` | `default_datum` | vacío | Datum a asumir cuando la leyenda no lo indica (`WGS84`, `PSAD56`, `SIRGAS2000`, `SAD69`). |
| `GEOREF_TRACE_IMAGES` | `trace_images` | `on` | Registra miniaturas en LangSmith; `off` deja solo metadatos. |

Por la línea de comandos, `--model`, `--effort`, `--region-hint` y `--default-datum` sustituyen los valores del entorno en esa corrida.

El `effort` por defecto de Opus 5.5 es `medium`; el proyecto lo fija en `high` porque con esfuerzo alto usa mejor las herramientas de imagen. El plan prevé medir si `medium` alcanza.

## 6. El bucle del agente

[runner.py](../src/georef/agent/runner.py), función `run_agent(session, settings, client, prepass, out_dir)`.

### 6.1 Primer mensaje

El primer turno de usuario lleva dos bloques:

1. **Una imagen:** la vista general del mapa completo (`render_view` con `settings.view_side` y la rejilla de referencia activada).
2. **Un texto:** `initial_text(nombre, prepass)`, con el nombre del archivo y el resumen del pre-análisis en JSON, seguido de "Comprueba este resultado, corrige lo que haga falta y cierra con submit_result."

### 6.2 Llamada al modelo

```python
runner = client.beta.messages.tool_runner(
    model=settings.model,
    max_tokens=settings.max_tokens,
    max_iterations=settings.max_turns,
    system=SYSTEM_PROMPT,
    tools=build_tools(session, settings),
    messages=messages,
    thinking={"type": "adaptive", "display": "summarized"},
    output_config={"effort": settings.effort},
    cache_control={"type": "ephemeral"},
)
```

Puntos a tener en cuenta con este modelo:

- El razonamiento no se desactiva y no acepta `budget_tokens` ni `temperature`; solo se regula con `effort`.
- No se fuerza `tool_choice`. El cierre se pide por prompt. Hay una prueba que verifica que la petición no lleva `tool_choice`.
- `cache_control` de tipo `ephemeral` activa la caché de prompts. Como cada turno reenvía toda la conversación, la caché reduce el costo de los turnos posteriores al primero.
- Las herramientas pueden devolver bloques de imagen en el resultado, así que las vistas llegan al modelo como texto más imagen.

### 6.3 Ciclo por turno

```python
for message in runner:
    report["turns"] += 1
    report["stop_reason"] = message.stop_reason
    # acumula tokens de uso
    _log_message(...)                       # thinking, texto y tool_use a trace.jsonl
    if message.stop_reason == "refusal":
        break
    _log_results(..., runner.generate_tool_call_response())   # resultados de herramientas
    if session.result is not None:          # submit_result ya se llamó
        break
```

El bucle termina cuando ocurre alguno de estos casos:

| Condición | Resultado |
|---|---|
| El modelo llama a `submit_result` | `session.result` queda fijado y el bucle sale. |
| `stop_reason == "refusal"` | Sale; el mapa queda con estado `failed` y nota "El modelo rechazó la solicitud." |
| El modelo termina sin llamar a `submit_result` | Aviso en `warnings`: "El agente terminó sin llamar a submit_result (…)". |
| Se alcanza `max_turns` | Mismo aviso (límite de turnos). |
| Error de la API | Se guarda en `report["error"]` y se añade un aviso; el mapa se procesa con el resultado del pre-análisis. |

El runner reutiliza el resultado de `generate_tool_call_response()`, de modo que las herramientas no se ejecutan dos veces.

### 6.4 Informe de la corrida

`run_agent` devuelve (y `georef.json` conserva bajo `agent`):

```json
{
  "used": true,
  "model": "claude-opus-5-5",
  "effort": "high",
  "turns": 3,
  "stop_reason": "tool_use",
  "submitted": true,
  "usage": {
    "input_tokens": 3000,
    "output_tokens": 150,
    "cache_read_input_tokens": 1600,
    "cache_creation_input_tokens": 600
  }
}
```

Si hubo un error de API aparece además `error` con el mensaje.

### 6.5 `trace.jsonl`

`TraceWriter` escribe en `out/<mapa>/trace.jsonl` una copia local mínima, una línea JSON por evento, **sin imágenes**:

| `type` | Campos |
|---|---|
| `thinking` | `turn`, `text` (resumen del razonamiento). |
| `text` | `turn`, `text`. |
| `tool_use` | `turn`, `name`, `input`. |
| `tool_result` | `turn`, `is_error`, `text` (máx. 4000 caracteres), `images` (cantidad). |
| `end` | El informe de la corrida (sin `used`). |

Cada registro lleva además `t` (marca de tiempo). Una prueba confirma que el archivo no contiene `base64`.

## 7. Prompt de sistema y mensaje inicial

[prompts.py](../src/georef/agent/prompts.py). El prompt está en español y es deliberadamente **descriptivo, no prescriptivo**: no dicta un procedimiento paso a paso, porque con estos modelos los prompts demasiado prescriptivos empeoran el resultado.

Secciones del prompt de sistema:

| Sección | Qué dice |
|---|---|
| **Objetivo** | Dejar el mapa con una transformación píxel-mundo verificada y un CRS respaldado por evidencia. Si no es posible, decirlo con claridad: un mapa marcado para revisión es un buen resultado; uno mal georreferenciado dado por bueno es el peor, porque nadie lo revisará. |
| **Reparto del trabajo** | Las herramientas miden píxeles; el OCR lee texto; el agente aporta el criterio. No estima a ojo la posición final de ningún punto. |
| **Punto de partida** | Recibe vista general con rejilla (celdas A1, B1, …) y el pre-análisis, que debe tratar como hipótesis (puede estar completo, incompleto o equivocado). Las coordenadas son píxeles originales con origen arriba a la izquierda. |
| **Qué debe quedar comprobado** | (1) **Grilla:** que las líneas detectadas sean las impresas y que el valor de cada una sea el del margen, con zoom en al menos dos zonas alejadas, porque un error de una línea en la numeración desplaza todo el mapa sin subir el error del ajuste. (2) **CRS:** datum, zona y hemisferio leídos en leyenda o márgenes, fijados con `set_crs` citando el texto. (3) **Ajuste:** residuos bajos y sin discrepantes sin explicar; usar `validate`. |
| **Lo que conviene saber** | Conocimiento de dominio (ver 7.1). |
| **Cierre** | Una sola llamada a `submit_result` con el significado de cada estado. |

Cierra con una nota de costo: "Cada vista ocupa contexto. Pide las que necesites para decidir y no repitas las que ya viste."

### 7.1 Conocimiento de dominio incluido en el prompt

- En UTM el Este tiene seis dígitos y el Norte siete en el hemisferio sur. Los valores Este se repiten en todas las zonas: por sí solos no dicen en qué zona está el mapa.
- El datum no se deduce de la geometría. PSAD56 y WGS84 difieren en cientos de metros y el ajuste sale igual de bien con ambos. Si el mapa no lo dice, no se inventa: cierra con `needs_review` y lo explica.
- Las etiquetas suelen ir abreviadas: solo una trae el valor completo ("8 650 000 m N") y las demás los dígitos principales ("51", "52"), a veces con cifras pequeñas en superíndice. Con dos líneas de valor completo por eje, `assign_labels` reconstruye el resto.
- Las coordenadas Norte suelen estar rotuladas en vertical en los márgenes laterales; si faltan, `ocr_region` con `rotation=90` las lee.
- Un mapa puede traer dos sistemas a la vez (cuadrícula UTM y marcas geográficas en el marco): usar uno para los valores de las líneas y el otro como comprobación.
- Un mapa rotulado en grados puede estar dibujado en una proyección; `rank_crs` compara.
- Sin líneas completas, probar `detect_grid` con `crosses` o `ticks`; si tampoco, agregar puntos con `edit_gcps` en las esquinas del marco cuando sus coordenadas están impresas.

## 8. Herramientas

Las 14 herramientas se definen con el decorador `@beta_tool` del SDK en `build_tools()`. Todas devuelven JSON compacto (`separators=(",", ":")`), y las de vista devuelven además una imagen. Cada una pasa por `run()`, que:

1. Captura cualquier excepción y devuelve `{"error": "<Tipo>: <mensaje>"}` al modelo, para que pueda corregir la llamada en lugar de abortar el bucle.
2. Delega en `tracing.run_tool`, que registra la herramienta como un run de tipo `tool` en LangSmith si las trazas están activas.

### 8.1 Resumen

| Grupo | Herramienta | Qué hace |
|---|---|---|
| Zoom | `view_image` | Vista completa o recorte ampliado, con reglas y rejilla de referencia. |
| OCR | `ocr_region` | OCR de una región; agrega el resultado al índice y reasocia etiquetas. |
| OCR | `search_text` | Busca en el índice OCR ya calculado. |
| Grilla | `detect_frame` | Detecta el marco del mapa. |
| Grilla | `detect_grid` | Detecta la cuadrícula en modo `lines`, `crosses` o `ticks`. |
| Grilla | `view_grid` | Dibuja la grilla reconstruida con ID, valores, etiquetas, puntos y residuos. |
| Coordenadas | `get_label_candidates` | Lista etiquetas de coordenada interpretadas, por margen. |
| Coordenadas | `assign_labels` | Asigna valores a líneas y reconstruye la grilla completa. |
| CRS | `rank_crs` | Resume la evidencia del CRS y compara candidatos por error de ajuste. |
| CRS | `set_crs` | Fija el CRS con evidencia obligatoria. |
| Georef | `edit_gcps` | Lista, agrega, deshabilita o habilita puntos de control. |
| Georef | `fit_transform` | Ajusta la transformación píxel↔mundo. |
| Georef | `validate` | Ejecuta los chequeos de coherencia. |
| Cierre | `submit_result` | Cierra el trabajo con estado, EPSG, confianza y notas. |

### 8.2 Regiones

Casi todas las herramientas aceptan un parámetro `region`. Se interpreta con `MapSession.parse_region`:

| Forma | Ejemplo | Significado |
|---|---|---|
| Vacío (o `all`, `todo`, `full`) | `""` | Todo el mapa. |
| Píxeles | `"1200,800,600,400"` | `x,y,w,h` en píxeles de la imagen original; se recorta a los límites de la imagen. |
| Celda de la rejilla | `"C4"` | Una celda de la rejilla de referencia. |
| Rango de celdas | `"B2:C3"` | Rectángulo entre dos celdas. |

La rejilla de referencia (`RefGrid`) divide el lado mayor de la imagen en unas 10 celdas (`target=10`), con columnas A, B, C… y filas 1, 2, 3… Una celda inválida o fuera de la rejilla produce un error explícito (p. ej. "La celda "Z9" está fuera de la rejilla (A1 a J7)").

### 8.3 `view_image`

```
view_image(region="", max_side=1568, enhance="none", ref_grid=True)
```

Devuelve un bloque de texto con metadatos (`region`, `scale`, `view_size`, `note`) y la imagen.

- `max_side`: se limita al rango 512 a `view_side_max` (2576). Usar 1568 para ubicarse y hasta 2576 para leer texto fino.
- `enhance`: `none`, `clahe`, `binarize`, `sharpen`, `invert`, `gray`, `red`, `green`, `blue`. Los canales de color aíslan tintas de color.
- `ref_grid`: dibuja las celdas A1, B1… en magenta para nombrar zonas.
- Un recorte pequeño se amplía hasta 4 veces.
- Si la escala resultante es menor que 0,5 agrega un `hint`: "Vista reducida: para leer texto pequeño pide una región menor."
- La imagen se codifica en JPEG (calidad 85), o en PNG si es binaria.
- Los bordes llevan reglas en píxeles originales (24 px arriba, 56 px a la izquierda).

### 8.4 `ocr_region`

```
ocr_region(region, rotation=0, upscale=1.0)
```

Lee con Document AI una región y **agrega el resultado al índice OCR**; después reconstruye las etiquetas y vuelve a asociarlas a las líneas.

- `rotation`: 0, 90, 180 o 270, giro horario del recorte antes de leerlo. 90 endereza el texto vertical que se lee de abajo hacia arriba; 270, el que se lee de arriba hacia abajo.
- `upscale`: ampliación previa de 1 a 4. Subir a 2 o 3 para texto muy pequeño.
- Devuelve hasta 80 textos con su caja (`bbox`), confianza y, cuando el texto parece una coordenada, su interpretación (`kind`, `value`, `axis`, `complete`), más `new_in_index` y `index_size`.

### 8.5 `search_text`

```
search_text(pattern="", region="", kind="")
```

Consulta el índice ya calculado, sin volver a leer la imagen.

- `pattern`: expresión regular, sin distinguir mayúsculas.
- `region`: limita la búsqueda.
- `kind`: `utm` o `geo` (coordenadas), `datum`, `zone` (zona UTM), `scale` (escala).
- Devuelve hasta 60 coincidencias y el total.

### 8.6 `detect_frame`

Detecta el marco: primero a partir de las líneas de la cuadrícula y, si no basta, a partir del contorno de la imagen. Devuelve `{"frame": {x, y, w, h, source}}` o `null`.

### 8.7 `detect_grid`

```
detect_grid(mode="lines", kernel_frac=None, gap_frac=None, min_support=None, peak_rel=None)
```

**Reemplaza la grilla anterior**, vuelve a asociar las etiquetas, reajusta la progresión y recalcula el ajuste si hay puntos suficientes.

| Parámetro | Efecto |
|---|---|
| `mode` | `lines` líneas completas; `crosses` cruces sueltas en las intersecciones; `ticks` marcas cortas sobre el marco. |
| `kernel_frac` | Largo mínimo de trazo continuo, relativo al lado menor (por defecto 0,06). Bajarlo si faltan líneas cortas. |
| `gap_frac` | Cortes que se puentean, relativo al lado menor (por defecto 0). Subirlo (0,01) para líneas de trazos o interrumpidas por texto. |
| `min_support` | Fracción mínima de la línea con tinta (por defecto 0,35). Bajarlo para líneas tenues. |
| `peak_rel` | Largo mínimo frente a las líneas más largas (por defecto 0,35). |

Devuelve `info` del detector, el marco, la grilla en formato breve, cantidad de puntos de control, el ajuste y los avisos.

**Formato breve de la grilla:** cada línea se escribe `ID@posición=valor`. Por ejemplo `V3@300=319000` es la línea vertical V3, ubicada en x = 300 px, con valor 319000. Si no tiene valor se omite el `=…`; si está fuera de la retícula se marca `(fuera de retícula)`.

### 8.8 `view_grid`

```
view_grid(region="", max_side=1568, show="lines,values,labels,gcps,residuals")
```

Dibuja la grilla reconstruida sobre el mapa. Capas (`show`): `lines`, `values`, `labels`, `gcps`, `residuals`. Si no se indica ninguna capa válida usa `lines,values`.

Leyenda de colores:

| Elemento | Color |
|---|---|
| Líneas verticales | Azul |
| Líneas horizontales | Rojo |
| Líneas fuera de la retícula | Gris |
| Cajas de etiquetas aceptadas | Verde |
| Cajas de etiquetas incoherentes | Rojo |
| Cajas de etiquetas sin asignar | Amarillo |
| Puntos de control habilitados | Círculo verde (gris si están deshabilitados) |
| Residuos | Naranja, **ampliados 25 veces** |

Las líneas con valor se dibujan más gruesas. Es la herramienta principal para comprobar que cada valor corresponde a la etiqueta que se lee en el margen. La misma capa completa se usa para generar `qa_overlay.png` en la exportación.

### 8.9 `get_label_candidates`

```
get_label_candidates(side="")
```

Lista las etiquetas de coordenada interpretadas a partir del OCR, con la línea a la que se asociaron. `side`: `top`, `bottom`, `left`, `right`, `inside`, `corner` o vacío para todos. Devuelve hasta 100 etiquetas y el total, ordenadas por margen y posición.

### 8.10 `assign_labels`

```
assign_labels(assignments=None, kind="", step_x=0.0, step_y=0.0)
```

Es la herramienta central para corregir la numeración.

- **Sin argumentos:** asocia cada etiqueta a su línea más cercana, ajusta la progresión de valores, descarta lecturas incoherentes y deduce el valor de las líneas sin etiqueta.
- **Con `assignments`:** el agente fija el valor de líneas concretas, y eso **manda sobre el OCR**. El resto se propaga. Con dos líneas fijadas por eje basta para definir toda la grilla.
  - Cada elemento es `{"line_id": "V3", "value": "320000"}`.
  - El valor puede ser un número (metros o grados decimales, con signo) o grados-minutos-segundos como `"76°30' W"`.
  - `"none"` (también `""`, `"null"`, `"-"`) excluye una línea que no es de la cuadrícula.
- `kind`: `utm` o `geo`; vacío para deducirlo.
- `step_x`, `step_y`: intervalo entre líneas verticales u horizontales, si se conoce; 0 para deducirlo.

Devuelve `label_kind`, el ajuste por eje (`axis_fits`), la grilla en formato breve, hasta 12 etiquetas incoherentes (`outlier_labels`), cantidad de puntos de control, el ajuste y los avisos. Falla con un mensaje claro si un `line_id` no existe o si un valor no se puede interpretar como coordenada.

### 8.11 `rank_crs`

```
rank_crs(candidates=None)
```

Resume la evidencia sobre el sistema de coordenadas y compara candidatos.

- Sin `candidates`, usa los que se deducen de la leyenda y de la configuración (`GEOREF_REGION_HINT`, `GEOREF_DEFAULT_DATUM`).
- Devuelve la lectura de la leyenda (datum, zona, hemisferio, escala), la evidencia acumulada, el CRS actual y el tipo de etiquetas.
- Con al menos 4 puntos de control habilitados, ajusta un afín en cada candidato y los ordena por RMSE (`ranking`). Con menos, devuelve solo la lista de candidatos.
- **Limitación importante:** si las etiquetas son UTM, todas las zonas ajustan igual, porque los valores ya están en la proyección. En ese caso la respuesta lo advierte: la zona y el datum solo pueden salir de la leyenda o de otra evidencia del mapa. El ranking solo discrimina cuando las etiquetas son geográficas.

### 8.12 `set_crs`

```
set_crs(epsg, evidence)
```

Fija el CRS de la sesión y lo marca como confiable. Dos validaciones:

- El EPSG debe existir ("EPSG:N no existe").
- `evidence` es obligatoria: debe citar el texto del mapa que respalda el sistema y dónde está (p. ej. `leyenda inferior "Datum WGS 84, Zona 18 Sur"`). Queda registrada como `agente: <evidencia>` en `crs_evidence`.

Tras fijarlo, reajusta la transformación. Ejemplos de códigos en la propia documentación de la herramienta: 32718 (WGS 84 / UTM zona 18S) y 24878 (PSAD56 / UTM zona 18S).

### 8.13 `edit_gcps`

```
edit_gcps(action, gcp_id="", px=0.0, py=0.0, x="", y="", snap=True)
```

Consulta o modifica los puntos de control. Tras cada cambio se reajusta la transformación.

| `action` | Comportamiento |
|---|---|
| `list` | Hasta 80 puntos, ordenados por residuo descendente, con el total y los habilitados. |
| `disable` / `enable` | Cambia el estado de un punto por su ID (`V3H5`) o de **todos** los puntos de una línea por el ID de la línea (`V3`). |
| `add` | Agrega un punto manual (ID `M1`, `M2`, …) en `px`, `py` con coordenadas `x`, `y`. Con `snap=True` (por defecto) lo lleva a la esquina o cruce de trazos más cercano con `cv2.cornerSubPix` en un radio de 12 px; si el refinamiento se aleja más que eso, conserva el punto original. Devuelve el píxel final y cuánto se movió (`moved_px`). |

### 8.14 `fit_transform`

```
fit_transform(kind="auto")
```

Ajusta la transformación con los puntos habilitados. Si no hay puntos de origen `grid`, los reconstruye antes. Necesita al menos 3 puntos habilitados.

| `kind` | Modelo | Puntos mínimos | Uso |
|---|---|---|---|
| `auto` | Afín, salvo que otro modelo mejore con claridad | 3 | Por defecto. |
| `affine` | Afín | 3 | Escaneos planos. |
| `projective` | Homografía | 4 | Fotos con perspectiva. |
| `poly2` | Polinómica de grado 2 | 7 | Papel deformado. |
| `tps` | Thin-plate spline | 5 | Papel deformado. |

**Regla de `auto`:** parte de un afín y solo acepta un modelo más flexible si el afín falla de verdad y el otro mejora con claridad fuera de muestra, para no sobreajustar el ruido de las líneas.

| Candidato | Puntos mínimos | Se evalúa si el RMSE de validación cruzada del afín supera | Se acepta si su validación cruzada es menor que |
|---|---|---|---|
| `projective` | 8 | 1,0 px | 0,6 × la del afín |
| `poly2` | 20 | 1,5 px | 0,5 × la del anterior |

Devuelve `kind`, `epsg`, `n_gcps`, `rmse_px`, `max_px`, `loo_rmse_px` (validación cruzada dejando uno fuera), `rmse_m`, `pixel_size`, `units`, `rotation_deg`, `outliers` y los 6 puntos con mayor residuo (`worst`).

### 8.15 `validate`

Ejecuta los chequeos de coherencia (sección 11) y devuelve la lista, si pasaron y el estado sugerido.

### 8.16 `submit_result`

```
submit_result(status, epsg, confidence, notes)
```

Cierra el trabajo. Debe llamarse una sola vez, al final.

| Parámetro | Detalle |
|---|---|
| `status` | `ok`, `needs_review`, `no_coordinates` o `failed`. |
| `epsg` | Código EPSG, o 0 si no se pudo determinar. |
| `confidence` | De 0 a 1; se recorta a ese rango. |
| `notes` | Qué se verificó, qué evidencia respalda el CRS y qué queda en duda. Dos o tres frases. |

Si el estado es `ok` o `needs_review` y el `epsg` indicado es distinto de cero y **no coincide** con el fijado en la sesión, la herramienta devuelve un error pidiendo usar `set_crs` antes de cerrar. Si `epsg` es 0 se usa el de la sesión.

El resultado queda en `session.result` y el bucle termina.

## 9. Estado de la sesión (`MapSession`)

[session.py](../src/georef/session.py). Un mapa y todo lo que se sabe de él. Las herramientas leen y modifican este objeto; el modelo solo recibe resúmenes.

| Atributo | Contenido |
|---|---|
| `image`, `gray`, `width`, `height`, `dpi`, `sha` | La imagen (BGR), su versión en grises, tamaño, DPI leído del archivo y un hash corto. |
| `pyramid` | Pirámide de resoluciones para generar vistas sin remuestrear la imagen completa cada vez. |
| `ref_grid` | Rejilla de referencia (celdas A1, B1…). |
| `ocr`, `ocr_provider` | Índice OCR y proveedor (Document AI, o un simulado en las pruebas). |
| `frame` | Marco del mapa. |
| `lines` | Líneas de la cuadrícula (`GridLine`): ID, eje `v`/`h`, ecuación, soporte, tipo (`line`, `tick`, `cross`), si está en la retícula, valor y origen del valor (`label`, `fit`, `agent`). |
| `labels` | Etiquetas de coordenada (`CoordLabel`): texto, tipo, valor, eje, si está completa, margen, línea asociada y estado (`unassigned`, `ok`, `outlier`). |
| `axis_fits`, `label_kind` | Relación valor↔posición por eje y tipo de coordenadas (`utm` o `geo`). |
| `crs_epsg`, `crs_evidence`, `crs_confident` | CRS actual, evidencia acumulada y si se considera respaldado. |
| `gcps` | Puntos de control (`GCP`): píxel, coordenada, origen (`grid` o `agent`), habilitado y residuo. |
| `fit` | Resultado del último ajuste (`FitResult`). |
| `result` | Lo que el agente envió con `submit_result`. |
| `warnings` | Avisos acumulados durante el proceso. |

Convención de IDs de puntos de control: los de la grilla combinan la línea vertical y la horizontal (`V3H5`); los manuales del agente son `M1`, `M2`, etc.

## 10. Cierre del agente y decisión del estado final

El estado definitivo lo decide `decide_status` en [pipeline.py](../src/georef/pipeline.py), no el agente. **El agente propone; los chequeos tienen la última palabra sobre un "ok".**

### 10.1 Significado de los estados

| Estado | Significado |
|---|---|
| `ok` | Grilla comprobada, CRS con evidencia y chequeos superados. |
| `needs_review` | Hay georreferenciación, pero queda una duda concreta (está en las notas). |
| `no_coordinates` | El mapa no trae cuadrícula ni coordenadas legibles. |
| `failed` | No se pudo procesar. |

### 10.2 Reglas de decisión

**Sin ajuste de transformación** (`session.fit is None`):

| Situación | Estado |
|---|---|
| El agente dijo `no_coordinates` o `failed` | Ese estado. |
| No hay OCR configurado y el agente no cerró | `needs_review`, con la nota de que no se puede saber si el mapa trae coordenadas. |
| No hay etiquetas completas ni puntos de control | `no_coordinates`. |
| Hay coordenadas, pero no alcanzaron para ajustar | `needs_review`. |

**Con ajuste:** se corre `validate`. El estado es el del agente o, si no cerró, el sugerido por los chequeos. Si el estado es `ok` pero algún chequeo crítico falla, se **rebaja a `needs_review`** y se añade la nota "Rebajado a revisión por los chequeos: …".

**Confianza:** la del agente si la dio; si no, 0,9 cuando los chequeos pasan, 0,5 si hay CRS pero no pasan, 0,3 en el resto.

### 10.3 Ejemplo del efecto de los chequeos

Si el agente responde `ok` pero el mapa no declara zona ni datum, `crs_confident` es falso, el chequeo crítico "evidencia del CRS" falla y el mapa queda en `needs_review`. Hay una prueba para este caso (`test_checks_overrule_an_ok_the_evidence_does_not_support`).

## 11. Chequeos de coherencia

[validate.py](../src/georef/georef/validate.py). Es la última barrera contra un "ok" falso. El mapa puede ser `ok` solo si pasan todos los chequeos **críticos**.

| Chequeo | Crítico | Condición |
|---|---|---|
| Puntos de control | Sí | Al menos `ok_min_gcps` (6). |
| Error de ajuste | Sí | RMSE ≤ `ok_rmse_px` (2,0 px). |
| Validación cruzada | Sí | RMSE dejando uno fuera ≤ 1,5 × `ok_rmse_px` (3,0 px). Se omite si no hay puntos suficientes. |
| Orientación | Sí | La imagen no está en espejo respecto al terreno (norte y este coherentes). |
| Rotación | No | Menos de 10°. |
| Píxel cuadrado | No | Relación entre tamaños de píxel X/Y dentro de ±3 % (solo en unidades métricas). |
| Escala declarada | No | Si la leyenda dice una escala y se conoce el DPI, el tamaño de píxel medido coincide con el esperado (`escala × 0,0254 / DPI`) dentro de ±5 %. |
| Monotonía eje x | Sí | Los valores de las líneas verticales crecen hacia la derecha (con al menos 2 líneas con valor). |
| Monotonía eje y | Sí | Los valores de las líneas horizontales decrecen hacia abajo (con al menos 2 líneas con valor). |
| Área de uso del CRS | Sí | El centro del mapa cae dentro del área de uso del CRS (con 1° de margen). |
| Evidencia del CRS | Sí | `crs_confident` es verdadero. |

Si no hay ajuste, el único chequeo es "ajuste" y falla.

Dos de estos chequeos protegen contra los errores más peligrosos que menciona el prompt: la monotonía y la validación cruzada detectan numeraciones desplazadas o lecturas erradas, y la evidencia del CRS impide dar por bueno un datum o zona asumidos.

## 12. Manejo de errores y casos límite

### 12.1 Errores de la API durante el bucle

El runner captura estas excepciones del SDK y las registra como `report["error"]`, más un aviso en `warnings` ("El agente no pudo ejecutarse: …"). **El mapa no falla**: se procesa igual con el resultado del pre-análisis.

| Excepción | Mensaje |
|---|---|
| `NotFoundError` (404) | "El modelo … no está disponible para este proyecto: …" (p. ej. modelo no habilitado en Vertex). |
| `RateLimitError` | "Límite de uso alcanzado: …" |
| `APIStatusError` | "Error N de la API: …" |
| `APIConnectionError` | "Sin conexión con la API: …" |

### 12.2 Fin sin cierre

Si el agente termina o llega al límite de turnos sin llamar a `submit_result`, se registra un aviso. En ese caso `session.result` queda vacío y `decide_status` usa el estado sugerido por los chequeos. Una prueba verifica que el límite de turnos no pasa por éxito.

### 12.3 Rechazo del modelo

Con `stop_reason == "refusal"` el bucle sale y `session.result` se fija a `{"status": "failed", "confidence": 0.0, "notes": "El modelo rechazó la solicitud."}`.

### 12.4 Errores dentro de una herramienta

Cualquier excepción dentro de una herramienta se devuelve al modelo como `{"error": "..."}` en lugar de propagarse. Esto permite que el agente corrija parámetros (una celda inexistente, un `line_id` equivocado, un valor ilegible) y siga trabajando.

### 12.5 Lote

En `georef run`, una excepción procesando un mapa se registra como `failed` con la nota correspondiente y **no detiene el lote**.

### 12.6 Mapas sin coordenadas

Los mapas sin cuadrícula ni coordenadas legibles terminan con estado `no_coordinates` y no generan transformación, GeoTIFF ni puntos (solo `georef.json`, `qa_overlay.png` y `trace.jsonl`). La búsqueda de ubicación por topónimos queda fuera del alcance actual.

## 13. Trazabilidad

[tracing.py](../src/georef/tracing.py). LangSmith es **opcional**: sin `LANGSMITH_TRACING=true` y una `LANGSMITH_API_KEY`, los decoradores llaman a la función tal cual y el pipeline corre igual. Eso mantiene las pruebas sin red.

### 13.1 Árbol de la traza

Una traza por mapa:

```
georef_map                      raíz; metadatos: archivo, hash, modelo, effort, si usó agente
├─ prepass
│   ├─ ocr_tiles                mosaicos procesados y aciertos de caché
│   ├─ detect_grid
│   └─ auto_fit
├─ agent
│   ├─ ChatAnthropic            un run por turno: tokens, caché leída, razonamiento resumido
│   └─ view_image / ocr_region / assign_labels / ...   un run por herramienta
└─ export
```

### 13.2 Cómo se instrumenta

- `wrap_client` (sobre el cliente del modelo) registra cada llamada, con uso de tokens y caché.
- `@tracing.traced(nombre)` marca las etapas (`georef_map`, `prepass`, `agent`, `auto_fit`, `export`…).
- `tracing.run_tool` registra cada herramienta como run `tool`. Al modelo le llega el resultado completo; a la traza, el texto y las miniaturas.
- Al cerrar el mapa, `send_feedback` adjunta al run raíz estas métricas: `status_ok`, `confidence`, `rmse_px`, `rmse_m`, `n_gcps` y el valor `status`. Permite filtrar mapas dudosos en LangSmith.
- `georef.json` guarda el ID del run y su URL (bloque `trace`) para ir del entregable a su traza.
- `tracing.flush()` se llama al terminar el lote para no perder las últimas trazas. Si el proyecto aún no existía (primera traza), el feedback pendiente se envía ahí.

### 13.3 Imágenes en las trazas

Cada turno reenvía toda la conversación, así que sin filtro cada imagen se registraría una vez por turno. El cliente de LangSmith se crea con `hide_inputs=strip_images` y `hide_outputs=strip_images`, que reemplazan el base64 por una referencia corta (`[imagen omitida: N KB, id <hash>]`). La imagen queda registrada **una sola vez**, como miniatura JPEG (768 px de lado largo, calidad 70), en el run de la herramienta que la produjo.

| Variable | Efecto |
|---|---|
| `GEOREF_TRACE_IMAGES=on` | Miniaturas en el run de cada herramienta (por defecto). |
| `GEOREF_TRACE_IMAGES=off` | Solo metadatos: "N imagen(es) no registradas". |

**Privacidad:** las trazas sacan datos de GCP; recortes de mapas, texto OCR y coordenadas viajan a LangSmith. Para mapas sensibles usar `GEOREF_TRACE_IMAGES=off` o las variables `LANGSMITH_HIDE_INPUTS` / `LANGSMITH_HIDE_OUTPUTS`, y elegir la región correcta (`LANGSMITH_ENDPOINT` para EU).

Las trazas nunca deben tumbar el procesamiento: los errores al enviar feedback se ignoran.

## 14. Costo y uso de tokens

El informe `agent.usage` y `summary.csv` recogen `input_tokens`, `output_tokens` y `cache_read_tokens` por mapa.

| Aspecto | Detalle |
|---|---|
| Costo de una vista | Una de 1568 px ronda 2 000–3 000 tokens; una de 2576 px, unos 4 800. |
| Política de vistas | Navegación a 1568 px; lectura fina a resolución máxima solo cuando hace falta. |
| Caché de prompts | `cache_control: ephemeral` abarata los turnos posteriores. |
| Caché de OCR | El OCR se guarda en disco por hash para no pagarlo dos veces. |
| Orden de magnitud previsto | Menos de un dólar por mapa, **a medir en el piloto**. |

Tarifas de referencia de la API de Anthropic (por millón de tokens, entrada/salida): Opus 5.5 $4 / $20; Sonnet 5.5 $2 / $10; Fable 5.1 $10 / $50 (solo como escalamiento, con retención de datos de 30 días). Vertex AI tiene tarifa propia que hay que confirmar con Google.

## 15. Pruebas

[tests/test_agent_loop.py](../tests/test_agent_loop.py) y [tests/test_tools.py](../tests/test_tools.py). Corren sin ningún servicio externo: un transporte HTTP simulado (`FakeModel`) devuelve turnos guionizados, un OCR simulado (`FakeOcr`) lee desde el mapa sintético, y una sesión simulada de LangSmith solo anota lo que recibe.

Casos cubiertos por las pruebas del bucle:

| Prueba | Qué verifica |
|---|---|
| `test_agent_verifies_and_submits` | Flujo completo de 3 turnos. Estado `ok`, confianza 0,93, uso de tokens acumulado, ruta del modelo en Vertex, parámetros de la petición (`thinking`, `output_config`, `cache_control`, 14 herramientas, sin `tool_choice`), dos herramientas en un turno con su imagen cada una, y contenido de `trace.jsonl` sin base64. |
| `test_checks_overrule_an_ok_the_evidence_does_not_support` | Un `ok` sin evidencia de CRS se rebaja a `needs_review`. |
| `test_agent_can_declare_no_coordinates` | Un croquis sin cuadrícula termina en `no_coordinates` y sin transformación. |
| `test_model_not_enabled_falls_back_to_the_prepass` | Un 404 de Vertex no impide procesar el mapa con el pre-análisis. |
| `test_turn_limit_and_refusal_do_not_pass_as_success` | Un bucle sin fin se corta en `max_turns` y un rechazo no termina en `ok`. |
| `test_langsmith_trace_tree_without_full_images` | Estructura del árbol de trazas, un run por turno y por herramienta, ningún base64 largo en los turnos, feedback en la raíz. |
| `test_trace_images_can_be_turned_off` | Con `GEOREF_TRACE_IMAGES=off` no viaja ninguna imagen. |

Ejecutar:

```powershell
.venv\Scripts\python -m pytest
```

## 16. Cómo ejecutarlo

Instalación y variables de entorno: ver [README.md](../README.md).

```powershell
georef doctor                              # comprueba Claude en Vertex, Document AI, GDAL y LangSmith
georef run mapa.tif --out salida           # un mapa con el agente
georef run data\samples                    # una carpeta
georef run data\samples --no-agent         # solo el pre-análisis, sin modelo
georef run data\samples --workers 4        # en paralelo
georef run mapa.tif --model claude-sonnet-5-5 --effort medium
georef run mapa.tif --region-hint "17S,18S,19S" --default-datum PSAD56
georef eval                                # batería sintética
georef eval data\truth --agent             # mapas reales con su .points de referencia, usando el agente
```

`georef doctor` comprueba por separado cuatro cosas, y para cada fallo indica cómo resolverlo:

| Comprobación | Qué hace |
|---|---|
| GDAL / rasterio | Escribe y abre un GeoTIFF mínimo. |
| Claude | Pide al modelo que responda "ok" con `effort: low`. |
| Document AI | Lee con OCR una imagen sintética con "8 650 000 N". |
| LangSmith | Envía una traza de prueba y muestra su URL. |

### 16.1 Entregables por mapa

En `out/<mapa>/`:

| Archivo | Contenido |
|---|---|
| `<mapa>_georef.tif` | GeoTIFF con el norte arriba. Si la imagen ya está alineada (afín con rotación menor de 0,002°) no se remuestrea. |
| `<mapa>.points` | Puntos de control en el formato del georreferenciador de QGIS (con `sourceY` negativa). |
| `<mapa>.tfw` / `.jgw` / `.pgw` y `.prj` | World file de la imagen original (aproximación afín) y su proyección. |
| `georef.json` | Estado, confianza, notas, informe del agente, CRS y evidencia, transformación y su matriz afín, puntos de control, ajustes por eje, avisos, archivos escritos, segundos y datos de la traza. |
| `qa_overlay.png` | La grilla reconstruida dibujada sobre el mapa, con etiquetas, puntos y residuos. |
| `trace.jsonl` | Lo que hizo el agente, paso a paso, sin imágenes. |

Además, `out/summary.csv` con una fila por mapa: `map`, `status`, `confidence`, `epsg`, `crs`, `n_gcps`, `rmse_px`, `rmse_m`, `fit`, `agent_turns`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `seconds`, `trace_url`, `notes`.

Si la imagen es demasiado grande para remuestrearla (más de 32 000 px de lado), no se escribe el GeoTIFF y queda un aviso: se usan el world file y los puntos de control.

## 17. Limitaciones y trabajo pendiente

Según [README.md](../README.md) y [PLAN.md](../PLAN.md):

**Verificado en local, sin servicios externos**

- El pipeline determinista sobre mapas sintéticos UTM y geográficos (líneas completas, cruces o marcas en el marco), con error menor de 0,5 px.
- Las 14 herramientas, el bucle completo del agente y el árbol de trazas de LangSmith.
- La escritura y lectura de los entregables.

**Pendiente de verificar con servicios reales**

- El agente contra Claude en Vertex AI. Hoy `georef doctor` falla porque el modelo no está habilitado en Vertex.
- El OCR contra Document AI. Falta el ID del procesador.
- El envío de trazas a LangSmith. Falta la API key.
- El comportamiento con mapas reales: los umbrales de detección están ajustados solo con sintéticos.

**Fuera de la versión actual**

- Mapas sin coordenadas (toponimia y comparación con mapa base): hoy solo se detectan y se apartan.
- Visor de revisión.
- Fase 5: subir el set de referencia como *dataset* de LangSmith y comparar Opus 5.5 con Sonnet 5.5 como experimentos. `georef eval` ya mide el error en local.

**Riesgos propios del dominio** (y su tratamiento)

| Riesgo | Tratamiento |
|---|---|
| Zona UTM ambigua (los valores Este son iguales en todas las zonas) | Exigir evidencia (leyenda, doble rotulado geográfico) o usar `--region-hint`; sin evidencia, `needs_review`. |
| Datum ambiguo (PSAD56 frente a WGS84; desfase de cientos de metros) | Leerlo de la leyenda; si no aparece, `--default-datum` y marca de incertidumbre. |
| Etiquetas verticales, abreviadas o con superíndices | OCR con rotaciones, ajuste de progresión que ancla las abreviadas en las completas, zoom del agente. |
| Dos grillas superpuestas (UTM y geográfica) | Separar familias por ángulo, espaciado y tipo de etiqueta; usar una para valores y otra para comprobar. |
| Líneas que no son grilla (vías, curvas de nivel) | Filtro por espaciado regular y longitud; se marcan "fuera de retícula". |
| Retícula geográfica curva en un mapa proyectado | Usar intersecciones y ajustar en el CRS proyectado candidato. |
| Imágenes muy grandes | Mosaicos para OCR (límite de Document AI: 40 MP y 40 MB por imagen), pirámide para las vistas. |
| Un `ok` que en realidad está mal | Validación cruzada, chequeos de coherencia y umbrales conservadores; la duda va a `needs_review`. |
| Líneas falsas con valor | Una línea solo recibe valor si cae a menos de 2 px (o 0,6 % del paso) de un valor redondo, y se retira si todos sus puntos se desvían hacia el mismo lado tras el ajuste. |

**Decisión de arquitectura a reconsiderar:** LangGraph se reevaluaría si hiciera falta pausar un mapa para revisión humana y reanudarlo, o cambiar de proveedor de modelo. Las herramientas son funciones Python normales, así que la migración sería barata.

## 18. Mapa de archivos

```
src/georef/
  cli.py              comandos doctor, run, eval, synth
  config.py           Settings y lectura del .env
  session.py          MapSession, GridLine, CoordLabel, GCP, AxisFit, Frame
  tracing.py          LangSmith: decoradores, filtro de imágenes, feedback, flush
  pipeline.py         orquesta ingesta, pre-análisis, agente y exportación; decide_status
  agent/
    client.py         cliente Vertex o Anthropic directo, con trazas
    prompts.py        SYSTEM_PROMPT e initial_text
    tools.py          las 14 herramientas (build_tools)
    runner.py         bucle del agente y trace.jsonl
  imaging/            loader, pyramid, enhance, annotate (reglas, rejilla, overlay)
  ocr/                docai, tiling, index, cache
  grid/               frame, lines, ticks, intersections
  coords/             parse, labels, crs
  georef/             transform, validate, export
  eval/               synth, metrics, run_eval
tests/
  test_agent_loop.py  bucle, estados y trazas con modelo simulado
  test_tools.py       herramientas
  test_pipeline_synth.py, test_parse.py, test_ocr.py, test_export.py
```

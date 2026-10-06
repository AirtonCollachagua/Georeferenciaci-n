# Plan: agente de georreferenciación de imágenes de mapas

## Contexto

`c:\Georeferenciacion` es un proyecto nuevo: hoy solo contiene `credentials/` con una cuenta de servicio de GCP (proyecto `extrac-datos-geosys-production`). No hay código que reutilizar.

Se quiere un agente que reciba la imagen de un mapa (escaneo, foto o PDF) y la devuelva georreferenciada. La información para hacerlo está impresa en el propio mapa: una cuadrícula con coordenadas en los márgenes y una leyenda con el sistema de referencia. El agente debe leer esas coordenadas, reconstruir la grilla completa con el valor de cada línea, deducir el sistema de coordenadas y generar los puntos de control.

Decisiones ya tomadas contigo:

| Tema | Decisión |
|---|---|
| Lenguaje | Python 3.11+ |
| Modelo del agente | Claude Opus 5.5 (`claude-opus-5-5`) vía Vertex AI, en tu proyecto GCP |
| OCR | Google Document AI (asumo que "doc aim" es Document AI), procesador Enterprise Document OCR |
| Visión clásica | OpenCV |
| Geoespacial | rasterio (trae GDAL en Windows) + pyproj |
| Construcción del agente | SDK de Anthropic directo (`tool_runner`), sin LangChain ni LangGraph |
| Trazabilidad | LangSmith: una traza por mapa, con cada turno del modelo y cada herramienta |
| Forma de uso | CLI por lotes |
| Tipos de mapa | Principalmente UTM y geográficas; los mapas sin coordenadas se detectan y se apartan (fuera del MVP) |
| "Grillas con etiquetas" | Las dos cosas: reconstruir la grilla del mapa con sus valores, y una rejilla de referencia para que el agente se ubique |

## Principio de diseño: el modelo decide, el código mide

Un modelo de visión no ubica un píxel con la precisión que exige un punto de control, y un detector de líneas no sabe qué número es una coordenada ni qué datum dice la leyenda. Por eso el trabajo se reparte así:

- **OpenCV** entrega las posiciones en píxeles (líneas e intersecciones con precisión sub-píxel).
- **Document AI** entrega el texto con su caja y su confianza.
- **El agente** decide qué etiqueta corresponde a qué línea, qué sistema de coordenadas es, qué está mal leído, y cuándo el resultado no es confiable. Usa el zoom para comprobar, nunca para estimar a ojo la posición final de un punto.

Toda herramienta trabaja en píxeles de la imagen original (origen arriba a la izquierda). Cada imagen que ve el agente lleva reglas con esos píxeles en los bordes, así no tiene que hacer cuentas de escala.

## Flujo por mapa

`ingesta → pre-análisis determinista → agente (verifica y corrige) → exportación + control de calidad`

1. **Ingesta.** Carga TIFF/JPG/PNG o rasteriza el PDF a 300–400 DPI. Construye una pirámide de resoluciones. La imagen completa nunca se envía al modelo.
2. **Pre-análisis determinista** (sin LLM, barato y cacheado):
   - detecta el marco del mapa;
   - corre OCR por mosaicos sobre márgenes y leyenda, con rotaciones 0° y 90° para las etiquetas verticales;
   - detecta las líneas de la cuadrícula, o marcas y cruces cuando no hay líneas completas;
   - extrae candidatos a coordenada (UTM, grados-minutos-segundos, grados decimales);
   - propone una asociación etiqueta↔línea, un CRS y un primer ajuste.
3. **Agente.** Recibe una vista general con rejilla de referencia y el resumen del pre-análisis. Hace zoom en esquinas, márgenes y leyenda, repite OCR donde faltó, corrige asociaciones, fija el CRS con evidencia, revisa residuos y termina con `submit_result`.
4. **Exportación.** Escribe los entregables y un reporte de calidad.

Los mapas sin grilla ni etiquetas terminan con estado `no_coordinates` y van a una cola aparte.

## Herramientas del agente

Son herramientas propias ejecutadas en local (el estado pesado, imagen e índice OCR, vive en un objeto `MapSession` y no en el contexto del modelo).

| Grupo | Herramienta | Qué hace |
|---|---|---|
| Zoom | `view_image(region, max_side, enhance, ref_grid)` | Devuelve la vista general o un recorte ampliado. `region` acepta píxeles o celdas de la rejilla (`"C4"`, `"B2:C3"`). `enhance`: CLAHE, binarizado, nitidez, canal de color. `ref_grid` dibuja la rejilla de referencia con etiquetas. |
| OCR | `ocr_region(region, rotation, upscale)` | Document AI sobre un recorte; devuelve palabras con caja en píxeles globales y confianza, y las agrega al índice. |
| OCR | `search_text(pattern, region, kind)` | Consulta el índice OCR ya calculado: regex o tipo (`utm`, `dms`, `scale`, `datum`, `zone`). |
| Grilla | `detect_frame()` | Marco del mapa (cuadrilátero). |
| Grilla | `detect_grid(mode, params)` | Familias de líneas verticales y horizontales con ID. `mode`: `lines`, `ticks`, `crosses`. Se puede relanzar con otros parámetros. |
| Grilla | `view_grid(region, show)` | Dibuja la grilla completa con etiquetas: cada línea con su ID y su valor asignado, más intersecciones, puntos de control y vectores de residuo. |
| Coordenadas | `get_label_candidates(side)` | Etiquetas de coordenada ya interpretadas, agrupadas por margen. |
| Coordenadas | `assign_labels(assignments, auto)` | Asocia etiqueta↔línea, ajusta la progresión (intervalo constante), completa las líneas sin etiqueta y marca las lecturas incoherentes. |
| CRS | `rank_crs(candidates)` | Ajusta el mapa en cada CRS candidato y los ordena por residuo; valida con pyproj que el mapa caiga en el área de uso. |
| CRS | `set_crs(epsg, evidence)` | Fija el CRS y registra la evidencia (texto de leyenda, doble rotulado, pista regional). |
| Georef | `edit_gcps(action, ...)` | Lista, agrega o quita puntos de control. Al agregar, ajusta el punto a la esquina, cruz o marca más cercana. |
| Georef | `fit_transform(kind)` | Afín, proyectiva, polinómica o TPS. Devuelve RMSE en píxeles y metros, residuo por punto y validación cruzada dejando uno fuera. |
| Georef | `validate()` | Coherencia: tamaño de píxel frente a escala y DPI, rotación, isotropía, monotonía de la grilla. |
| Cierre | `submit_result(status, crs, confidence, notes)` | Resultado final con esquema estricto. `status`: `ok`, `needs_review`, `no_coordinates`, `failed`. |

## Modelo recomendado

**Agente principal: Claude Opus 5.5 (`claude-opus-5-5`).** Es el que mejor lee material visual denso (diagramas, planos técnicos) y el que más provecho saca de herramientas de recorte y zoom, que es exactamente este diseño. Acepta imágenes de hasta 2576 px en el lado largo y sus coordenadas corresponden 1:1 a los píxeles de la imagen que ve.

| Modelo | Precio entrada / salida por millón de tokens | Papel aquí |
|---|---|---|
| Claude Opus 5.5 | $4 / $20 | Agente principal (recomendado) |
| Claude Sonnet 5.5 | $2 / $10 | Alternativa para volumen: evaluarla contra el set de pruebas cuando el pipeline funcione |
| Claude Fable 5.1 | $10 / $50 | Solo como escalamiento para mapas que Opus no resuelva; exige retención de datos de 30 días |

Los precios son los de la API de Anthropic; Vertex AI tiene tarifa propia que hay que confirmar en la página de precios de Google.

Configuración prevista:

```python
from anthropic import AnthropicVertex
from langsmith.wrappers import wrap_anthropic

client = wrap_anthropic(
    AnthropicVertex(project_id="extrac-datos-geosys-production", region="global")
)
runner = client.beta.messages.tool_runner(
    model="claude-opus-5-5",
    max_tokens=16000,
    thinking={"type": "adaptive", "display": "summarized"},  # el resumen va a la traza
    output_config={"effort": "high"},   # el valor por defecto en Opus 5.5 es "medium"
    cache_control={"type": "ephemeral"},
    system=SYSTEM_PROMPT,
    tools=TOOLS,                         # funciones con @beta_tool
    messages=[...],
)
```

Puntos a respetar con este modelo:

- El razonamiento no se puede desactivar y no acepta `budget_tokens` ni `temperature`; la profundidad se regula solo con `effort`. Empezar en `high`, porque usa mejor las herramientas de imagen con esfuerzo alto, y medir si `medium` alcanza.
- No acepta `tool_choice` forzado. El cierre se pide por prompt y `submit_result` lleva `strict: true`.
- Las herramientas con `@beta_tool` pueden devolver bloques de imagen (verificado en el SDK), así que `view_image` y `view_grid` devuelven texto + imagen directamente.
- En Vertex AI no hay Batch API, Files API ni `fallbacks` del lado servidor. Nada de eso es necesario aquí; si un turno termina con `stop_reason: "refusal"`, el mapa se marca `failed`.
- Si `tool_runner` no funcionara sobre el cliente de Vertex (se comprueba en la fase 0), se usa el bucle manual con `client.messages.create`, que sí está soportado.

Costo: una vista de 1568 px ronda 2 000–3 000 tokens y una de 2576 px llega a unos 4 800. Por eso las vistas de navegación van a 1568 px y solo las de lectura fina a resolución máxima. Con caché de prompts, el orden de magnitud esperado es de menos de un dólar por mapa, a medir en el piloto. El OCR se cachea en disco por hash para no pagarlo dos veces.

## Trazabilidad con LangSmith

Cada mapa procesado genera una traza en LangSmith. Sirve para tres cosas: ver por qué el agente tomó cada decisión, medir costo y tiempo por mapa, y comparar configuraciones.

Árbol de la traza:

```
georef_map                      raíz; metadatos: archivo, hash, lote, modelo, effort
├─ ingest
├─ prepass
│   ├─ ocr_tiles                mosaicos procesados y aciertos de caché
│   ├─ detect_frame / detect_grid
│   └─ auto_fit
├─ agent
│   ├─ ChatAnthropic            un run por turno: tokens, caché leída, razonamiento resumido
│   ├─ view_image / ocr_region / assign_labels / ...   un run por herramienta
│   └─ submit_result
└─ export
```

Cómo se instrumenta:

- `wrap_anthropic` sobre el cliente registra cada llamada al modelo con su uso de tokens, incluidos los de caché.
- `@traceable` en la función por mapa (raíz), en cada etapa y en cada herramienta (`run_type="tool"`).
- Al cerrar el mapa se adjunta *feedback* al run raíz: `rmse_px`, `rmse_m`, `n_gcps`, `confidence` y `status`. Así se filtran en LangSmith los mapas dudosos.
- `georef.json` guarda el ID y la URL del run, para ir de un entregable a su traza.
- El CLI vacía la cola de trazas antes de salir, para no perder las últimas del lote.

Imágenes en las trazas. Cada turno reenvía toda la conversación, de modo que sin filtro cada imagen se registraría una vez por turno y la traza crecería sin control. El filtro `hide_inputs` del cliente de LangSmith reemplaza el base64 de los turnos del modelo por una referencia corta (hash, región, tamaño), y la imagen queda registrada una sola vez como miniatura en el run de la herramienta que la produjo. Una variable `GEOREF_TRACE_IMAGES=off` deja solo metadatos, sin ninguna imagen.

LangSmith es opcional en ejecución: sin `LANGSMITH_TRACING=true` los decoradores no hacen nada y el pipeline corre igual, lo que mantiene las pruebas sin red. `trace.jsonl` se conserva como copia local mínima.

Variables en `.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=georeferenciacion`, y `LANGSMITH_ENDPOINT` solo si la cuenta es de la región EU.

En la fase de evaluación, el set con referencia manual se sube como *dataset* de LangSmith y cada configuración (Opus 5.5 frente a Sonnet 5.5, `effort` alto frente a medio) corre como un experimento, con el error en metros y el acierto de CRS como evaluadores. Las comparaciones quedan lado a lado.

LangSmith frente a LangChain. No son alternativas: LangSmith es la plataforma de trazas y evaluación, y LangChain/LangGraph es un framework para construir el agente. Se usa LangSmith sin LangChain por tres razones. El agente es un solo bucle con herramientas propias, que el `tool_runner` resuelve con menos código. El SDK da acceso directo a `effort`, razonamiento adaptativo, caché de prompts e imágenes en resultados de herramientas, mientras que con LangChain cada uno depende de que su integración lo exponga. Y la trazabilidad es la misma con `wrap_anthropic` y `@traceable`. LangGraph se reconsidera si más adelante hace falta pausar un mapa para revisión humana y reanudarlo, o cambiar de proveedor de modelo; las herramientas son funciones Python normales, así que la migración sería barata.

Dos cosas se comprueban en la fase 0, porque la documentación de `wrap_anthropic` solo nombra el cliente `Anthropic` y los métodos `messages.create`, `messages.stream`, `beta.messages.create` y `beta.messages.parse`: que funcione sobre `AnthropicVertex`, y que capture los turnos que lanza `tool_runner`. Si alguna falla, cada turno se registra a mano dentro del bucle con un run de tipo `llm`.

## Estructura del proyecto

```
c:\Georeferenciacion\
  PLAN.md                  copia de este plan
  pyproject.toml  .env.example  .gitignore
  credentials/             ya existe; nunca se versiona
  data/samples/            mapas de prueba
  data/truth/              georreferenciación manual de referencia
  src/georef/
    cli.py  config.py  session.py  tracing.py
    imaging/   loader.py  pyramid.py  enhance.py  annotate.py
    ocr/       docai.py  tiling.py  index.py  cache.py
    grid/      frame.py  lines.py  ticks.py  intersections.py
    coords/    parse.py  labels.py  crs.py
    georef/    gcps.py  transform.py  validate.py  export.py
    agent/     client.py  tools.py  prompts.py  runner.py
    eval/      synth.py  metrics.py  run_eval.py
  tests/
```

`tracing.py` concentra la configuración de LangSmith: cliente, filtro de imágenes, metadatos comunes y envío de *feedback*.

Dependencias: `anthropic[vertex]`, `langsmith`, `google-cloud-documentai`, `opencv-python-headless`, `numpy`, `scikit-image`, `rasterio`, `pyproj`, `pymupdf`, `pillow`, `pydantic`, `typer`, `python-dotenv`, `pytest`.

Entregables por mapa en `out/<mapa>/`:

| Archivo | Contenido |
|---|---|
| `<mapa>_georef.tif` | GeoTIFF georreferenciado |
| `<mapa>.points`, `.tfw`, `.prj` | Puntos de control para QGIS y world file |
| `georef.json` | CRS, transformación, puntos, residuos, RMSE, confianza, estado, evidencia, tokens usados, ID y URL de la traza |
| `qa_overlay.png` | Grilla reconstruida con etiquetas y vectores de residuo |
| `trace.jsonl` | Copia local de las llamadas a herramientas, sin las imágenes |

Además, `out/summary.csv` con una fila por mapa.

## Fases de implementación

| Fase | Contenido | Se da por buena cuando |
|---|---|---|
| 0. Base | `git init` con `.gitignore` antes que nada, `pyproject.toml`, `.env`, copia de este plan a `PLAN.md`, `tracing.py`, comando `georef doctor` | `doctor` llama a Claude en Vertex, a Document AI, abre un GeoTIFF y deja una traza de prueba visible en LangSmith |
| 1. Imagen y OCR | Carga, pirámide, `view_image` con reglas y rejilla, OCR por mosaicos con caché | Se puede navegar un mapa real y buscar texto en su índice |
| 2. Pipeline determinista | Marco, grilla, parseo de etiquetas, asociación, CRS, ajuste; generador de mapas sintéticos | RMSE < 1 px en mapas sintéticos limpios, sin LLM |
| 3. Agente | Herramientas, prompt de sistema, bucle, trazas con el árbol completo, `submit_result` | Resuelve un mapa real de principio a fin y su traza muestra cada turno y cada herramienta |
| 4. Exportación y lote | Entregables, `qa_overlay.png`, `georef run <carpeta>`, `summary.csv`, *feedback* en la traza | Procesa una carpeta completa sin intervención |
| 5. Evaluación y ajuste | *Dataset* y experimentos en LangSmith, comparación Opus 5.5 / Sonnet 5.5, ajuste de `effort` y umbrales | Tasa de aciertos y costo por mapa medidos |
| 6. Posterior | Mapas sin coordenadas (topónimos + comparación con mapa base), visor de revisión | Fuera del MVP |

El prompt de sistema describe el objetivo, el principio "medir con herramientas", los criterios de cierre y cuándo declarar `needs_review`. No dicta un procedimiento paso a paso: con estos modelos, los prompts demasiado prescriptivos empeoran el resultado.

## Lo que necesito de ti antes de empezar

1. **Mapas de muestra** en `data/samples/`: 10 a 20, cubriendo UTM, geográficas y algunos sin coordenadas. Sin ellos solo se puede probar con sintéticos.
2. **Vertex AI**: API habilitada, Claude Opus 5.5 activado en Model Garden y rol `roles/aiplatform.user` para la cuenta de servicio.
3. **Document AI**: API habilitada, un procesador Enterprise Document OCR creado (necesito su ID y su ubicación, `us` o `eu`) y rol `roles/documentai.apiUser`.
4. **LangSmith**: una API key, la región de la cuenta (US o EU) y tu decisión sobre las imágenes: registrar miniaturas (recomendado para depurar) o solo metadatos.
5. **Referencia manual** de 5 a 10 de esos mapas (georreferenciados en QGIS), para medir el error real.

## Riesgos y tratamiento

| Riesgo | Tratamiento |
|---|---|
| Zona UTM ambigua: los valores Este son iguales en todas las zonas | Exigir evidencia (leyenda, doble rotulado geográfico) o usar `--region-hint`; sin evidencia, `needs_review` |
| Datum ambiguo (PSAD56 frente a WGS84): la geometría no lo distingue y el desfase es de cientos de metros | Leerlo de la leyenda; si no aparece, `--default-datum` y marca de incertidumbre en `georef.json` |
| Etiquetas verticales, abreviadas o con superíndices | OCR con rotaciones, ajuste de progresión que ancla las abreviadas en las completas, zoom del agente |
| Dos grillas superpuestas (UTM y geográfica) | Separar familias por ángulo, espaciado y tipo de etiqueta |
| Líneas que no son grilla (vías, curvas de nivel) | Filtro por espaciado regular y longitud |
| Retícula geográfica curva en un mapa proyectado | Usar intersecciones en lugar de rectas y ajustar en el CRS proyectado candidato |
| Imágenes muy grandes | Mosaicos para OCR (límite de Document AI: 40 MP y 40 MB por imagen), pirámide para las vistas |
| Resultado `ok` que en realidad está mal | Validación cruzada, chequeos de coherencia y umbrales conservadores; la duda va a `needs_review` |
| Las trazas sacan datos de GCP: recortes de mapas, texto OCR y coordenadas viajan a LangSmith | Elegir región, filtro de imágenes, `GEOREF_TRACE_IMAGES=off` o `LANGSMITH_HIDE_INPUTS` / `LANGSMITH_HIDE_OUTPUTS` para mapas sensibles |
| Clave privada en `credentials/` y API key de LangSmith | `.gitignore` desde el primer commit; se referencian por variables de entorno |

## Verificación

1. `georef doctor`: confirma acceso a Claude en Vertex, a Document AI, a GDAL y a LangSmith.
2. `pytest`: pruebas de los parsers de coordenadas, del ajuste de progresión y de las transformaciones, más el pipeline determinista sobre mapas sintéticos con verdad conocida. Corren sin LangSmith.
3. `georef run data/samples/<mapa>.tif --out out/`: abrir el GeoTIFF en QGIS sobre un mapa base y comprobar que calza; revisar `qa_overlay.png` y `georef.json`.
4. Abrir en LangSmith la URL que trae `georef.json`: el árbol debe mostrar las etapas, un run por turno del modelo con tokens y caché leída mayor que cero, un run por herramienta con su miniatura, y el *feedback* de calidad en la raíz. Ningún turno debe llevar base64 completo.
5. `georef eval data/truth/`: error en metros y píxeles contra la referencia manual, acierto de CRS, proporción de `ok` falsos, costo y tiempo por mapa, publicado como experimento en LangSmith.

## Notas de implementación (6 de octubre de 2026)

Lo que cambió o se resolvió al construir las fases 0 a 4:

- **`wrap_anthropic` no funciona tal cual sobre `AnthropicVertex`**: busca `client.completions`, que el cliente de Vertex no tiene. `tracing.wrap_client` añade ese atributo antes de envolver. Con eso sí captura los turnos que lanza `tool_runner`, así que no hizo falta registrar los turnos a mano.
- **Se añadió `src/georef/pipeline.py`**, que orquesta ingesta, pre-análisis, agente y exportación. No estaba en la estructura prevista.
- **El pre-análisis prueba los tres modos de cuadrícula** (líneas, cruces, marcas) y se queda con el primero que resuelve ambos ejes.
- **Dos salvaguardas nuevas contra líneas falsas**: una línea solo recibe valor si cae a menos de 2 px (o 0,6 % del paso) de un valor redondo, y se le retira si todos sus puntos se desvían hacia el mismo lado tras el ajuste.
- **El *feedback* de LangSmith necesita el ID del proyecto.** Si el proyecto aún no existe (primera traza), el *feedback* se envía al final del lote, en `tracing.flush()`.
- **`GEOREF_PROVIDER=anthropic`** permite usar la API de Anthropic en lugar de Vertex sin tocar código.

Pendiente respecto al plan:

- Fase 5: subir el set de referencia como *dataset* de LangSmith y comparar Opus 5.5 con Sonnet 5.5 como experimentos. Requiere los accesos y mapas reales. `georef eval` ya mide el error en local.
- Verificación con los servicios reales: hoy `georef doctor` falla en Claude (modelo no habilitado en Vertex), Document AI (falta el ID del procesador) y LangSmith (falta la API key).

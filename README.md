# Georeferenciación de mapas con un agente

Recibe la imagen de un mapa (escaneo, foto o PDF) y la devuelve georreferenciada, leyendo
la cuadrícula y las coordenadas impresas en el propio mapa. El diseño completo está en
[PLAN.md](PLAN.md).

El trabajo se reparte en tres: OpenCV mide las posiciones en píxeles, Document AI lee el
texto, y un agente (Claude Opus 5.5) decide qué significa cada cosa y comprueba el resultado.

## Instalación

Requiere Python 3.11 o superior.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env
```

Completar `.env`:

| Variable | Para qué |
|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Cuenta de servicio de GCP (Vertex AI y Document AI) |
| `DOCAI_PROCESSOR_ID`, `DOCAI_LOCATION` | Procesador Enterprise Document OCR |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Trazas en LangSmith (opcional) |
| `GEOREF_REGION_HINT`, `GEOREF_DEFAULT_DATUM` | Zona UTM y datum a asumir cuando el mapa no los dice (opcional) |

Comprobar el entorno:

```powershell
.venv\Scripts\georef doctor
```

## Uso

```powershell
georef run data\samples                 # todos los mapas de una carpeta
georef run mapa.tif --out salida        # un mapa
georef run data\samples --no-agent      # solo el pre-análisis determinista, sin modelo
georef run data\samples --workers 4     # en paralelo
georef eval                             # batería sintética: error contra verdad conocida
georef eval data\truth --agent          # mapas reales con su .points de referencia al lado
georef synth                            # escribe los mapas sintéticos para verlos
```

Por cada mapa se crea `out/<mapa>/` con:

| Archivo | Contenido |
|---|---|
| `<mapa>_georef.tif` | GeoTIFF con el norte arriba |
| `<mapa>.points` | Puntos de control para el georreferenciador de QGIS |
| `<mapa>.tfw` / `.jgw` / `.pgw` y `.prj` | World file de la imagen original y su proyección |
| `georef.json` | Estado, CRS y su evidencia, transformación, residuos, chequeos, tokens y enlace a la traza |
| `qa_overlay.png` | La grilla reconstruida dibujada sobre el mapa |
| `trace.jsonl` | Lo que hizo el agente, paso a paso |

`out/summary.csv` resume el lote. Los estados posibles:

| Estado | Significado |
|---|---|
| `ok` | Grilla comprobada, CRS con evidencia y chequeos superados |
| `needs_review` | Hay georreferenciación, pero queda una duda concreta (está en las notas) |
| `no_coordinates` | El mapa no trae cuadrícula ni coordenadas legibles |
| `failed` | No se pudo procesar |

Un `ok` del agente no basta: si los chequeos de coherencia no pasan, el mapa baja a `needs_review`.

## Pruebas

```powershell
.venv\Scripts\python -m pytest
```

No llaman a ningún servicio. Usan mapas sintéticos con verdad conocida, un OCR simulado y
un modelo y un LangSmith simulados para el bucle del agente y sus trazas.

## Estado

Verificado en local, sin servicios externos:

- El pipeline determinista sobre mapas sintéticos UTM y geográficos, con líneas completas,
  cruces o marcas en el marco: error menor de 0,5 px en todos los casos.
- Las 14 herramientas del agente, el bucle completo y el árbol de trazas de LangSmith.
- La escritura y lectura de los entregables.

Pendiente de verificar con los servicios reales (ver `georef doctor`):

- El agente contra Claude en Vertex AI.
- El OCR contra Document AI.
- El envío de trazas a LangSmith.
- El comportamiento con mapas reales: los umbrales de detección están ajustados solo con sintéticos.

Fuera de esta versión: mapas sin coordenadas (se detectan y se apartan) y la comparación de
modelos como experimentos de LangSmith.

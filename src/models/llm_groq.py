"""llm_groq.py — Variante (d) de Fase 3: clasificación con un LLM vía Groq.

Clasifica cada texto preguntándole a un modelo servido por Groq
(``openai/gpt-oss-20b`` por defecto) si su autor muestra señales de anorexia/TCA.
Es zero-shot: ``fit`` no entrena nada. ``predict_proba`` devuelve P(anorexia)
en [0, 1] a partir de la respuesta del modelo.

Configuración por ``.env`` (cargado con python-dotenv):
  * ``GROQ_API_KEY``  — clave de API (obligatoria).
  * ``GROQ_MODEL``    — modelo a usar (por defecto ``openai/gpt-oss-20b``).

Caché por texto: cada respuesta se guarda en disco con clave
``hash(modelo + versión-de-prompt + texto)``. Así, reintentos, reordenamientos o
una segunda variante que comparta textos NO vuelven a gastar llamadas de API
(importa por costo y por los rate limits de Groq).

Límite de tasa: las llamadas se espacian para respetar el plan de Groq
(``GROQ_RPM`` peticiones/min, por defecto 30 = una cada ~2 s). Ante un 429 se
respeta el ``Retry-After`` devuelto por la API. Como ``fit`` es no-op, sólo se
llama a la API sobre el conjunto de PRUEBA.

La dependencia ``groq`` se importa de forma diferida dentro de los métodos, igual
que torch/transformers en las demás variantes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from src.models.base import AnorexiaClassifier

# Versión del prompt: súbela si cambias el system/template para invalidar caché.
_PROMPT_VERSION = "v1"

_SYSTEM_PROMPT = (
    "Eres un clasificador clínico de salud mental. Analizas un texto de redes "
    "sociales en español y determinas si su autor muestra señales de anorexia u "
    "otro trastorno de la conducta alimentaria (TCA). Respondes ÚNICAMENTE con un "
    "objeto JSON válido, sin texto adicional."
)

_USER_TEMPLATE = (
    'Texto:\n"""\n{text}\n"""\n\n'
    "¿El autor muestra señales de anorexia/TCA? Responde EXACTAMENTE con este JSON:\n"
    '{{"p_anorexia": <número entre 0 y 1>, "etiqueta": "anorexia" | "control"}}\n'
    "donde p_anorexia es tu probabilidad estimada de que el autor tenga un TCA."
)

# Plantilla por LOTES: varios textos en una sola petición. El límite de Groq es
# por PETICIÓN, así que agrupar N textos multiplica el rendimiento dentro del plan
# (p. ej. 30 req/min × 10 textos = 300 textos/min en vez de 30).
_BATCH_TEMPLATE = (
    "Clasifica los siguientes {n} textos de redes sociales. Para CADA uno, estima "
    "la probabilidad (0 a 1) de que su autor muestre señales de anorexia/TCA.\n\n"
    "{items}\n\n"
    "Responde EXACTAMENTE con un arreglo JSON de {n} objetos, en el MISMO orden que "
    "los textos, sin texto adicional ni claves extra:\n"
    '[{{"p_anorexia": <0..1>}}, {{"p_anorexia": <0..1>}}, ...]'
)


class GroqLLMClassifier(AnorexiaClassifier):
    """Clasificación zero-shot mediante un LLM servido por Groq.

    Attributes:
        model: Identificador del modelo en Groq (open-weights de OpenAI).
        temperature: Temperatura de muestreo (0.0 = determinista).
        max_retries: Reintentos ante errores/rate-limit antes de abortar.
        cache_path: Archivo JSON de caché por texto.
        requests_per_minute: Tope de peticiones/min del plan (Groq gpt-oss-20b
            ≈ 30 en free tier). Override por env ``GROQ_RPM``.
        batch_size: Textos por petición. >1 multiplica el rendimiento dentro del
            límite de tasa (el tope de Groq es por petición). Override ``GROQ_BATCH``.
        prompt_version: Versión del prompt (parte de la clave de caché).
    """

    name = "llm_groq"

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 6,
        cache_path: str = "output/cache_llm_groq.json",
        requests_per_minute: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        # Cargar .env ANTES de leer las variables, para que GROQ_MODEL/GROQ_RPM/etc.
        # del archivo tengan efecto (si no, __init__ usaría sólo el shell/los defaults).
        from dotenv import load_dotenv

        load_dotenv()
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_path = cache_path
        self.requests_per_minute = requests_per_minute or int(os.environ.get("GROQ_RPM", "30"))
        self.batch_size = batch_size or int(os.environ.get("GROQ_BATCH", "10"))
        self.prompt_version = _PROMPT_VERSION

        # Intervalo mínimo entre llamadas para no exceder el plan.
        self._min_interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.0
        self._last_call = 0.0  # marca de tiempo monotónica de la última llamada
        self._client = None
        self._cache: dict[str, float] | None = None

    def config(self) -> dict:
        """Sólo lo que afecta la PREDICCIÓN entra en la huella de caché.

        ``cache_path``/``max_retries``/``request_pause`` no cambian la salida, así
        que se omiten para no invalidar la caché al ajustarlos.
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "prompt_version": self.prompt_version,
        }

    def fit(self, texts: list[str], labels: list[int]) -> "GroqLLMClassifier":
        # Zero-shot: el LLM no se entrena.
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Devuelve P(anorexia) por texto. Sólo llama a la API por los NO cacheados,
        agrupándolos en lotes para multiplicar el rendimiento dentro del rate limit.
        """
        client = self._get_client()
        cache = self._load_cache()

        normalized = [t if isinstance(t, str) else "" for t in texts]
        keys = [self._cache_key(t) for t in normalized]

        # Textos únicos que faltan en la caché (dedup por clave).
        pending: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key, text in zip(keys, normalized):
            if key not in cache and key not in seen:
                seen.add(key)
                pending.append((key, text))

        if pending:
            print(f"  [llm_groq] {len(pending)} textos nuevos · lote={self.batch_size} "
                  f"· ~{self.requests_per_minute} req/min")
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start:start + self.batch_size]
            results = self._classify_batch(client, [t for _, t in chunk])
            for (key, _), prob in zip(chunk, results):
                cache[key] = prob
            self._save_cache(cache)
            print(f"  [llm_groq] {min(start + len(chunk), len(pending))}/{len(pending)} "
                  f"clasificados; caché total {len(cache)}")

        return np.asarray([cache[key] for key in keys], dtype=float)

    # ------------------------------------------------------------------
    # Llamada al modelo
    # ------------------------------------------------------------------

    def _classify_one(self, client, text: str) -> float:
        """Una llamada al LLM → P(anorexia), con reintentos y backoff."""
        if not text.strip():
            return 0.0

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            # Acotamos el texto para no exceder el contexto en posts muy largos.
            {"role": "user", "content": _USER_TEMPLATE.format(text=text[:4000])},
        ]

        for attempt in range(self.max_retries):
            try:
                self._throttle()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content or ""
                return self._parse_prob(content)
            except Exception as exc:  # noqa: BLE001 — backoff genérico ante la API
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Groq falló tras {self.max_retries} intentos: {exc}"
                    ) from exc
                time.sleep(self._retry_after(exc, attempt))
        return 0.5  # inalcanzable; satisface al type checker

    def _classify_batch(self, client, batch: list[str]) -> list[float]:
        """Clasifica varios textos en UNA petición → lista de P(anorexia).

        Si el lote tiene un solo texto, usa la ruta individual. Si la respuesta no
        se puede alinear (conteo distinto, JSON inválido), degrada a llamadas
        individuales para no perder ni desalinear resultados.
        """
        if len(batch) == 1:
            return [self._classify_one(client, batch[0])]
        if all(not t.strip() for t in batch):
            return [0.0] * len(batch)

        items = "\n".join(f"[{i + 1}] {t[:1500]}" for i, t in enumerate(batch))
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _BATCH_TEMPLATE.format(n=len(batch), items=items)},
        ]

        for attempt in range(self.max_retries):
            try:
                self._throttle()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content or ""
                parsed = self._parse_batch(content, len(batch))
                if parsed is not None:
                    return parsed
                # Respuesta no alineable → degradar a individual (robusto).
                return [self._classify_one(client, t) for t in batch]
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    # Último recurso: individual (cada uno con sus propios reintentos).
                    return [self._classify_one(client, t) for t in batch]
                time.sleep(self._retry_after(exc, attempt))
        return [0.5] * len(batch)

    @staticmethod
    def _parse_batch(content: str, n: int) -> list[float] | None:
        """Parsea un arreglo JSON de ``n`` probabilidades; ``None`` si no alinea."""
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            return None
        try:
            arr = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(arr, list) or len(arr) != n:
            return None
        out: list[float] = []
        for item in arr:
            if isinstance(item, dict) and "p_anorexia" in item:
                try:
                    out.append(_clip01(float(item["p_anorexia"])))
                except (ValueError, TypeError):
                    return None
            elif isinstance(item, (int, float)):
                out.append(_clip01(float(item)))
            else:
                return None
        return out

    def _throttle(self) -> None:
        """Espacia las llamadas para respetar el límite de peticiones/min del plan."""
        if self._min_interval <= 0:
            return
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @staticmethod
    def _retry_after(exc: Exception, attempt: int) -> float:
        """Segundos a esperar tras un error: usa ``Retry-After`` (429) si está."""
        # 1) Header Retry-After del response (rate limit 429).
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers is not None:
            retry = headers.get("retry-after") or headers.get("Retry-After")
            if retry:
                try:
                    return float(retry) + 0.5
                except (ValueError, TypeError):
                    pass
        # 2) Mensaje del tipo "try again in 1.23s".
        match = re.search(r"try again in ([\d.]+)s", str(exc))
        if match:
            return float(match.group(1)) + 0.5
        # 3) Backoff exponencial acotado.
        return min(2.0 ** attempt, 30.0)

    @staticmethod
    def _parse_prob(content: str) -> float:
        """Extrae P(anorexia) de la respuesta del modelo de forma robusta."""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if "p_anorexia" in obj:
                    return _clip01(float(obj["p_anorexia"]))
                if str(obj.get("etiqueta", "")).lower().startswith("anor"):
                    return 0.9
                return 0.1
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # Plan B: el primer número del texto.
        num = re.search(r"\d*\.?\d+", content)
        if num:
            try:
                return _clip01(float(num.group(0)))
            except ValueError:
                pass
        return 0.5

    # ------------------------------------------------------------------
    # Cliente y caché
    # ------------------------------------------------------------------

    def _get_client(self):
        """Crea (una sola vez) el cliente de Groq leyendo la key desde ``.env``."""
        if self._client is not None:
            return self._client

        from dotenv import load_dotenv
        from groq import Groq

        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Falta GROQ_API_KEY. Crea un archivo .env (ver .env.example) con "
                "tu clave de Groq: GROQ_API_KEY=..."
            )
        self._client = Groq(api_key=api_key)
        return self._client

    def _cache_key(self, text: str) -> str:
        raw = f"{self.model}\x00{self.prompt_version}\x00{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, float]:
        if self._cache is not None:
            return self._cache
        path = Path(self.cache_path)
        if path.exists():
            try:
                self._cache = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._cache = {}
        else:
            self._cache = {}
        return self._cache

    def _save_cache(self, cache: dict[str, float]) -> None:
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))

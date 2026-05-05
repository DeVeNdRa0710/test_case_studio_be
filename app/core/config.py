import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(override=False)


def _get(name: str, default: str = "") -> str:
    """Read env var and strip surrounding whitespace / matching quotes."""
    raw = os.getenv(name)
    if raw is None:
        return default
    val = raw.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        val = val[1:-1]
    return val


def _get_int(name: str, default: int) -> int:
    v = _get(name)
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    v = _get(name)
    try:
        return float(v) if v else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_env: str
    app_host: str
    app_port: int
    log_level: str

    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    embeddings_provider: str

    gemini_api_key: str
    gemini_model: str
    gemini_embed_model: str

    openai_api_key: str
    openai_embed_model: str

    groq_api_key: str
    groq_model: str

    aws_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str
    bedrock_model_id: str

    pinecone_api_key: str
    pinecone_index: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_dimension: int

    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str

    chunk_size: int
    chunk_overlap: int
    top_k: int

    redis_url: str
    cache_embed_ttl: int
    cache_result_ttl: int
    cache_enabled: bool

    rate_limit_enabled: bool
    rate_limit_generate: str
    rate_limit_ingest: str
    rate_limit_vision: str

    max_upload_bytes: int
    max_image_bytes: int
    max_pdf_pages: int

    database_url: str

    job_ttl_seconds: int

    figma_token: str
    figma_api_base: str
    figma_http_timeout: float


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=_get("APP_NAME", "ERP Test Case Generator"),
        app_env=_get("APP_ENV", "development"),
        app_host=_get("APP_HOST", "0.0.0.0"),
        app_port=_get_int("APP_PORT", 8000),
        log_level=_get("LOG_LEVEL", "INFO"),

        llm_provider=_get("LLM_PROVIDER", "gemini").lower(),
        llm_model=_get("LLM_MODEL", "gemini-2.5-flash"),
        llm_temperature=_get_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_get_int("LLM_MAX_TOKENS", 4096),

        embeddings_provider=_get("EMBEDDINGS_PROVIDER", "gemini").lower(),

        gemini_api_key=_get("GEMINI_API_KEY"),
        gemini_model=_get("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_embed_model=_get("GEMINI_EMBED_MODEL", "text-embedding-004"),

        openai_api_key=_get("OPENAI_API_KEY"),
        openai_embed_model=_get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),

        groq_api_key=_get("GROQ_API_KEY"),
        groq_model=_get("GROQ_MODEL", "llama-3.3-70b-versatile"),

        aws_region=_get("AWS_REGION", "us-east-1"),
        aws_access_key_id=_get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=_get("AWS_SESSION_TOKEN"),
        bedrock_model_id=_get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),

        pinecone_api_key=_get("PINECONE_API_KEY"),
        pinecone_index=_get("PINECONE_INDEX", "erp-testgen"),
        pinecone_cloud=_get("PINECONE_CLOUD", "aws"),
        pinecone_region=_get("PINECONE_REGION", "us-east-1"),
        pinecone_dimension=_get_int("PINECONE_DIMENSION", 768),

        neo4j_uri=_get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=_get("NEO4J_USER", "neo4j"),
        neo4j_password=_get("NEO4J_PASSWORD", "password"),
        neo4j_database=_get("NEO4J_DATABASE", "neo4j"),

        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
        top_k=_get_int("TOP_K", 6),

        redis_url=_get("REDIS_URL"),
        cache_embed_ttl=_get_int("CACHE_EMBED_TTL", 3600),
        cache_result_ttl=_get_int("CACHE_RESULT_TTL", 900),
        cache_enabled=_get("CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"},

        rate_limit_enabled=_get("RATE_LIMIT_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
        rate_limit_generate=_get("RATE_LIMIT_GENERATE", "10/minute"),
        rate_limit_ingest=_get("RATE_LIMIT_INGEST", "30/minute"),
        rate_limit_vision=_get("RATE_LIMIT_VISION", "15/minute"),

        max_upload_bytes=_get_int("MAX_UPLOAD_BYTES", 20 * 1024 * 1024),
        max_image_bytes=_get_int("MAX_IMAGE_BYTES", 8 * 1024 * 1024),
        max_pdf_pages=_get_int("MAX_PDF_PAGES", 200),

        database_url=_get("DATABASE_URL", "sqlite+aiosqlite:///./data/app.db"),

        job_ttl_seconds=_get_int("JOB_TTL_SECONDS", 24 * 3600),

        figma_token=_get("FIGMA_TOKEN"),
        figma_api_base=_get("FIGMA_API_BASE", "https://api.figma.com"),
        figma_http_timeout=_get_float("FIGMA_HTTP_TIMEOUT", 20.0),
    )


settings = get_settings()

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from psycopg.conninfo import make_conninfo


@dataclass(frozen=True)
class Config:
    database_url: str = os.environ.get("DATABASE_URL", "")
    db_host: str = os.environ.get("DB_HOST", "")
    db_port: str = os.environ.get("DB_PORT", "")
    db_name: str = os.environ.get("DB_NAME", "")
    db_user: str = os.environ.get("DB_USER", "")
    db_password: str = os.environ.get("DB_PASSWORD", "")
    username: str = os.environ.get("APP_USERNAME", "")
    password: str = os.environ.get("APP_PASSWORD", "")
    session_secret: str = os.environ.get("SESSION_SECRET", "")
    app_origin: str = os.environ.get("APP_ORIGIN", "")
    data_dir: str = os.environ.get("DATA_DIR", "/app/data")
    auto_train_min_new_examples: int = int(os.environ.get("AUTO_TRAIN_MIN_NEW_EXAMPLES", "100"))
    cookie_secure: bool = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
    db_pool_max: int = int(os.environ.get("DB_POOL_MAX", "10"))

    @property
    def database_conninfo(self) -> str:
        if self.database_url:
            return self.database_url
        return make_conninfo(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        )

    def validate(self, auth_required: bool = True) -> None:
        database_configured = bool(self.database_url) or all(
            (self.db_host, self.db_port, self.db_name, self.db_user, self.db_password)
        )
        required = {"DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD": database_configured}
        if auth_required:
            required.update({
                "APP_USERNAME": self.username,
                "APP_PASSWORD": self.password,
                "SESSION_SECRET": self.session_secret,
                "APP_ORIGIN": self.app_origin,
            })
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise RuntimeError("Missing required environment settings: " + ", ".join(missing))
        if auth_required:
            if len(self.session_secret) < 32:
                raise RuntimeError("SESSION_SECRET must contain at least 32 characters")
            origin = urlsplit(self.app_origin)
            if origin.scheme not in {"http", "https"} or not origin.hostname or origin.path not in {"", "/"} or origin.query or origin.fragment or origin.username:
                raise RuntimeError("APP_ORIGIN must be an exact http(s) origin")
        if not 2 <= self.db_pool_max <= 100:
            raise RuntimeError("DB_POOL_MAX must be between 2 and 100")
        if not 1 <= self.auto_train_min_new_examples <= 1_000_000:
            raise RuntimeError("AUTO_TRAIN_MIN_NEW_EXAMPLES must be between 1 and 1000000")


config = Config()

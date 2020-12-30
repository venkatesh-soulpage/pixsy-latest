from starlette.config import Config
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates


config = Config(".env")

DEBUG = config("DEBUG", cast=bool, default=False)
SECRET = config("SECRET", cast=str)
KAFKA_HOST = config("KAFKA_HOST", cast=str)
HTTPS_ONLY = config("HTTPS_ONLY", cast=bool, default=False)

templates = Jinja2Templates(directory="server/templates")
statics = StaticFiles(directory="server/static")

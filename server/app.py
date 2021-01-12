import httpx

from starlette.applications import Starlette
from starlette.routing import Router, Route, Mount

from server import settings, endpoints


# fmt: off
routes = [
    Route("/", endpoints.upload_photo, name="dashboard", methods=["GET", "POST"]),
    Route("/predict", endpoints.infer, name="infer", methods=["GET", "POST"]),
    Route("/500", endpoints.error),
    Mount("/static", settings.statics, name="static"),
    
]

exception_handlers = {
    404: endpoints.not_found,
    500: endpoints.server_error,
}

app = Starlette(
    debug=settings.DEBUG,
    routes=routes,
    exception_handlers=exception_handlers,
)

def url_for(*args, **kwargs):
    return app.url_path_for(*args, **kwargs)

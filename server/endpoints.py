from starlette.exceptions import HTTPException
from starlette.responses import RedirectResponse, Response, JSONResponse
from starlette.requests import Request
import io
from PIL import Image
from skimage import io as skio


from server.settings import templates
from ml.feature_extractor import extract_features


async def error(request):
    """
    An example error. Switch the `debug` setting to see either tracebacks or 500 pages.
    """
    raise RuntimeError("Oh no")


async def not_found(request, exc):
    """
    Return an HTTP 404 page.
    """
    template = "404.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context, status_code=404)


async def server_error(request, exc):
    """
    Return an HTTP 500 page.
    """
    template = "500.html"
    context = {"request": request}
    return templates.TemplateResponse(template, context, status_code=500)


async def dashboard(request):
    return JSONResponse({"hello": "world"})


async def upload_photo(request):
    template = "upload-photo.html"
    context = {"request": request}
    if request.method == "POST":
        form = await request.form()
        photo = form["photo"].file
        # extract_features()

    return templates.TemplateResponse(template, context, status_code=200)


async def infer(request):
    request_data = await request.json()
    return JSONResponse(request_data)

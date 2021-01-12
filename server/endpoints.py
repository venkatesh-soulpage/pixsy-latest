from starlette.exceptions import HTTPException
from starlette.responses import RedirectResponse, Response, JSONResponse
from starlette.requests import Request
import io
import uuid
import logging
from PIL import Image
from skimage import io as skio


from server.settings import templates
from ml.feature_extractor import extract_features
from werkzeug.datastructures import ImmutableMultiDict
from server.kafkaservice import kafkaconsumer


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
        # format request body to list the files
        form = await request.form()
        form = form.__dict__["_list"]
        data = {"photo": [], "matches": []}
        for item in form:
            if item[0] == "photo[]":
                data["photo"].append(item[1])
            elif item[0] == "matches[]":
                data["matches"].append(item[1])
        if len(data.get("photo")) > 0:
            photo_id = extract_features(data["photo"], "photo")
        else:
            photo_id = None
        if len(data.get("matches")) > 0:
            matches_id = extract_features(data["matches"], "matches")
        else:
            matches_id = None
        return JSONResponse({"photo_id": photo_id, "matches_id": matches_id})

    return templates.TemplateResponse(template, context, status_code=200)


async def infer(request):
    template = "prediction.html"
    context = {"request": request}
    if request.method == "POST":
        request_data = await request.form()
        print(request_data)
        data = kafkaconsumer(
            request_data.get("photo_id"), request_data.get("matches_id")
        )
        return Response(data.to_html())

    return templates.TemplateResponse(template, context, status_code=200)

from fastapi import FastAPI
from fastapi import Request
from fastapi import File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from inference import inference


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, id=1):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


@app.post("/upload")
async def upload(fileupload: UploadFile = File(...)):
	if not fileupload.filename.endswith(".wav"):
		return {"message": "Only wav files are allowed"}
	try:
		content = fileupload.file.read()
		with open("static/" + fileupload.filename, "wb") as f: f.write(content)
		result = inference("static/" + fileupload.filename)
		print(result)
	except Exception:
		return {"message": "Error while uploading file"}
	finally:
		fileupload.file.close()
	return {"message": "ok", "intervals": result, "filename": fileupload.filename}
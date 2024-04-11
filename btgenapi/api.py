import uvicorn

from typing import List, Optional
from fastapi import Depends, FastAPI, Header, Query, Response, UploadFile, APIRouter, Depends
from fastapi.params import File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from btgenapi.args import args
from btgenapi.models import *
from btgenapi.api_utils import req_to_params, generate_async_output, generate_streaming_output, generate_image_result_output, api_key_auth
import btgenapi.file_utils as file_utils
from btgenapi.parameters import GenerationFinishReason, ImageGenerationResult
from btgenapi.task_queue import TaskType
from btgenapi.worker import worker_queue, process_top, blocking_get_task_result
from btgenapi.models_v2 import *
from btgenapi.img_utils import base64_to_stream, read_input_image
import requests

from modules.util import HWC3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from all sources
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all request headers
)

secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)
vps_ip = "204.12.253.115"

img_generate_responses = {
    "200": {
        "description": "PNG bytes if request's 'Accept' header is 'image/png', otherwise JSON",
        "content": {
            "application/json": {
                "example": [{
                    "base64": "...very long string...",
                    "seed": "1050625087",
                    "finish_reason": "SUCCESS"
                }]
            },
            "application/json async": {
                "example": {
                    "job_id": 1,
                    "job_type": "Text to Image"
                }
            },
            "image/png": {
                "example": "PNG bytes, what did you expect?"
            }
        }
    }
}


def get_task_type(req: Text2ImgRequest) -> TaskType:
    if isinstance(req, ImgUpscaleOrVaryRequest) or isinstance(req, ImgUpscaleOrVaryRequestJson):
        return TaskType.img_uov
    elif isinstance(req, ImgPromptRequest) or isinstance(req, ImgPromptRequestJson):
        return TaskType.img_prompt
    elif isinstance(req, ImgInpaintOrOutpaintRequest) or isinstance(req, ImgInpaintOrOutpaintRequestJson):
        return TaskType.img_inpaint_outpaint
    else:
        return TaskType.text_2_img


def call_worker(req: Text2ImgRequest, accept: str) -> Response | AsyncJobResponse | List[GeneratedImageResult]:
    if accept == 'image/png':
        streaming_output = True
        # image_number auto set to 1 in streaming mode
        req.image_number = 1
    else:
        streaming_output = False

    task_type = get_task_type(req)
    params = req_to_params(req)
    async_task = worker_queue.add_task(task_type, params, req.webhook_url)

    if async_task is None:
        # add to worker queue failed
        failure_results = [ImageGenerationResult(im=None, seed='', finish_reason=GenerationFinishReason.queue_is_full)]

        if streaming_output:
            return generate_streaming_output(failure_results)
        if req.async_process:
            return AsyncJobResponse(job_id='',
                                    job_type=get_task_type(req),
                                    job_stage=AsyncJobStage.error,
                                    job_progress=0,
                                    job_status=None,
                                    job_step_preview=None,
                                    job_result=failure_results)
        else:
            return generate_image_result_output(failure_results, False)

    if req.async_process:
        # return async response directly
        return generate_async_output(async_task)
    
    # blocking get generation result
    results = blocking_get_task_result(async_task.job_id)

    if streaming_output:
        return generate_streaming_output(results)
    else:
        return generate_image_result_output(results, req.require_base64)


def stop_worker():
    process_top()


@app.get("/")
def home():
    return Response(content='Swagger-UI to: <a href="/docs">/docs</a>', media_type="text/html")


@app.get("/ping", description="Returns a simple 'pong' response")
def ping():
    return Response(content='pong', media_type="text/html")



@secure_router.post("/v2/generation/text-to-image-with-ip", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def text_to_img_with_ip(rawreq: SimpleText2ImgRequestWithPrompt,
                        accept: str = Header(None),
                        accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query
    req = Text2ImgRequestWithPrompt()
    req.image_number = rawreq.image_number
    req.prompt = rawreq.prompt
    req.image_prompts = rawreq.image_prompts
    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for raw_img_prompt in rawreq.image_prompts:
        img_prompt = ImagePrompt(cn_img=None)
        img_prompt.cn_img = base64_to_stream(raw_img_prompt)
        image_prompts_files.append(img_prompt)

    # while len(image_prompts_files) <= 4:
    #     image_prompts_files.append(default_image_promt)

    req.image_prompts = image_prompts_files

    result = call_worker(req, "application/json")
    return result


def generate_work(rawreq: SimpleText2ImgRequestWithPrompt): 
    req = Text2ImgRequestWithPrompt()
    req.image_number = rawreq.image_number
    req.prompt = rawreq.prompt
    req.image_prompts = rawreq.image_prompts
    default_image_promt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for raw_img_prompt in rawreq.image_prompts:
        img_prompt = ImagePrompt(cn_img=None)
        img_prompt.cn_img = base64_to_stream(raw_img_prompt)
        image_prompts_files.append(img_prompt)

    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_promt)

    req.image_prompts = image_prompts_files

    result = call_worker(req, "application/json")
    return result


@secure_router.post("/v2/generation/text-to-image-with-ip-multi", response_model=List[GeneratedImageResult] | AsyncJobResponse, responses=img_generate_responses)
def text_to_img_with_ip(req: Text2ImgRequestWithPromptMulti,
                        accept: str = Header(None),
                        accept_query: str | None = Query(None, alias='accept', description="Parameter to overvide 'Accept' header, 'image/png' for output bytes")):
    gToken = req.token
  
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query
    result = []
    callback_payload_images = []
    for text_prompt in req.text_prompts:
        req.prompt = text_prompt
        tmp = generate_work(req)
        for item_result in tmp:
            result.append(item_result)
            result_url = item_result.url
            remote_url = result_url.replace("127.0.0.1", vps_ip)
            item_result.url = remote_url
            callback_payload_images.append({"url": remote_url, "prompt": text_prompt})

    try:
        # Define the GraphQL query and variables as a dictionary
        graphql_request = {
            "query": "mutation UpdateImagesGeneration($data: ImageGenerationInput!) { updateImagesGeneration(data: $data) { status }}",
            "variables": {
                "data": {
                    "images":callback_payload_images
                }
            }
        }

        # Define the headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + gToken,
            "Cookie": "jgb_cs=s%3A96Q5_rfHS3EaRCEV6iKlsX7u_zm4naZD.yKB%2BJ35mmaGGryviAAagXeCrvkyAC9K4rCLjc4Xzd8c"
        }

        # Define the GraphQL API endpoint
        url = "https://stage-graphql.beautifultechnologies.app/"

        # Send the HTTP request using the `requests` library
        response = requests.post(url, json=graphql_request, headers=headers)

        # Print the response content and status code
        print(response.content)
        print(response.status_code)
    except Exception as e:

        print(e)

    return result


app.mount("/files", StaticFiles(directory=file_utils.output_dir), name="files")

app.include_router(secure_router)

def start_app(args):
    file_utils.static_serve_base_url = args.base_url + "/files/"
    uvicorn.run("btgenapi.api:app", host=args.host,
                port=args.port, log_level=args.log_level)

from solution.utils import process_image
from solution.solver import TransferSolver
from starlette.responses import StreamingResponse
import io
from fastapi import FastAPI, UploadFile
import torchvision

app = FastAPI()
transfersolver = TransferSolver()


@app.post("/transfer/")
async def transfer_image(content_image: UploadFile, style_image: UploadFile):
    '''
    func: transfer image follow by style_image
    content_image: image for transfering
    style_image: used to extract style, context for  transfer to content_image 
    output: StreamingResponse()
    '''
    content_image = await content_image.read()
    style_image = await style_image.read()

    # process image
    processed_content_image = process_image(content_image)
    processed_style_image = process_image(style_image)

    # transfer image by style
    transfered_image = transfersolver.solve(processed_content_image, processed_style_image, steps = 100)

    # process output (a tensor) to image 
    image = torchvision.transforms.ToPILImage()(transfered_image.squeeze())
    image_bytes = io.BytesIO()
    image.save(image_bytes, "PNG")
    image_bytes.seek(0)
    return StreamingResponse(content=image_bytes, media_type="image/png")
import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from base64 import b64encode
import io
import dill
import skimage
from skimage import measure


export_file_url = './' 
export_file_name1 = 'gan_model.pkl'
export_file_name2 = 'upsampler_model_2.pkl'

classes = ['images','gen_images']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

class ResModel(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.m(x) * self.res_scale
        return x

def res_block(nf):
    return ResModel(
        [conv(nf, nf), nn.BatchNorm2d(nf), conv(nf, nf), conv(nf, nf, actn=False)],
        0.1)

def upsample(ni, nf, sc):
    layers = []
    for i in range(int(math.log(sc,2))):
        layers += [conv(ni, nf*4), nn.PixelShuffle(2)]
        return nn.Sequential(*layers)

class SRResnet(nn.Module):
    def __init__(self, nf, scale):
        super().__init__()
        features = [conv(3, 32)]
        for i in range(4): features.append(res_block(32))
        features += [conv(32,32), upsample(32, 32, scale), nn.BatchNorm2d(32), conv(32, 3, actn=False)]
        self.features = nn.Sequential(*features)
        
    def forward(self, x): return self.features(x)


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name1)
    await download_file(export_file_url, path / export_file_name2)
    try:
        learn1 = load_learner(path, export_file_name1)
        learn2 = load_learner(path, export_file_name2)
        return [learn1, learn2]
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/enhance', methods=['POST'])
async def enhance(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    channel,height,width = img.shape
    img = img.resize((3,512,512))
    # img.save('./orig_img.jpeg')
    prediction1 = learn[0].predict(img)[0]
    prediction2 = learn[1].predict(prediction1)[0]
    

    prediction2.save('./res_img.jpeg')

    with open('./res_img.jpeg', "rb") as imageFile:
        res = b64encode(imageFile.read())
    res = res.decode('ascii')
    return JSONResponse({'result': str(res)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
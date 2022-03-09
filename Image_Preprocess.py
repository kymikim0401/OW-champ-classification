from glob import glob
from PIL import Image
import os
from tqdm import tqdm

fpaths = glob('heroes/*/*')
for fp in tqdm(fpaths):
    ad = fp.split('\\')
    ad[0] = 'dataset'
    result_path = os.path.join(*ad)
    
    image = Image.open(fp).convert('RGB')
    re = image.resize((256,256))
    try:
        re.save(result_path)
    except Exception:
        print(fp)
    
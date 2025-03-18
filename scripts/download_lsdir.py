from PIL import Image
from datasets import load_dataset, config


ds = load_dataset("danjacobellis/LSDIR", cache_dir="/projects/ydlin/lsdir")
print(ds['train'].column_names)
print(type(ds['train'][0]['image']))
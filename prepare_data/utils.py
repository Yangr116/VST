import io
import random
from PIL import Image
from loguru import logger
import datasets as hf_datasets


def save_to_parquet(data, parquet_path):
    dataset = hf_datasets.Dataset.from_list(data)
    dataset.to_parquet(parquet_path)
    print(f"Saved parquet with {len(data)} samples to {parquet_path}")


def encode_image_to_bytes(img):
    with io.BytesIO() as buffer:
        img.save(buffer, format='PNG')
        return buffer.getvalue()


def resize_image(image: Image.Image):
    orig_width, orig_height = image.size

    new_width, new_height = orig_width, orig_height
    if orig_width < 28 or orig_height < 28:
        scale = max(56.0 / orig_width, 56.0 / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
    
    if 100.0 < new_width / new_height or new_width / new_height < 1 / 100.0:
        logger.error(f"extream scale ratio {new_width / new_height}")
        return None

    if (new_width, new_height) != (orig_width, orig_height):
        resized_img = image.resize((new_width, new_height))
    else:
        resized_img = image

    meta_info = {
        'resized_width': new_width if new_width != orig_width else -1,
        'resized_height': new_height if new_height != orig_height else -1,
        'width': orig_width,
        'height': orig_height
    }
    return resized_img, meta_info


def split_into_chunks(data, chunk_size=1000):
    """将数据分割成块，不足chunk_size的与上一个块合并"""
    chunks = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunks.append(chunk)
    
    # 如果最后一个chunk小于chunk_size且不是第一个chunk，则与前一个合并
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)
    
    return chunks


def check_image_size(image: Image.Image):
    orig_width, orig_height = image.size

    # 计算调整尺寸
    new_width, new_height = orig_width, orig_height
    if orig_width < 28 or orig_height < 28:
        scale = max(56.0 / orig_width, 56.0 / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
    
    if 100.0 < new_width / new_height or new_width / new_height < 1 / 100.0:
        logger.error(f"extream scale ratio {new_width / new_height}")
        return None

    # 执行尺寸调整
    if (new_width, new_height) != (orig_width, orig_height):
        resized_img = image.resize((new_width, new_height))
    else:
        resized_img = image

    return resized_img


def random_resize_image_pil_list(image_pil_list_save):
    image_wh = image_pil_list_save[0].size
    scale_factor = random.choice([0.5, 1.0, 1.5, 2])
    image_wh_new = [int(wh // scale_factor) for wh in image_wh]

    image_pil_list_save = [image_pil.resize(image_wh_new) for image_pil in image_pil_list_save]
    return image_pil_list_save

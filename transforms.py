import math
import random
import torch

from torchvision import transforms
from PIL import Image, ImageOps

def _zoom(base, annotated, zoom_ratio=0.85):
  base_width = base.width
  base_height = base.height

  width = math.ceil(base_width * zoom_ratio)
  height = math.ceil(base_height * zoom_ratio)
  h = random.randint(0, base_width - width)
  v = random.randint(0, base_height - height)

  base_zoomed = base.crop((h, v, h+width, v+height))
  annotated_zoomed = annotated.crop((h, v, h+width, v+height))

  return base_zoomed.resize((base_width, base_height)), annotated_zoomed.resize((base_width, base_height))

def _translate_and_rotate(base, annotated, translate=True, rotate=True, background=None, transformations = None):
  h = random.randint(0, base.width // 2)
  v = random.randint(0, base.height // 2)

  rotation = random.randint(0, 360)

  hc = random.randint(0, (base.width - h) // 2)
  vc = random.randint(0, (base.height - v) // 2)

  def sub(base, background):
    base_translate = base.copy()
    base_translate.putalpha(256)
    if translate:
      base_translate = base_translate.transform(base.size, Image.AFFINE, (1, 0, h, 0, 1, v))
    if rotate:
      base_translate = base_translate.rotate(rotation, Image.NEAREST, expand=1)

    base_copy = base.copy() if not background else background.copy()
    base_copy.paste(base_translate, (hc, vc), base_translate)
    return base_copy

  if transformations:
    base, annotated = transformations(base, annotated)

  return sub(base, background), sub(annotated, background)

def _HSV(base, annotation, hue=True, saturation=True):
  hue = random.randint(0, 360)
  saturation = random.uniform(0.5, 1)

  HSV = base.convert('HSV')
  H, S, V = HSV.split()
  H = H.point(lambda p: p + hue)
  S = S.point(lambda p: int(p * saturation))

  HSVr = Image.merge('HSV', (H, S, V))
  RGBr = HSVr.convert('RGB')

  return RGBr, annotation

def _apply_transformations(base, annotation, de_zoom=True, de_mirror=True, de_HSV=True, de_tar=True):
  choices = random.choices([True, False], k=7)

  ap_zoom = choices[0] and de_zoom
  ap_mirror = choices[1] and de_mirror
  ap_HSV = choices[2] and de_HSV
  ap_tar = choices[3] and de_tar

  ap_zoom_tar = choices[4] and de_zoom
  ap_mirror_tar = choices[5] and de_mirror
  ap_HSV_tar = choices[6] and de_HSV

  def transformations(base, annotation):
    if ap_zoom_tar:
      base, annotation = _zoom(base, annotation)
    if ap_mirror_tar:
      base, annotation = _mirror(base, annotation)
    if ap_HSV_tar:
      base, annotation = _HSV(base, annotation)
    return base, annotation

  if ap_zoom:
    base, annotation = _zoom(base, annotation)
  if ap_mirror:
    base, annotation = _mirror(base, annotation)
  if ap_HSV:
    base, annotation = _HSV(base, annotation)
  if ap_tar:
    base, annotation = _translate_and_rotate(base, annotation, transformations=transformations)

  return base, annotation

def _mirror(base, annotated):
  return ImageOps.mirror(base), ImageOps.mirror(annotated)


class Transforms:
  def __init__(self, 
               zoom_ratio=0.85, 
               translate_zoom_ratio=0.85, 
               hue=True, 
               saturation=True, 
               de_zoom=True, 
               de_mirror=True, 
               de_HSV=True, 
               de_tar=True):
    self.zoom_ratio = zoom_ratio
    self.translate_zoom_ratio = translate_zoom_ratio
    self.hue = hue
    self.saturation = saturation
    self.de_zoom = de_zoom
    self.de_mirror = de_mirror
    self.de_HSV = de_HSV
    self.de_tar = de_tar

  def zoom(self, x):
    return _zoom(x[0], x[1], self.zoom_ratio)
  
  def apply_transformations(self, x):
    return _apply_transformations(x[0], x[1], self.de_zoom, self.de_mirror, self.de_HSV, self.de_tar)
  
  def to_tensor(self, x):
    return (torch.Tensor(x[0]), torch.Tensor(x[1]))
  
  

  
def zoom(zoom_ratio=0.85):
  def wrapper(x):
    return _zoom(x[0], x[1], zoom_ratio)
  return transforms.Lambda(wrapper)
  
def translate_and_rotate(zoom_ratio=0.85):
  def wrapper(x):
    return _translate_and_rotate(x[0], x[1], zoom_ratio)
  return transforms.Lambda(wrapper)

def HSV(hue=True, saturation=True):
  def wrapper(x):
    return _HSV(x[0], x[1], hue, saturation)
  return transforms.Lambda(wrapper)

def mirror():
  def wrapper(x):
    return _mirror(x[0], x[1])
  return transforms.Lambda(wrapper)

def apply_transformations(de_zoom=True, de_mirror=True, de_HSV=True, de_tar=True):
  def wrapper(x):
    return _apply_transformations(x[0], x[1], de_zoom, de_mirror, de_HSV, de_tar)
  return transforms.Lambda(wrapper)



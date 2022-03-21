# SPADE (It is easy to train!!)

`I made the codes to aim at synthesizing face images`

## Installation

* PyTorch
* Python
* tqdm
* pillow
* matplotlib
* numpy

## Configuration

* 1 gpu (GeForce RTX 3080 ti)
* batch size is 1

## Dataset

To train quickly, only 10,000 images were used. 

* Image : [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
* Segmentation map : [Merging masks and colorizing them](https://github.com/ji-in/CelebAMask-HQ_to_colorImage)

Something to keep in mind : It requires only `png` image. [convert jpg image to png image.py](https://github.com/ji-in/SPADE-torch-simple/blob/main/convert_jpg_to_png.py)

## Training

```python
python main.py
```

## Test

```
python test.py
```

* [pretrained](https://drive.google.com/file/d/156D5fdVyjrqAEfTMugTtKiiXZ1sLJBy-/view?usp=sharing)
* segmap_label.txt : You can use the same file used for train.

## Result

<img src="https://user-images.githubusercontent.com/40796092/159211114-d797bf7b-8e53-42d8-bd48-25bfa797d1a2.png" alt="image" style="zoom: 100%;" />

## References

https://github.com/NVlabs/SPADE

https://github.com/taki0112/SPADE-Tensorflow

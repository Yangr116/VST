
### Dependency
```
pip install pycocotools pytorch3d loguru
```

### Useage

```python
from evaluate import evaluate, create_dummy_prediction

gtfile = 'annotations/sunrgbd_val_20250118.json'

# create a dummy prediction file
create_dummy_prediction(gtfile=gtfile, savefile='./temp.json')

# evaluate
evaluate(gtfile, './temp.json', './output/evaluate')
```

A sample in the dummy prediction file:
```
{'uid': 'c270451c-b315-444d-bb24-3bc7e896c814', 'answer': [[-1.07, -0.31, 1.89, 0.71, 2.1, 1.56, 7.14, 5.09, 12.3]]}
```

You will get the results:
```
mode=3D  Average Precision  (AP) @[ IoU=0.05:0.50 | depth=   all | maxDets=100 ] = 1.000
mode=3D  Average Precision  (AP) @[ IoU=0.15      | depth=   all | maxDets=100 ] = 1.000
mode=3D  Average Precision  (AP) @[ IoU=0.25      | depth=   all | maxDets=100 ] = 1.000
mode=3D  Average Precision  (AP) @[ IoU=0.50      | depth=   all | maxDets=100 ] = 1.000
mode=3D  Average Precision  (AP) @[ IoU=0.05:0.50 | depth=  near | maxDets=100 ] = 1.000
mode=3D  Average Precision  (AP) @[ IoU=0.05:0.50 | depth=medium | maxDets=100 ] = -1.000
mode=3D  Average Precision  (AP) @[ IoU=0.05:0.50 | depth=   far | maxDets=100 ] = -1.000
mode=3D  Average Recall     (AR) @[ IoU=0.05:0.50 | depth=   all | maxDets=  1 ] = 0.913
mode=3D  Average Recall     (AR) @[ IoU=0.05:0.50 | depth=   all | maxDets= 10 ] = 1.000
mode=3D  Average Recall     (AR) @[ IoU=0.05:0.50 | depth=   all | maxDets=100 ] = 1.000
mode=3D  Average Recall     (AR) @[ IoU=0.05:0.50 | depth=  near | maxDets=100 ] = 1.000
mode=3D  Average Recall     (AR) @[ IoU=0.05:0.50 | depth=medium | maxDets=100 ] = -1.000
mode=3D  Average Recall     (AR) @[ IoU=0.05:0.50 | depth=   far | maxDets=100 ] = -1.000
```

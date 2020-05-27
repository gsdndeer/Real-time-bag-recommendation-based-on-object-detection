# Real-time-bag-recommendation-based-on-object-detection
This system uses YOLOv3 to detect the bag and recommends the similar bags.

<img src="https://github.com/gsdndeer/Real-time-bag-recommendation-based-on-object-detection/blob/master/figures/demo.gif">

The work structure :
```
Real-time-bag-recommendation-based-on-object-detection
      |
       --- yolo3
      |
       --- model_data
      |
       --- figures
      |
       --- font
      |
       --- image
      |
       --- system.py
      |
       --- yolo_multiple_output.py
      |
       --- yolo_detector.py
```
## Usage

1. Clone the repository
```
git clone https://github.com/gsdndeer/Real-time-bag-recommendation-based-on-object-detection.git
```

2. Add a folder named "image" in "Real-time-bacg-recommendation-based-on-object-detection"

3. Run ```python system.py```


## Acknowledgement

1. [keras-yolo3](https://github.com/qqwweee/keras-yolo3)

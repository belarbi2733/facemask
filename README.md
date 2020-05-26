How to use:

1. Install darknet: https://pjreddie.com/darknet/yolo/

Once installed you can use the weights with some image:

2. Download weights from [here](https://drive.google.com/open?id=1VdAufws4H2xj644DkR-AOcqBMKVTDC8u)

2. Put mask.data, classes.names and facemask.weights in same folder

3. ./darknet detector test  mask.data yolov3mod.cfg facemask.weights something.jpg 

4 .Check for predictions.jpg


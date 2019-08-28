# FacialExpressionRecognition
This project includes face detection, face alignment and facial expression recognition.

# Facial Expression Recognition
![image](https://github.com/lusihua/FacialExpressionRecognition/blob/master/images/results/demo.png)
### Prepare

* pytorch>0.4
* python>2.7
* h5py
* sklearn
* cv2

### Prepare  Datasets

1. Prepare expresion images put in **/datasets**. If you only have three categories, delete the extra folders. This images should be saved gray image and 48*48 or uniform pixels.
    * Batch processing image that reshape and generating gray scale image can running **reshape.py** in **/script**
2. Running **prepocess_CK+.py** in **/script** , Generating CK_data.h5 formatted data sets.

### Train model 

```
python train.py --model VGG19 --bs 128 --lr 0.01 --fold 1
```

### Test whole folder accuracy

* Test the accuracy of the whole folder, folders are named by category

```
python test.py
```

### Video demo, --video "your camer number"
* If you do it with a camera,run

```
python video_demo.py --video 0
```
* If detect the video file, run
```
python video_demo.py --video "your video path"
```

### Visualize result
`For a test image by a test model,Output the highest score categories and scores of each category`

* Thr pre-trained model can find in  **/model/Resnet50/1/pre_model.t7**
* running **visualize.py**, and the result saved in **/images/results/**

```
python visualize.py
```


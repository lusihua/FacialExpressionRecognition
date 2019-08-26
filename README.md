# FacialExpressionRecognition
This project includes face detection, face alignment and facial expression recognition.
We'll be opening up all the pre-training models and code soon
# Facial Expression Recognition

### Prepare

* pytorch>0.4
* python>2.7
* h5py
* sklearn
* cv2

### Prepare  Datasets

1. Prepare expresion images put in **/datasets**. If you only have three categories, delete the extra folders. This images should be saved gray image and 48*48 or uniform pixels.
    * Batch processing image that reshape and generating gray scale image can running **reshape.py** in **/script**
2. Running **prepocess_data.py** in **/script** , Generating _data.h5 formatted data sets.

### Train model 

```
python train.py --model Resnet34 --bs 128 --lr 0.01 --fold 1
```

### Test whole folder accuracy

* Test the accuracy of the whole folder, folders are named by category

```
python test.py
```

### Visualize result
`For a test image by a test model,Output the highest score categories and scores of each category`

* Thr pre-trained model can find in  **/model/Resnet50/5/Test_model.t7**
* running **visualize.py**, and the result saved in **/images/results/**

```
python visualize.py
```




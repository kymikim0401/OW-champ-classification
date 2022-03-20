# Overwatch Champ Classification

## Objective: Implement image classification with the images of Overwatch heroes

Tried implementing image classification neural network with the dataset given in the following link: https://www.kaggle.com/datasets/renanmav/overwatch-heroes-recognition

![image](https://user-images.githubusercontent.com/82307352/159163212-94393425-df96-4faf-83b3-5f2cb5fd3567.png)


Two models were use: 1)Self-made, crude CNN architecture and 2)Transfer learning **_RESNET50_** architecture. Despite the lack of training data (approximately 85 images per hero), RESNET50 seems to show a high classification precision whereas the self-made one obviously didn't work. You can check how I implemented transfer learning of RESNET50 in **'Model'** folder.

The sequence of the process is the following:

1. Image_Preprocess.py: Obtain image file path and reshape the raw dataset into (256,256) with RGB channel.
2. Data_preparation.py: Convert the preprocessed image dataset to csv file 
3. ow_dataset.py file in **'data'** folder: Create a OverWatchDataset class object 
4. train.py: Train the chosen neural network with prepared dataset
5. test.py: Test it! Voila!


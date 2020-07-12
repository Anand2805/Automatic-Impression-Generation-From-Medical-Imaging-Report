# Automatic-Impression-Generation-From-Medical-Imaging-Report
#### Process of generating textual description from medical report – end-to-end Deep learning model

## Business Problem
The problem statement here is to find the impression from the given chest X-Ray images. These images are in two types Frontal and Lateral view of the chest. With these two types of images as input we need to find the impression for given X-Ray.
To achieve this problem, we will be building a predictive model which involves both image and text processing to build a deep learning model. Automatically describing the content of the given image is one of the recent artificial intelligence models that connects both computer vision and natural language processing.

## File Structure
1.[EDA](https://github.com/Anand2805/Automatic-Impression-Generation-From-Medical-Imaging-Report/blob/master/EDA.ipynb)  
2.[Basic Model](https://github.com/Anand2805/Automatic-Impression-Generation-From-Medical-Imaging-Report/blob/master/Basic_Model.ipynb)  
3.[Inception_training_xray](https://github.com/Anand2805/Automatic-Impression-Generation-From-Medical-Imaging-Report/blob/master/inception_training_xray.ipynb)   
4.[Final Model](https://github.com/Anand2805/Automatic-Impression-Generation-From-Medical-Imaging-Report/blob/master/Final_Model.ipynb)  
5.[Error Anaysis](https://github.com/Anand2805/Automatic-Impression-Generation-From-Medical-Imaging-Report/blob/master/error_analysis.ipynb)  
6.[Inference (Final)](https://github.com/Anand2805/Automatic-Impression-Generation-From-Medical-Imaging-Report/blob/master/final.ipynb)  

##	My Approach – Solution
Initially I will be doing the Exploratory Data Analysis part I both image input and text output with EDA I could find the data imbalance, Images availability per patient, Type of images associated for each patient. After the EDA I will be implementing deep learning model with two different approach to find the improvement on one another.
1. The basic model: 
A simple encoder and decoder architecture. In encoder part it will have the CNN single fully connected layer to get the feature vector of images from pretrained InceptionV3 model. Decoder part will be having LSTM layer where it takes two inputs one is image feature vector and the sequence of text to word in each time step.

2. Main Model: 
I will be using encoder-decoder architecture to generate the impression from the chest X-ray. The encoder will output the image feature vectors. The feature vectors are then passed to decoder with attention mechanism this will generate the next word for the content of the image. With same model approach from basic model I will be creating a new architecture which is implemented using the research paper Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification 

As initial step I will do an image classification using InceptionV3 model over this dataset https://www.kaggle.com/yash612/covidnet-mini-and-gan-enerated-chest-xray. With this classification model I will save the weights over this training and use this weight in Encoder feature extraction by loading the saved weights to InceptionV3. 
	Encoder: 
The encoder is a single fully connected linear model. The input image is given to InceptionV3 to extract the features. this extracted feature of two images are added and input to the FC layer to get the output vector. This last hidden state of the Encoder is connected to the Decoder.
	Decoder: 
The Decoder is a have a Bidirectional LSTM layer which does language modelling up to the word level. The first-time step receives the encoded output from the encoder and the <start> vector. This input passed to 2 stage Bidirectional LSTM layer with attention mechanism. The output vector is two vector one is predicted label and other is the previous hidden state of decoder this fed back again to decoder on each time step. Detailed Architecture is mentioned below.


![Model Architecture](https://github.com/Anand2805/Automatic-Impression-Generation-From-Medical-Imaging-Report/blob/master/CS.png)

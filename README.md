# Non-linear-Video-Navigation

## Requirements:
- python3
- redis db
- training dataset

## Setup : 
 - Step 1 - Clone the repository
 - Step 2 - Create a virtual environment ```virtualenv -p python3 knnenv```
 - Step 3 - Install the dependencies from requirements.txt ```pip3 install -r requirements.txt```
 - Step 4 - Create a folders named ```trainingframes/Train``` inside the KNN folder.
 - Step 5 - Place all the labelled images inside the ```Train``` folder.
 - Step 6 - Now run the command ```python3 knn_classifier -d path_to_trainingframes_folder``` inside the KNN folder.
 - Step 7 - Now the trained model will be stored in the redis database.
  
  


# League-of-Legends-Challenger-Grandmaster-Outcome-Predictor
## Description:
This project uses variables scraped from the RIOT API to predict, given two teams or game states, which team would be more likely to win. It uses a two-step process using 2 models to determine a prediction. The web application was built using Flask and html.
# Requirements
Python Libraries:
- Pandas
- os
- csv
- json
- numpy
- Pytorch
- Scikit-learn
- Pickle
- Matplotlib
- Flask
- Flask-WTF
- WTForms
- Secrets
Device Specifications:
- At base, **game_data_predictor.ipynb** requires a CUDA gpu but this can be easily changed to cpu by replacing torch.device("cuda"); "cuda" to "cpu"
# How to Run 
- Open powershell prompt and use command: **cd** path/to/front-end
- Use command: flask --app front_end.py --debug run
- Copy the link produced by the prompt and open it in your browser of choice
- Select which ML model to use and 10 champions for each role.
- A prediction should be provided where a percentage for Logistic Regression and Random forest are given.

# Contents
## Data Processing
Dataset Obtained from: [League of Legends EUW Challenger/Grandmaster games by Bence Szilagyi](https://www.kaggle.com/datasets/benceszilagyi/league-of-legends-euw-challengergrandmaster-games).

### Data Pre-processing.ipynb
Takes the scraped data from the API in JSON form and extracts the meaningful attributes for our model.
### Exploratory_Data_Analysis.ipynb
Explores the win/loss ratios of specified attributes and their prescence within games.
### game_data_predictor.ipynb
Uses a Feed Forward Neural Network to predict in game variable using Pytorch on CUDA gpu.
### match_data.csv
Contains the cleaned match data

## Models
### KNN.ipynb
Builds a KNN classifier given the dataset. Hyperparameters tuned using GridSearchCV.
### Logsitic_Regression.ipynb
Builds a logistic regression classifier given the dataset. Hyperparameters tuned using GridSearchCV.
### MLP.ipynb
Builds a MLP classifier given the dataset. Hyperparameters tuned using GridSearchCV.
### Random_Forest.ipynb
Builds a Random Forest classifier given the dataset. Hyperparameters tuned using GridSearchCV.
### encoding_scheme.pkl
Saves the encoding scheme given a 10 permutation of 163 champions.
### kNN_model.pkl
The saved kNN model.
### logistic_regression_model.pkl
The saved logistic regression model.
### mpl_model.pkl
The saved MPL model.
### random_forest_model.pkl
The saved random forest model.
## front-end
### encoding_scheme.pkl
Saves the encoding scheme given a 10 permutation of 163 champions.
### fnn_num_model.pth
The saved FNN model built using Pytorch for our multi-output predictor.
### for front end.ipynb
A test notebook for our front end application.
### front_end.py
Runs the Flask front end of the application
### kNN_model.pkl
The saved kNN model.
### logistic_regression_model.pkl
The saved logistic regression model.
### mpl_model.pkl
The saved MPL model.
### random_forest_model.pkl
The saved random forest model.
### scaler.pkl
Scales all attributes to be between 0 and 1.


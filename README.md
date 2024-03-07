# FortuneCookies
A Markov chains model that learns the patterns within a fortune (or proverb), and generates unique ones on command

## Structure of the folders
- _datasets_: contains the combined training dataset and 100 example generated fortunes (which were used for evaluation)
- _notebooks_: contains the hyperparameter optimization and final training process for the model 
- _gui-assets_: contains everything related to the gui; to run please **run fortune_gui.py**
- _evaluation_: contains all scripts needed to calculate evaluation metrics for the generated dataset; to run again please **run evaluation.py**
- _pre-trained models_: contains all pre-trained models
- _results_: contains the results from the evaluation as csv files 

## GUI
The Markov model is fully integrated, and generates new fortunes upon each request, further customizable by the state size.

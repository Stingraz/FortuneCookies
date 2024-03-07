# FortuneCookies
A Markov chains model that learns the patterns within a fortune (or proverb), and generates unique ones on command

## Structure of the folders
- _datasets_: contains the combined training dataset and 100 generated fortunes from each model
- _notebooks_: contains the training process for each model as python notebook files
- _gui-assets_: contains everything related to the gui; to run please **run fortune_gui.py**
- _evaluation_: contains all scripts needed to calculate evaluation metrics for the generated datasets; to run again please **run evaluation.py**
- _presentation_report_: contains the prelim. presentation and the written report
- _pre-trained models_: contains all pre-trained models with a generate fortune function
- _results_: contains the results from the evaluation as csv files for each model
- _visulization_: contains the visualization of the dataset as well as the visualization of the evalution results as python notebooks

## GUI
The Markov model is fully integrated, and generates new fortunes upon each request, further customizable by the state size.

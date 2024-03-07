# FortuneCookies
A Markov chains model that learns the patterns within a fortune (or proverb), and generates unique ones on command

## Structure of the folders

datasets: contains the trainingdataset and 100 generated fortunes from each model
gui-assets: contains everything related to the gui
evaluation: contains all scripts needed to calculate evaluation metrics for the generated datasets
pre-trained models: contains all pre-trained models with a generate fortune function
results: contains the results from the evaluation as csv files

##GUI

The Markov model is fully integrated, and generates new fortunes upon each request, further customizable by the state size.

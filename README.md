# csci1470-final-project-ihasan2-mlunghi-tliu46

## Set Up
1. Create and activate virtual environment
2. Install packages in requirements.txt

## Preprocessing (preprocessed files are already available in repository)
1. Download politeness.tsv from the link below into the data folder
2. Inside data-prep/src folder, run python preprocess.py
3. preprocess.py should generate 10 files in the data folder

## Training and Evaluation (trained models are already available in links below)
1. Inside train/src folder, run python train.py --model ["TAGGER" or "GENERATOR"] --load [FILEPATH] --save [FILEPATH]
2. Specify whether you want to train the tagger or generator using the --model parameter
3. If you want to continue training an existing model, provide a filepath for an existing model with the --load parameter
4. If you want to save the resulting trained model for later use, provide a filepath with the --save parameter
5. After running, the script will generate a pickle file with the model's loss over epochs, display a plot of the model's loss over epochs, print out a sample output, and print out ROUGE and BLEU metrics for the model

## Links (requires Brown University email)
1. politeness.tsv: https://drive.google.com/file/d/1pPLMbpcn5Xy54dX2EWoqtMGjO8AB8wlM/view?usp=sharing
2. Trained tagger model: https://drive.google.com/file/d/116vcJeqBJdPH-lcm-6ciwOGAW9lLCMVg/view?usp=sharing
3. Trained generator model: https://drive.google.com/file/d/1T1NSASXjISl00IWqzb41n4tBUeyjopbs/view?usp=sharing
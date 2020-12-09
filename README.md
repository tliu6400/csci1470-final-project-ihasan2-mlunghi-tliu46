# csci1470-final-project-ihasan2-mlunghi-tliu46

## Set Up (required)
1. Create and activate virtual environment
2. Install packages in requirements.txt

## Preprocessing (optional: preprocessed files are already available in repository)
1. Download politeness.tsv from the link below (filepath: data/politeness.tsv)
2. Inside data-prep/src folder, run python preprocessing.py
3. Preprocessing.py should generate 10 files in the data folder

## Training and Evaluation (optional: trained models are already available in links below)
1. Inside train/src folder, run python train.py --model ["TAGGER" or "GENERATOR"] --load [FILEPATH] --save [FILEPATH]
2. Specify whether you want to train the tagger or generator using the --model parameter
3. If you want to continue training an existing model, provide a filepath for an existing model with the --load parameter
4. If you want to save the resulting trained model for later use, provide a filepath with the --save parameter
5. After running, 

## Links (requires Brown University email)
politeness.tsv: https://drive.google.com/file/d/1pPLMbpcn5Xy54dX2EWoqtMGjO8AB8wlM/view?usp=sharing
data.tsv: https://drive.google.com/file/d/1Pqx_dAQkTUvkwAdvkLo580z-sOgOY-bR/view?usp=sharing
Trained tagger model: https://drive.google.com/file/d/116vcJeqBJdPH-lcm-6ciwOGAW9lLCMVg/view?usp=sharing
Trained generator model: https://drive.google.com/file/d/1T1NSASXjISl00IWqzb41n4tBUeyjopbs/view?usp=sharing
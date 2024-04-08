"""
This script can be used to use the hybrid model to make predictions for the input text

Arguments:
    --t (str): Input text
    --d (str): Device on which calculations will be performed (cpu or cuda)
    --v (bool): Whether to show logger's messages.
    --p (bool) If True, then model will output probabilities for each class. Otherwise - predicted label.
"""


import warnings
warnings.filterwarnings("ignore")

import argparse
from hybrid_model_for_russian_sentiment_analysis.model import CustomHybridModel


def t_or_f(arg):
    arg = str(arg)
    if arg == 'True':
       return True
    elif arg == 'False':
       return False
    else:
       raise ValueError("invalid value for boolean argument")

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=str)
parser.add_argument('--d', type=str)
parser.add_argument('--v', type=t_or_f)
parser.add_argument('--p', type=t_or_f)

args = parser.parse_args()

model = CustomHybridModel(device=args.d, verbose=args.v)

if args.p:
    prediction = model.predict_proba(data=[args.t])
    print(f"Predicted probabilities: {prediction}")
else:
    prediction = model.predict(data=[args.t])
    prediction = ['sensitive' if k==0 else 'non-sensitive' for k in prediction]
    print(f"Predicted labels: {prediction}")
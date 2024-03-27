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
       raise ValueError("invalid value for boolean argument --p")

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
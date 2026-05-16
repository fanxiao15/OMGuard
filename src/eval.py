import json, argparse, sys
from pathlib import Path
import yaml
# from utils import *
from sklearn.metrics import classification_report, confusion_matrix
import re

def extract_json_string(s: str) -> str:
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace == -1 or last_brace == -1:
        raise ValueError("No valid JSON found in the string")
    return s[first_brace:last_brace+1]


def main(args):

    
    vanilla_misleading_ground = json.load(open(f"../datas/test_data.json", encoding='utf-8'))
   
    print(f"Vanilla misleading ground size: {len(vanilla_misleading_ground)}")
    
    results_datas = json.load(open(f"../results/misleading_detection_test_OMGuard.json", encoding='utf-8'))

    labels = []
    predictions = []
    all_ = {}

    total = 0
    for key in results_datas:
        
        ground_data = vanilla_misleading_ground[key]
        results_data = results_datas[key]

      
        ground = ground_data['Misleading']
        ground_clean = re.sub(r'[\x00-\x1f\x7f]', '', ground)
        ground_label = json.loads(ground_clean)['Misleading']
        s = results_data['Misleading']
        s_clean = re.sub(r'[\x00-\x1f\x7f]', '', s)
        open_label = json.loads(s_clean)['Misleading']
            
        if ground_label == "Yes":
            labels.append(1)
        else:
            labels.append(0)

        if open_label == "Yes":
            predictions.append(1)
        else:
            predictions.append(0)  
        total += 1
      
    target_names = ['Non-Misleading', 'Misleading']
    print(classification_report(labels, predictions, target_names=target_names, digits=4))
    print(confusion_matrix(labels, predictions))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from the last time')
    parser.add_argument("--num_samples", type=int, default=1000)

    args = parser.parse_args()
    main(args)

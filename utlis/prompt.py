from typing import *
from openprompt.data_utils import InputExample
import json

def get_examples(data_dir: Optional[str] , split: Optional[str] = None) -> List[InputExample]:
    outputExamples = []
    print(f'File loaded from {data_dir} ')
    with open(data_dir, "r", encoding='UTF-8') as f:
        data = json.load(f)
    for item in data:
        outputExamples.append(InputExample(guid = item['id'], text_a = item['question'], label = int(item['answer']) ))
    return outputExamples
    

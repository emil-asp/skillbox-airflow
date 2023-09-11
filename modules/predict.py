import dill
import os
from pathlib import Path
import logging
import json


path = os.environ.get('PROJECT_PATH', '.')

def predict():
    import pandas as pd

    model_filename = f'{path}/data/models/cars_pipe_202309112236.pkl'
    tests_path = f'{path}/data/test'
    
    logging.info('paths: ' + model_filename + ', ' + tests_path)
    
    # Load model
    with open(model_filename, 'rb') as file:
        model = dill.load(file)

    logging.info('Model loaded:' + model_filename)
    
    test_data = []
    for child in Path(tests_path).iterdir():
        if child.is_file():
            logging.info('Test file load: ' + str(child.resolve()))
            with open(str(child.resolve()), 'rb') as file:
                item_data = json.load(file)
                test_data.append(item_data)

    logging.info('Tests loaded: ' + str(len(test_data)))

    data = pd.DataFrame(test_data)
    data['preds'] = model.predict(data)

    logging.info('predicted: ', str(len(data[['id', 'preds', 'price']])))
    
    data[['id', 'preds', 'price']].to_csv(f'{path}/data/predictions/preds.csv', sep='\t', encoding='utf-8')
        
    #with open(f'{path}/data/predictions/preds.json', 'w') as f:
    #    json.dump(data.to_json(orient='records')[1:-1].replace('},{', '} {'), f)
        
if __name__ == '__main__':
    predict()

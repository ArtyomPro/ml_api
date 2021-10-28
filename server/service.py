from flask import Flask, request, session, escape, send_file
from flask_restx import Api, Resource, fields
from flask import send_from_directory
from dao import *
import pandas as pd
import os
import shutil
import uuid
import json
import numpy as np

app = Flask(__name__)
api = Api(app)

# папка для сохранения загруженных файлов
UPLOAD_FOLDER = './data'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = os.urandom(24)

ml_models_dao = MLModelsDAO()

def allowed_file(filename):
    """ Функция проверки расширения файла """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/login', methods = ['GET'])
def login():
    """
    Для создание индивидуальной сессии. Сессией считается экземпляр класса MLApi на стороне клиента
    """
    if len(session) == 0:
        session['uuid'] = uuid.uuid4().__str__()
    return ''

@app.route('/api/logout', methods=['GET'])
def logout():
    """
    Удаление всех файлов при окончании сессии
    """
    if len(session) != 0:
        uuid = session.pop('uuid', None)
        shutil.rmtree(f"{UPLOAD_FOLDER}/{uuid}")
        
        ml_models_dao.delete_user(uuid)
        
    return ''

@app.route('/api/datasets/<dataset_type>', methods=['POST'])
def upload_files(dataset_type):
    """
    Функция для загрузки датасета. Если в хранилище уже существует датасет с типом <dataset_type>, то произойдет перезапись.
    """
    if len(session) == 0:
        return 'you are not authorized', 401
    if dataset_type not in ['train','eval','test']:
        return 'Only train, eval, test value in type_.', 400
    try:
        df = api.payload
        df = pd.DataFrame(df)
        uuid = escape(session['uuid'])
        if not os.path.isdir(f"{UPLOAD_FOLDER}/{uuid}"):
            os.mkdir(f"{UPLOAD_FOLDER}/{uuid}")
        df.to_csv(f"{UPLOAD_FOLDER}/{uuid}/{dataset_type}.csv", index=False, sep='\t')
    except Exception as ex:
        return ex.__str__(), 500
    return ''

@app.route('/api/ml_models/available', methods=['GET'])
def models_available():
    """
    Выводит список доступных на сервере моделей
    """
    if len(session) == 0:
        return 'you are not authorized', 401
    
    return {'models_types' : list(models_types.keys())}

@api.route('/api/ml_models')
class MLModels(Resource):
    '''
    Класс для создания, получения ml моделей
    '''

    def get(self):
        if len(session) == 0:
            return 'you are not authorized', 401
        
        return ml_models_dao.get_all_by_uuid(escape(session['uuid']))

    def post(self):
        if len(session) == 0:
            return 'you are not authorized', 401
        try:
            return ml_models_dao.add(escape(session['uuid']), api.payload['model_type'], api.payload['params'])
        except AttributeError as a:
            return  a.__str__(), 500

@app.route('/api/fit/<id>', methods=['POST'])
def fit_model(id):
    '''
    функция обучения модели
    '''
    if len(session) == 0:
        return 'you are not authorized', 401
    
    try:
        target = api.payload['target']
        feat_list = api.payload['feat_list']
        fit_params = api.payload['fit_params']
        uuid = escape(session['uuid'])

        df = pd.read_csv(f"{UPLOAD_FOLDER}/{uuid}/train.csv", sep='\t')
    
        ml_models_dao.fit(uuid, id, df, target, feat_list, fit_params)
        return ''
    except Exception as ex:
        return ex.__str__(), 500

@app.route('/api/evaluate/<id>', methods=['POST'])
def evaluate(id):
    '''
    функция для оценки модели по ключевым метрикам
    '''
    if len(session) == 0:
        return 'you are not authorized', 401
    
    try:
        metrics = api.payload['metrics']
        uuid = escape(session['uuid'])
    
        df = pd.read_csv(f"{UPLOAD_FOLDER}/{uuid}/eval.csv", sep='\t')
        ml_models_dao.evaluate(uuid, id, df, metrics)
        return ''
    except Exception as ex:
        return ex.__str__(), 500
    
@app.route('/api/predict/<id>', methods=['GET'])
def predict(id):
    '''
    Функция возвращает предсказания модели  
    '''
    if len(session) == 0:
        return 'you are not authorized', 401
    
    try:
        sample_type = api.payload['sample_type']
        predict_type = api.payload['predict_type']
        uuid = escape(session['uuid'])
    
        df = pd.read_csv(f"{UPLOAD_FOLDER}/{uuid}/{sample_type}.csv", sep='\t')
        if predict_type == 'predict':
            preds = ml_models_dao.predict(uuid, id, df)
        elif predict_type == 'predict_proba':
            preds = ml_models_dao.predict_proba(uuid, id, df)
        else:
            raise AttributeError(f'predict_func only predict or predict_proba')
            
        lists = preds.tolist()
        json_str = json.dumps(lists)

        return json_str
    except Exception as ex:
        return ex.__str__(), 500


@api.route('/api/ml_models/<int:id>')
class MLModelsID(Resource):
    '''
    Класс для редактирования гиперпарметров модели и для удаления модели
    '''

    def put(self, id):
        if len(session) == 0:
            return 'you are not authorized', 401
    
        try:
            uuid = escape(session['uuid'])
            ml_models_dao.update(uuid, id, api.payload['params'])
            return '', 204
        except Exception as ex:
            return ex.__str__(), 500

    def delete(self, id):
        if len(session) == 0:
            return 'you are not authorized', 401
    
        try:
            uuid = escape(session['uuid'])
            ml_models_dao.delete(uuid, id)
            return '', 204
        except Exception as ex:
            return ex.__str__(), 500


if __name__ == '__main__':
    app.run()
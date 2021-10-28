from collections import defaultdict
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import copy

# этот список можно пополнять и дальше в процессе расширения апи :)
models_types = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier' : RandomForestClassifier,    
    'LinearRegression' : LinearRegression,
    'RandomForestRegressor' : RandomForestRegressor,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'KNeighborsClassifier': KNeighborsClassifier,
    'KNeighborsRegressor': KNeighborsRegressor
}

class MLModelsDAO:
    '''
    Класс для хранения моделей и их производных для всех пользователей
    '''
    def __init__(self):
        self.__ml_models = defaultdict(dict)
        self.__instances = defaultdict(dict)
        self.__feats = defaultdict(dict)
        self.counter = defaultdict(int)  
    
    def add(self, uuid, type_, params):
        '''
        Добавление модели в общий список
        '''
        if type_ in models_types:
            ml_model = {}
            ml_model['type'] = type_
            ml_model['params'] = params
            ml_model['fitted'] = False
            ml_model['metrics'] = {}
            self.__ml_models[uuid][self.counter[uuid]] = ml_model
            self.__instances[uuid][self.counter[uuid]] = models_types[type_](**params)
            self.counter[uuid] += 1
            return self.counter[uuid] - 1
        else:
            raise AttributeError(f'Model type {type_} doesnt exist.')
    
    def get_all_by_uuid(self,uuid):
        '''
        Возвращает список моделей пользователя
        '''
        return self.__ml_models[uuid]
    
    def fit(self,uuid, id_, X, target, feat_list, fit_params):
        '''
        Функция обучения и refit модели
        '''
        if self.__ml_models[uuid][int(id_)]['fitted']:
            #refit model
            params = copy.deepcopy(self.__instances[uuid][int(id_)].get_params())
            self.__instances[uuid][int(id_)] = copy.deepcopy(type(self.__instances[uuid][int(id_)]))(**params)
            self.__ml_models[uuid][int(id_)]['fitted'] = False
            # так как модель уже новая необходимо обнулить старые рузультаты
            self.__ml_models[uuid][int(id_)]['metrics']  = {}
            
        self.__instances[uuid][int(id_)].fit(X[feat_list],X[target],**fit_params)
        self.__feats[uuid][int(id_)] = (feat_list,target)
        self.__ml_models[uuid][int(id_)]['fitted'] = True
        
    def predict(self,uuid,id_,X):
        '''
        Функция предсказания модели 
        '''
        if not self.__ml_models[uuid][int(id_)]['fitted']:
            raise Exception('Model not fitted')
            
        feat_list,target = self.__feats[uuid][int(id_)]
        
        pred = self.__instances[uuid][int(id_)].predict(X[feat_list])
        
        return pred

    def predict_proba(self,uuid,id_,X):
        '''
        Функция предсказания модели (вероятности)
        '''
        if not self.__ml_models[uuid][int(id_)]['fitted']:
            raise Exception('Model not fitted')
            
        feat_list,target = self.__feats[uuid][int(id_)]
        
        pred = self.__instances[uuid][int(id_)].predict_proba(X[feat_list])[:,1]
        
        return pred
        
    def evaluate(self,uuid,id_,X,metrics):
        '''
        Оценка ключевых метрик модели
        '''
        for metric,predict_func in metrics.items():
        
            if predict_func == 'predict':
                pred = self.predict(uuid,id_,X)
            elif predict_func == 'predict_proba':
                pred = self.predict_proba(uuid,id_,X)
            else:
                raise AttributeError(f'predict_func only predict or predict_proba')
            
            _,target = self.__feats[uuid][int(id_)]
        
            self.__ml_models[uuid][int(id_)]['metrics'][metric] = getattr(sklearn.metrics, metric)(X[target], pred) 

    def delete(self,uuid, id_):
        '''
        Удаление моделей из списка
        '''
        del self.__ml_models[uuid][id_]
        del self.__instances[uuid][id_]
        del self.__feats[uuid][id_]
        
    def update(self,uuid,id_,params):
        '''
        Измненеие гиперпараметров модели
        '''
        self.__instances[uuid][id_] = copy.deepcopy(type(self.__instances[uuid][id_]))(**params)
        self.__ml_models[uuid][id_]['params'] = params
        # так как модель уже новая необходимо обнулить старые рузультаты
        self.__ml_models[uuid][id_]['metrics']  = {}
        self.__ml_models[uuid][id_]['fitted'] = False
    
    def delete_user(self,uuid):
        '''
        Удаление пользователя
        '''
        del self.__ml_models[uuid]
        del self.__instances[uuid]
        del self.__feats[uuid]
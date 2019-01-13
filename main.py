import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

def xgboost(argsDict):
    max_depth = argsDict["max_depth"] + 3
    n_estimators = argsDict['n_estimators'] * 100 + 100
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.01
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"] + 1
    gamma = argsDict["gamma"]/10.0
    colsample_bytree = argsDict["colsample_bytree"] * 0.1 + 0.7
    reg_alpha = (10**argsDict["reg_alpha"])*1e-5
    xgboost = XGBClassifier(max_depth=max_depth,  # 最大深度
                            n_estimators=n_estimators,  # 树的数量
                            learning_rate=learning_rate,  # 学习率
                            subsample=subsample,  # 采样数
                            min_child_weight=min_child_weight,  # 孩子数
                            gamma=gamma,
                            colsample_bytree=colsample_bytree,
                            reg_alpha=reg_alpha,
                            max_delta_step=10,  # 10步不降则停止
                            objective="binary:logistic",
                            n_jobs=16)
    return xgboost

def obj(argsDict):
    xgb = xgboost(argsDict)
    global X_train, y_train
    metric = cross_val_score(xgb, X_train, y_train, cv=5, scoring="roc_auc",n_jobs=16).mean()
    print(metric)
    return -metric

def modelTrain():
    space = {"max_depth": hp.randint("max_depth", 11),
             "n_estimators": hp.randint("n_estimators", 6),  # [0,1,2,3,4,5] -> [50,]
             "learning_rate": hp.randint("learning_rate", 6),  # [0,1,2,3,4,5] -> 0.05,0.06
             "subsample": hp.randint("subsample", 4),  # [0,1,2,3] -> [0.7,0.8,0.9,1.0]
             "min_child_weight": hp.randint("min_child_weight", 5),
             'gamma':  hp.randint("gamma", 5),
             'colsample_bytree':hp.randint("colsample_bytree", 4),
             'reg_alpha':hp.randint("reg_alpha", 7)}
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(obj, space, algo=algo, max_evals=100)  # max_evals表示想要训练的最大模型数量，越大越容易找到最优解
    return(best)

def preprocess_feature(online_train_feature,offline_train_feature,dataset):
    print(offline_train_feature.head())
    user_offline_feature=offline_train_feature.groupby('user_id')[['user_id']].max().reset_index(drop=True)
    user_offline_feature['count_merchant']=offline_train_feature.groupby('user_id').apply(lambda x:x[x['date'].notnull()].drop_duplicates('merchant_id')['merchant_id'].count()).reset_index(drop=True)
    user_offline_feature[['user_min_distance','user_max_distance','user_mean_distance','user_median_distance']]=offline_train_feature.groupby('user_id').apply(lambda x:x[(x['coupon_id'].notnull())&(x['date'].notnull())]['distance'].agg(['min','max','mean','median'])).reset_index(drop=True)
    user_offline_feature['buy_use_coupon'] = offline_train_feature.groupby('user_id').apply(lambda x: x[(x['coupon_id'].notnull())&(x['date'].notnull())]['user_id'].count()).reset_index(drop=True)
    user_offline_feature['buy_total'] = offline_train_feature.groupby('user_id').apply(lambda x: x[x['date'].notnull()]['user_id'].count()).reset_index(drop=True)
    user_offline_feature['coupon_received'] = offline_train_feature.groupby('user_id').apply(lambda x: x[x['coupon_id'].notnull()]['user_id'].count()).reset_index(drop=True)

    def get_user_date_datereceived_gap(s):
        date = pd.to_datetime(s[(s['date_received'].notnull())&(s['date'].notnull())]['date'], format='%Y%m%d')
        date_received = pd.to_datetime(s[(s['date_received'].notnull())&(s['date'].notnull())]['date_received'], format='%Y%m%d')
        return((date-date_received).dt.days.agg(['mean','min','max']))

    user_offline_feature[['avg_user_date_datereceived_gap','min_user_date_datereceived_gap','max_user_date_datereceived_gap']]=offline_train_feature.groupby('user_id').apply(get_user_date_datereceived_gap)
    user_offline_feature['buy_use_coupon_rate']=user_offline_feature['buy_use_coupon']/user_offline_feature['buy_total']
    user_offline_feature['user_coupon_transfer_rate']=user_offline_feature['buy_use_coupon']/user_offline_feature['coupon_received']

    merchant_feature = offline_train_feature.groupby('merchant_id')[['merchant_id']].max().reset_index(drop=True)
    merchant_feature['total_sales'] = offline_train_feature.groupby('merchant_id').apply(lambda x: x[x['date'].notnull()]['merchant_id'].count()).reset_index(drop=True)
    merchant_feature['sales_use_coupon'] = offline_train_feature.groupby('merchant_id').apply(lambda x: x[(x['coupon_id'].notnull())&(x['date'].notnull())]['merchant_id'].count()).reset_index(drop=True)
    merchant_feature['total_coupon'] = offline_train_feature.groupby('merchant_id').apply(lambda x: x[x['coupon_id'].notnull()]['merchant_id'].count()).reset_index(drop=True)
    merchant_feature[['merchant_min_distance', 'merchant_max_distance', 'merchant_mean_distance','merchant_median_distance']] = offline_train_feature.groupby('merchant_id').apply(lambda x: x[(x['coupon_id'].notnull()) & (x['date'].notnull())]['distance'].agg(['min', 'max', 'mean', 'median'])).reset_index(drop=True)
    merchant_feature['merchant_coupon_transfer_rate']=merchant_feature['sales_use_coupon']/merchant_feature['total_coupon']
    merchant_feature['coupon_rate'] = merchant_feature['sales_use_coupon'] / merchant_feature['total_sales']

    user_merchant_feature = offline_train_feature.groupby(['user_id','merchant_id'])[['user_id','merchant_id']].max().reset_index(drop=True)
    user_merchant_feature['user_merchant_buy_total'] = offline_train_feature.groupby(['user_id','merchant_id']).apply(lambda x: x[x['date'].notnull()]['user_id'].count()).reset_index(drop=True)
    user_merchant_feature['user_merchant_received'] = offline_train_feature.groupby(['user_id', 'merchant_id']).apply(lambda x: x[x['coupon_id'].notnull()]['user_id'].count()).reset_index(drop=True)
    user_merchant_feature['user_merchant_buy_use_coupon'] = offline_train_feature.groupby(['user_id', 'merchant_id']).apply(lambda x: x[(x['coupon_id'].notnull())&(x['date'].notnull())]['user_id'].count()).reset_index(drop=True)
    user_merchant_feature['user_merchant_any'] = offline_train_feature.groupby(['user_id', 'merchant_id'])['user_id'].count().reset_index(drop=True)
    user_merchant_feature['user_merchant_buy_common'] = offline_train_feature.groupby(['user_id', 'merchant_id']).apply(lambda x: x[(x['coupon_id'].isnull()) & (x['date'].notnull())]['user_id'].count()).reset_index(drop=True)
    user_merchant_feature['user_merchant_coupon_transfer_rate']=user_merchant_feature['user_merchant_buy_use_coupon']/user_merchant_feature['user_merchant_received']
    user_merchant_feature['user_merchant_coupon_buy_rate']=user_merchant_feature['user_merchant_buy_use_coupon']/user_merchant_feature['user_merchant_buy_total']
    user_merchant_feature['user_merchant_rate']=user_merchant_feature['user_merchant_buy_total']/user_merchant_feature['user_merchant_any']
    user_merchant_feature['user_merchant_common_buy_rate']=user_merchant_feature['user_merchant_buy_common']/user_merchant_feature['user_merchant_buy_total']

    other_feature = dataset[['user_id', 'merchant_id', 'coupon_id']]
    t1= dataset.groupby('user_id')['date_received'].count().reset_index()
    other_feature = pd.merge(other_feature,t1,how='left',on=['user_id'])
    other_feature = other_feature.rename(columns={'date_received':'this_month_user_receive_all_coupon_count'})
    t2= dataset.groupby(['user_id','coupon_id'])['date_received'].count().reset_index()
    other_feature = pd.merge(other_feature, t2, how='left', on=['user_id','coupon_id'])
    other_feature = other_feature.rename(columns={'date_received':'this_month_user_receive_same_coupon_count'})
    t3 = dataset.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x.astype('str'))).reset_index()
    t3['receive_number'] = t3['date_received'].apply(lambda s: len(s.split(':')))
    print(t3)
    print(other_feature)

    # feature=dataset[['user_id','merchant_id','coupon_id','distance']]
    # feature['day_of_week']=dataset['date_received'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d').weekday()+1)
    # feature['day_of_month']=dataset['date_received'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d').day)
    # def get_discount_man(s):
    #     s = str(s)
    #     s = s.split(':')
    #     if len(s) == 1:
    #         return np.NaN
    #     else:
    #         return int(s[0])
    # feature['discount_man']=dataset['discount_rate'].apply(get_discount_man)
    # def get_discount_jian(s):
    #     s = str(s)
    #     s = s.split(':')
    #     if len(s) == 1:
    #         return np.NaN
    #     else:
    #         return int(s[1])
    # feature['discount_jian']=dataset['discount_rate'].apply(get_discount_jian)
    # def is_man_jian(s):
    #     s = str(s)
    #     s = s.split(':')
    #     if len(s) == 1:
    #         return 0
    #     else:
    #         return 1
    # feature['is_man_jian'] = dataset['discount_rate'].apply(is_man_jian)
    # def calc_discount_rate(s):
    #     s = str(s)
    #     s = s.split(':')
    #     if len(s) == 1:
    #         return float(s[0])
    #     else:
    #         return 1.0 - float(s[1]) / float(s[0])
    # feature['discount_rate'] = dataset['discount_rate'].apply(calc_discount_rate)
    # print(feature)
    # feature = pd.merge(feature,user_offline_feature,on=['user_id'],how='left')
    # # feature = pd.merge(feature,user_online_feature,on=['user_id'],how='left')
    # feature = pd.merge(feature,merchant_feature, on=['merchant_id'], how='left')
    # feature = pd.merge(feature,user_merchant_feature, on=['user_id','merchant_id'], how='left')

    return feature


def preprocess_label(s):
    s['date'] = pd.to_datetime(s['date'], format='%Y%m%d')
    s['date_received'] = pd.to_datetime(s['date_received'], format='%Y%m%d')
    s['label']=0
    s.ix[(s['date']-s['date_received']).dt.days+1<=15,'label'] = 1
    return s['label']

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    online_train = pd.read_csv('../data/ccf_online_stage1_train.csv',header=None,nrows=1000)
    online_train.columns=['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']
    offline_train = pd.read_csv('../data/ccf_offline_stage1_train.csv',header=None,nrows=10000)
    offline_train.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']
    offline_test = pd.read_csv('../data/ccf_offline_stage1_test_revised.csv',header=None,nrows=1000)
    offline_test.columns=['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']
    online_train_feature1=online_train[((online_train['date']>=20160101)&(online_train['date']<=20160413))|((online_train['date'].isnull())&(online_train['date_received']>=20160101)&(online_train['date_received']<=20160413))]
    offline_train_feature1 = offline_train[((offline_train['date'] >= 20160101) & (offline_train['date'] <= 20160413)) | ((offline_train['date'].isnull()) & (offline_train['date_received'] >= 20160101) & (offline_train['date_received'] <= 20160413))]
    dataset1=offline_train[(offline_train['date_received'] >= 20160414) & (offline_train['date_received'] <= 20160514)]
    online_train_feature2 = online_train[((online_train['date'] >= 20160201) & (online_train['date'] <= 20160514)) | ((online_train['date'].isnull()) & (online_train['date_received'] >= 20160201) & (online_train['date_received'] <= 20160514))]
    offline_train_feature2 = offline_train[((offline_train['date'] >= 20160201) & (offline_train['date'] <= 20160514)) | ((offline_train['date'].isnull()) & (offline_train['date_received'] >= 20160201) & (offline_train['date_received'] <= 20160514))]
    dataset2 = offline_train[(offline_train['date_received'] >= 20160515) & (offline_train['date_received'] <= 20160615)]
    online_train_feature3 = online_train[((online_train['date'] >= 20160315) & (online_train['date'] <= 20160630)) | ((online_train['date'].isnull()) & (online_train['date_received'] >= 20160315) & (online_train['date_received'] <= 20160630))]
    offline_train_feature3 = offline_train[((offline_train['date'] >= 20160315) & (offline_train['date'] <= 20160630)) | ((offline_train['date'].isnull()) & (offline_train['date_received'] >= 20160315) & (offline_train['date_received'] <= 20160630))]
    dataset3 = offline_test

    feature1=preprocess_feature(online_train_feature1,offline_train_feature1,dataset1)
    label1=preprocess_label(dataset1)
    feature2 = preprocess_feature(online_train_feature2, offline_train_feature2, dataset2)
    label2 = preprocess_label(dataset2)
    feature12=pd.concat([feature1,feature2])
    label12=pd.concat([label1,label2])

    columns = [x for x in feature12.columns if x not in ['user_id','merchant_id','coupon_id']]
    X_train, X_test, y_train, y_test = train_test_split(feature12[columns], label12, test_size=0.20, random_state=0)
    # best = modelTrain()
    # print(best)
    args = {'colsample_bytree': 0, 'gamma': 3, 'learning_rate': 5, 'max_depth': 0, 'min_child_weight': 2,
            'n_estimators': 4, 'reg_alpha': 2, 'subsample': 1}
    xgb = xgboost(args)
    xgb.fit(X_train, y_train)
    y_train_pred = xgb.predict(X_train)
    y_test_pred = xgb.predict(X_test)
    print(roc_auc_score(y_train, y_train_pred))
    print(roc_auc_score(y_test, y_test_pred))
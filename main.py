from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from mesa import Mesa
from arguments import parser
from utils import Rater, load_dataset
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    
    # load dataset & prepare environment
    args = parser.parse_args()
    rater = Rater(args.metric)
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(args.dataset)
    base_estimator = DecisionTreeClassifier(max_depth=None)

    # meta-training
    print ('\nStart meta-training of MESA ... ...\n')
    mesa = Mesa(
        args=args, 
        base_estimator=base_estimator, 
        n_estimators=args.max_estimators)
    mesa.meta_fit(X_train, y_train, X_valid, y_valid, X_test, y_test)

    # test
    print ('\nStart ensemble training of MESA ... ...\n')
    runs = 50
    scores_list, time_list = [], []
    for i_run in tqdm(range(runs)):
        start_time = time.clock()
        mesa.fit(X_train, y_train, X_valid, y_valid, verbose=False)
        end_time = time.clock()
        time_list.append(end_time - start_time)
        score_train = rater.score(y_train, mesa.predict_proba(X_train)[:,1])
        score_valid = rater.score(y_valid, mesa.predict_proba(X_valid)[:,1])
        score_test = rater.score(y_test, mesa.predict_proba(X_test)[:,1])
        scores_list.append([score_train, score_valid, score_test])
    
    # print results to stdout
    df_scores = pd.DataFrame(scores_list, columns=['train', 'valid', 'test'])
    info = f'Dataset: {args.dataset}\nMESA {args.metric}|'
    for column in df_scores.columns:
        info += ' {} {:.3f}-{:.3f} |'.format(column, df_scores.mean()[column], df_scores.std()[column])
    info += ' {} runs (mean-std) |'.format(runs)
    info += ' ave run time: {:.2f}s'.format(np.mean(time_list))
    print (info)
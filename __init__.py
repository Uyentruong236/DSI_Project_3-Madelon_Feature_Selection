import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from psycopg2.extras import RealDictCursor


import seaborn as sns
from IPython.display import display

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ShuffleSplit,\
                                    StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.metrics import roc_auc_score, make_scorer
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)
from tqdm import tqdm 
from sklearn.decomposition import PCA
from scipy.stats import boxcox
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import matplotlib as mpl



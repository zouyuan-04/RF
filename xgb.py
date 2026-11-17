import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import warnings
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

warnings.filterwarnings("ignore")

folder = Path(f"D://kaggle/houseprices")
for items in folder.iterdir():
    print(items)

df = pd.read_csv(f"D://kaggle/houseprices/test.csv")
print(df.head())
df = df.set_index('Id',drop=True)
print(df.head())
print(df.info())
print(df.describe())


dd = pd.read_csv(f"D://kaggle/houseprices/train.csv")
print(dd['SalePrice'].describe())
dd = dd.set_index('Id',drop=True)
plt.figure(figsize=(10,10))
sns.histplot(dd['SalePrice'],kde=True,stat='density')
plt.title('SalePrice Distribution')
plt.xlabel('SalePrice')
plt.ylabel('Density')
plt.show()

list(set(dd.dtypes.tolist()))
df_num = dd.select_dtypes(['int64','float64'])
print(df_num.head())
X = df_num.drop('SalePrice',axis=1)
y = df_num['SalePrice']
X_test = df.select_dtypes(['int64','float64'])

X_train,X_valid,y_train,y_valid = train_test_split(X,y,train_size=0.8,random_state=42)

axes = df_num.hist(figsize=(20,24),bins=30,xlabelsize=8,ylabelsize=8)
for ax in axes.flatten():
    ax.tick_params(axis='x', labelrotation=15)
plt.subplots_adjust(hspace=0.8, wspace=0.3)
plt.show()

param_grid = {
    'xgb__n_estimators': [100,200,300],
    'xgb__max_depth': [3,5,7],
    'scale__with_mean': [True, False],
    'scale__with_std': [True, False],
}
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scale',StandardScaler()),
     ('xgb',xgb.XGBRegressor(random_state=0,learning_rate=0.1)),
])

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train,y_train)
print("最佳参数：",grid.best_params_)
print("最佳得分:",np.sqrt(-grid.best_score_))
scores = cross_val_score(grid.best_estimator_,X,y,cv=5,scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"rmse_scores:{rmse_scores}")
predictions = grid.predict(X_valid)

print(predictions)
mae = mean_absolute_error(y_valid,predictions)
mse = mean_squared_error(y_valid,predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid,predictions)
print(f"MAE:{mae}")
print(f"MSE:{mse}")
print(f"RMSE:{rmse:.2f}")
print(f"R^2:{r2}")

results = grid.predict(X_test)
submission = pd.DataFrame({
    "Id":X_test.index,
    "prediction":results,
})
submission.to_csv('submission.csv',index=False,encoding='utf-8')


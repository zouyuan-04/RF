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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

folder = Path("D://kaggle/houseprices")
for items in folder.iterdir():
    print(items)

df = pd.read_csv("D://kaggle/houseprices/test.csv")
print(df.head())
df = df.set_index('Id', drop=True)
print(df.head())
print(df.info())
print(df.describe())


dd = pd.read_csv("D://kaggle/houseprices/train.csv")
print(dd['SalePrice'].describe())
dd = dd.set_index('Id', drop=True)

# SalePrice distribution
plt.figure(figsize=(10, 10))
sns.histplot(dd['SalePrice'], kde=True, stat='density')
plt.title('SalePrice Distribution')
plt.xlabel('SalePrice')
plt.ylabel('Density')
plt.show()

X = dd.drop('SalePrice', axis=1)
y = dd['SalePrice']
X_test = df[X.columns]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print(f"numeric features: {len(numeric_features)}")
print(f"categorical features: {len(categorical_features)}")

# Numeric feature distributions
df_num = X.select_dtypes(include=['int64', 'float64'])
axes = df_num.hist(figsize=(20, 24), bins=30, xlabelsize=8, ylabelsize=8)
for ax in axes.flatten():
    ax.tick_params(axis='x', labelrotation=45)
plt.subplots_adjust(hspace=0.8, wspace=0.3)
plt.show()

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('xgb', xgb.XGBRegressor(random_state=0, learning_rate=0.1)),
])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)

param_grid = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'prep__num__scaler__with_mean': [True, False],
    'prep__num__scaler__with_std': [True, False],
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)
print("最佳参数：", grid.best_params_)
print("最佳RMSE：", np.sqrt(-grid.best_score_))

scores = cross_val_score(grid.best_estimator_, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"rmse_scores:{rmse_scores}")

predictions = grid.predict(X_valid)

print(predictions)
mae = mean_absolute_error(y_valid, predictions)
mse = mean_squared_error(y_valid, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, predictions)
print(f"MAE:{mae}")
print(f"MSE:{mse}")
print(f"RMSE:{rmse:.2f}")
print(f"R^2:{r2}")

# Plot: predicted vs actual
plt.figure(figsize=(8, 8))
plt.scatter(y_valid, predictions, alpha=0.5, edgecolor='k', linewidth=0.3)
lims = [min(y_valid.min(), predictions.min()), max(y_valid.max(), predictions.max())]
plt.plot(lims, lims, 'r--', lw=2, label='Perfect fit')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title(f'Predicted vs Actual (RMSE={rmse:.2f}, R2={r2:.3f})')
plt.legend()
plt.tight_layout()
plt.show()

residuals = y_valid - predictions
plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals, alpha=0.5, edgecolor='k', linewidth=0.3)
plt.axhline(0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted SalePrice')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()

results = grid.predict(X_test)
submission = pd.DataFrame({
    "Id": X_test.index,
    "SalePrice": results,
})
submission.to_csv('submission_prediction.csv', index=False, encoding='utf-8')

plt.figure(figsize=(8, 6))
sns.histplot(results, kde=True, stat='density')
plt.xlabel('Predicted SalePrice')
plt.ylabel('Density')
plt.title('Test Predictions Distribution')
plt.show()

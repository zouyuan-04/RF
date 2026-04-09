import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df_train = pd.read_csv(r'c:\users\86182\Desktop\titanic\train.csv')
df_test = pd.read_csv(r'c:\users\86182\Desktop\titanic\test.csv')


def clean_data(df, age_medians=None, is_train=True):

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')


    if is_train:

        age_medians = df.groupby('Title')['Age'].median()

    df['Age'] = df.apply(
        lambda row: age_medians[row['Title']] if pd.isnull(row['Age']) else row['Age'],
        axis=1)

    df.drop('Cabin', axis=1, inplace=True)

    print("处理后缺失值检查：")
    print(df[['Age', 'Embarked']].isnull().sum())

    if is_train:
        return df, age_medians
    else:
        return df



df_train_clean, train_age_medians = clean_data(df_train, is_train=True)
df_test_clean = clean_data(df_test, age_medians=train_age_medians, is_train=False)

def add_features(df):
    # 家庭规模
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # 是否独行
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

df_train_clean = add_features(df_train_clean)
df_test_clean = add_features(df_test_clean)

fare_median = df_train_clean['Fare'].median()
df_test_clean['Fare'] = df_test_clean['Fare'].fillna(fare_median)


categorical_cols = ['Sex', 'Embarked', 'Title']
df_train_encoded = pd.get_dummies(df_train_clean, columns=categorical_cols, drop_first=True)

df_test_encoded = pd.get_dummies(df_test_clean, columns=categorical_cols, drop_first=True)

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Survived']
X = df_train_encoded.drop(columns=drop_cols, errors='ignore')
y = df_train_encoded['Survived']
X_test = df_test_encoded.drop(columns=['PassengerId', 'Name', 'Ticket'], errors='ignore')

common_cols = X.columns.intersection(X_test.columns)
X_test = X_test[common_cols]
X = X[common_cols]

print(f"训练集特征维度: {X.shape}")
print(f"测试集特征维度: {X_test.shape}")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"5折交叉验证平均准确率: {scores.mean():.4f}")


rf.fit(X, y)
predictions = rf.predict(X_test)
print(predictions)



df_test_clean['Survived_Pred'] = predictions
submission = df_test_clean[['PassengerId', 'Survived_Pred']]
submission.to_csv('titanic_predictions.csv', index=False)

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
feature_importances.to_csv('titanic_feature_importance.csv', index=False)

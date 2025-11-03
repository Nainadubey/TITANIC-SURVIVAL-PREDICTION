import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "Titanic-Dataset.csv"  
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
display(df.head())

print("\nInfo:")
display(df.info())

print("\nMissing values per column:")
display(df.isnull().sum())

data = df.copy()

data['Title'] = data['Name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
rare_titles = data['Title'].value_counts()[data['Title'].value_counts() < 10].index.tolist()
data['Title'] = data['Title'].replace(rare_titles, 'Rare')

if data['Embarked'].isnull().sum() > 0:
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

if data['Fare'].isnull().sum() > 0:
    data['Fare'] = data.groupby('Pclass')['Fare'].apply(lambda x: x.fillna(x.median()))

age_median_by_title = data.groupby('Title')['Age'].median()
data['Age'] = data.apply(
    lambda row: age_median_by_title[row['Title']] if pd.isnull(row['Age']) else row['Age'],
    axis=1
)

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data_model = data.drop(columns=[c for c in drop_cols if c in data.columns])

print("\nAfter feature engineering - preview:")
display(data_model.head())

print("\nMissing values now:")
display(data_model.isnull().sum())

if 'Survived' not in data_model.columns:
    raise ValueError("No 'Survived' column found in dataset!")

y = data_model['Survived'].astype(int)
X = data_model.drop(columns=['Survived'])

numeric_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

models = {
    'LogisticRegression': Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
    ]),
    'RandomForest': Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'GradientBoosting': Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
}

param_grids = {
    'LogisticRegression': {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2']
    },
    'RandomForest': {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [4, 8, None],
        'clf__min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.01, 0.1],
        'clf__max_depth': [3, 5]
    }
}

best_estimators = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipeline in models.items():
    print(f"\n Tuning {name} ...")
    grid = GridSearchCV(pipeline, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params for {name}: {grid.best_params_}")
    best_estimators[name] = grid.best_estimator_

results = []
plt.figure(figsize=(8,6))

for name, estimator in best_estimators.items():
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:,1] if hasattr(estimator.named_steps['clf'], "predict_proba") else estimator.decision_function(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc:.3f})")
    results.append((name, acc, prec, rec, f1, roc))

plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Titanic Models")
plt.legend()
plt.show()

res_df = pd.DataFrame(results, columns=['Model','Accuracy','Precision','Recall','F1','ROC_AUC']).sort_values(by='ROC_AUC', ascending=False)
print("\nSummary of model performance:")
display(res_df)

ohe = best_estimators['RandomForest'].named_steps['pre'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)

rf = best_estimators['RandomForest'].named_steps['clf']
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
print("\n Top 15 feature importances (Random Forest):")
display(feat_imp.head(15))

lr = best_estimators['LogisticRegression'].named_steps['clf']
X_trans = best_estimators['LogisticRegression'].named_steps['pre'].transform(X_train)
coef = lr.coef_[0]
coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef}).sort_values(by='coefficient', ascending=False)
print("\n Top 10 features increasing survival:")
display(coef_df.head(10))
print("\n Top 10 features decreasing survival:")
display(coef_df.tail(10))

os.makedirs("outputs", exist_ok=True)
res_df.to_csv('outputs/titanic_model_performance_summary.csv', index=False)
feat_imp.to_csv('outputs/titanic_feature_importances.csv', index=False)
coef_df.to_csv('outputs/titanic_logreg_coefficients.csv', index=False)

print("\n Files saved in 'outputs/' folder:")
print(" - titanic_model_performance_summary.csv")
print(" - titanic_feature_importances.csv")
print(" - titanic_logreg_coefficients.csv")

print("\n Sample of engineered training data:")
display(X_train.head(10).join(y_train.head(10)))

print("\n Modeling completed successfully.")


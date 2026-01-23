import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib.colors import LinearSegmentedColormap



def load_data(feature_file, label_file):
    X = pd.read_csv(feature_file, header=0)
    y = pd.read_csv(label_file, header=0).values.ravel()

    X = X.applymap(lambda x: str(x).replace(",", "") if isinstance(x, str) else x)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())
    return X, y

def tune_hyperparameters(X_train, y_train):
    param_space = {
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'n_estimators': Integer(50, 200),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0)
    }

    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        scoring='accuracy',
        cv=5,
        n_iter=200,
        n_jobs=-1,
        random_state=42
    )
    bayes_search.fit(X_train, y_train)
    return bayes_search.best_estimator_

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = tune_hyperparameters(X_train, y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = model.get_params()
    params['seed'] = 42  # Random seed
    num_boost_round = 100
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, 'test')])

    y_pred_prob = bst.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"Acc: {accuracy_score(y_test, y_pred):.4f}, Pre: {precision_score(y_test, y_pred):.4f}, "
          f"Recall: {recall_score(y_test, y_pred):.4f}, F1: {f1_score(y_test, y_pred):.4f}, AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
    return bst, X_train

def shap_analysis(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_train)
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    csv_path = f"shap_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"The shap_importance has been saved as '{csv_path}'")
    plt.figure()
    shap.summary_plot(shap_values.values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"shap_summary_bar.pdf", dpi=300)
    plt.close()
    print(f"The shap_summary_bar has been saved as 'shap_summary_bar.pdf'")
    plt.figure()
    shap.summary_plot(shap_values.values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_summary_beeswarm.pdf", dpi=300)
    plt.close()
    print(f"The shap_summary_beeswarm has been saved as 'shap_summary_beeswarm.pdf'")

def plot_top_shap_importance(importance_df, output_plot='top20_shap_importance.pdf'):
    top_20_features = importance_df.head(20)
    plt.figure(figsize=(10, 6))
    plt.scatter(top_20_features['Feature'], top_20_features['Importance'], color='Burlywood', s=80)
    for feature, importance in zip(top_20_features['Feature'], top_20_features['Importance']):
        plt.text(feature, importance + 0.05, f"{importance:.4f}", fontsize=10, ha='center', va='center', rotation=30)
    plt.ylabel('SHAP Importance')
    plt.xlabel('Feature')
    plt.title('Top 20 SHAP Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_plot, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Top 20 SHAP Feature Importance Map saved as '{output_plot}'")

def plot_top_shap_correlation(X_train, importance_df, output_plot='top20_shap_correlation.pdf'):
    top_20_features = importance_df['Feature'].head(20)
    corr_matrix = X_train[top_20_features].corr(method='pearson')
    plt.figure(figsize=(12, 10))
    custom_cmap = 'coolwarm'
    heatmap = sns.heatmap(
        corr_matrix,
        cmap=custom_cmap,
        annot=False,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5}
    )
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12)
    # sns.heatmap(corr_matrix, cmap=custom_cmap, annot=False, linewidths=0.5, cbar_kws={'shrink': 0.5})
    plt.title('Top 20 SHAP features Pearson correlation coefficient matrix')
    plt.savefig(output_plot, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Pearson correlation coefficient plot for the top 20 SHAP features has been saved as '{output_plot}'")

def main():
    feature_file = 'D:/py/pycode/PPROMpj/add_new/final_data/features_Translated_modified.csv'
    label_file = 'D:/py/pycode/PPROMpj/add_new/final_data/y_balanced_1.csv'

    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']

    X, y = load_data(feature_file, label_file)
    model, X_train = train_and_evaluate(X, y)
    shap_analysis(model, X_train)

    model.save_model('xgboost_model.json')
    print("The model has been saved as: 'xgboost_model.json'")

if __name__ == '__main__':
    main()

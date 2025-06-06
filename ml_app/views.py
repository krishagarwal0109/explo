import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from django.shortcuts import render, redirect
from .models import BatteryData
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def index(request):
    return render(request, 'ml_app/index.html')
    
def train_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'Final Database.csv')
    try:
        df = pd.read_csv(file_path)
        print(df.head())
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")

    df['Battery ID'] = 0
    batteries = []
    ID = 1
    for rul in df['RUL']:
        batteries.append(ID)
        if rul == 0:
            ID += 1
    df['Battery ID'] = batteries

    def plot_to_base64(plt):
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return image_base64

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    feature_correlation_plot = plot_to_base64(plt)

    X_all_features = df.drop(columns=['RUL', 'Battery ID'])
    y_all_features = df['RUL']
    rf_all_features = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_features='sqrt')
    rf_all_features.fit(X_all_features, y_all_features)

    feature_importance_all = rf_all_features.feature_importances_
    feature_names_all = X_all_features.columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_names_all, y=feature_importance_all, palette='magma')
    plt.title('Feature Importance (All Features)')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    feature_importance_all_plot = plot_to_base64(plt)

    feature_names = ['Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)']
    df_train = df[df['Battery ID'] < 9]
    df_test = df[df['Battery ID'] >= 9]

    X_train = df_train[feature_names]
    y_train = df_train['RUL']
    X_test = df_test[feature_names]
    y_test = df_test['RUL']

    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # --- Hyperparameter tuning using RandomizedSearchCV ---

    # KNN
    knn_params = {
        'n_neighbors': np.arange(1, 20),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    knn_search = RandomizedSearchCV(
        KNeighborsRegressor(),
        knn_params,
        n_iter=10,
        scoring='neg_root_mean_squared_error',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    knn_search.fit(X_train, y_train)
    knn_model = knn_search.best_estimator_

    # SVR
    svr_params = {
        'C': np.logspace(0, 4, 10),
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 5)),
        'epsilon': np.logspace(-3, -1, 3),
        'kernel': ['rbf']
    }
    svr_search = RandomizedSearchCV(
        SVR(),
        svr_params,
        n_iter=10,
        scoring='neg_root_mean_squared_error',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    svr_search.fit(X_train, y_train)
    svm_model = svr_search.best_estimator_

    # Random Forest
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [1, 2, 4],
        'max_depth': [None, 10, 20, 30]
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        rf_params,
        n_iter=10,
        scoring='neg_root_mean_squared_error',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    rf_model = rf_search.best_estimator_

    # --- Evaluation ---
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return mae, rmse, r2, y_pred

    knn_mae, knn_rmse, knn_r2, y_pred_knn = evaluate_model(knn_model, X_test, y_test)
    svm_mae, svm_rmse, svm_r2, y_pred_svm = evaluate_model(svm_model, X_test, y_test)
    rf_mae, rf_rmse, rf_r2, y_pred_rf = evaluate_model(rf_model, X_test, y_test)

    model_metrics = {
        'kNN': {'MAE': knn_mae, 'RMSE': knn_rmse, 'R2': knn_r2},
        'SVM': {'MAE': svm_mae, 'RMSE': svm_rmse, 'R2': svm_r2},
        'Random Forest': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2},
    }

    return (
        knn_model, svm_model, rf_model, sc, model_metrics,
        knn_mae, svm_mae, rf_mae, knn_rmse, svm_rmse, rf_rmse, knn_r2, svm_r2, rf_r2,
        y_pred_knn, y_pred_svm, y_pred_rf, X_test, y_test, df_train, df_test, feature_names, df,
        feature_correlation_plot, feature_importance_all_plot,
    )
(
    knn_model, svm_model, rf_model, scaler, model_metrics,
    knn_mae, svm_mae, rf_mae, knn_rmse, svm_rmse, rf_rmse, knn_r2, svm_r2, rf_r2,
    y_pred_knn, y_pred_svm, y_pred_rf, X_test, y_test, df_train, df_test, feature_names, df,
    feature_correlation_plot, feature_importance_all_plot
) = train_models()

def predict_rul(request):
    if request.method == 'POST':
        input_data = {
            'Max. Voltage Dischar. (V)': float(request.POST['max_voltage_discharge']),
            'Min. Voltage Charg. (V)': float(request.POST['min_voltage_charge']),
        }

        features = ['Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)']
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df[features])

        knn_pred = knn_model.predict(input_scaled)[0]
        svm_pred = svm_model.predict(input_scaled)[0]
        rf_pred = rf_model.predict(input_scaled)[0]

        BatteryData.objects.create(
            max_voltage_discharge=input_data['Max. Voltage Dischar. (V)'],
            min_voltage_charge=input_data['Min. Voltage Charg. (V)'],
            rul=rf_pred
        )

        def generate_insights(knn_pred, svm_pred, rf_pred, input_data, model_metrics):
            insights = []

            best_model = min(model_metrics, key=lambda x: model_metrics[x]['RMSE'])
            insights.append(f"The {best_model} model is the best for prediction, with the lowest RMSE of {model_metrics[best_model]['RMSE']:.2f}.")

            insights.append("Model Performance Comparison:")
            for model, metrics in model_metrics.items():
                insights.append(f"- {model}: MAE = {metrics['MAE']:.2f}, RMSE = {metrics['RMSE']:.2f}, R² = {metrics['R2']:.2f}")

            std_dev = np.std([knn_pred, svm_pred, rf_pred])
            insights.append(f"The predictions have a standard deviation of {std_dev:.2f}, indicating {'high' if std_dev > 10 else 'low'} variability across models.")

            feature_importance = rf_model.feature_importances_
            insights.append(f"Feature importance: 'Max. Voltage Dischar. (V)' contributes {feature_importance[0]*100:.2f}%, while 'Min. Voltage Charg. (V)' contributes {feature_importance[1]*100:.2f}% to the RUL prediction.")

            historical_data = BatteryData.objects.all().order_by('-id')[:5] 
            historical_rul = [data.rul for data in historical_data]
            avg_historical_rul = np.mean(historical_rul)
            insights.append(f"The current RUL prediction ({rf_pred:.2f}) is {'above' if rf_pred > avg_historical_rul else 'below'} the historical average of {avg_historical_rul:.2f} cycles.")
            
            if rf_pred > 100:
                health_status = "Good"
            elif rf_pred > 50:
                health_status = "Fair"
            else:
                health_status = "Poor"
            insights.append(f"Your battery is in {health_status} condition and can be used for approximately {rf_pred:.2f} cycles.")

            avg_cycles_per_day = 2
            estimated_days = rf_pred / avg_cycles_per_day
            insights.append(f"Based on your usage, the battery can last for approximately {estimated_days:.2f} days.")

            insights.append("To extend your battery's life:")
            insights.append("- Avoid deep discharges (below 20%).")
            insights.append("- Avoid charging above 80% frequently.")
            insights.append("- Keep the battery cool and avoid exposure to high temperatures.")

            if rf_pred < 30:
                insights.append("Warning: Your battery is nearing the end of its life. It is recommended to replace it soon to avoid unexpected failures")
            avg_battery_lifespan = 500 
            insights.append(f"The average lifespan of similar batteries is {avg_battery_lifespan} cycles. Your battery is performing {'better' if rf_pred > avg_battery_lifespan else 'worse'} than average.")

            if rf_pred < 50:
                insights.append(f"Recommendation: Based on the prediction, it is recommended to replace the battery in approximately {max(0, rf_pred / avg_cycles_per_day / 7):.2f} weeks.")

            return insights

        insights = generate_insights(knn_pred, svm_pred, rf_pred, input_data, model_metrics)

        def plot_to_base64(plt):
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            return image_base64

        models = ['kNN', 'SVM', 'Random Forest']
        predictions = [knn_pred, svm_pred, rf_pred]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=models, y=predictions, hue=models, palette='viridis', legend=False)
        plt.title('RUL Predictions by Model')
        plt.ylabel('Predicted RUL')
        model_comparison_plot = plot_to_base64(plt)

        plt.figure(figsize=(8, 5))
        sns.kdeplot([knn_pred, svm_pred, rf_pred], fill=True, palette='magma')
        plt.title('Distribution of Predicted RUL')
        plt.xlabel('Predicted RUL')
        plt.ylabel('Density')
        rul_distribution_plot = plot_to_base64(plt)

        metrics = ['MAE', 'RMSE', 'R2']
        plt.figure(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i + 1)
            sns.barplot(x=list(model_metrics.keys()), y=[model_metrics[model][metric] for model in model_metrics], palette='coolwarm')
            plt.title(f'{metric} Comparison')
            plt.ylabel(metric)
        plt.tight_layout()
        model_performance_plot = plot_to_base64(plt)

        plt.figure(figsize=(8, 5))
        sns.barplot(x=feature_names, y=rf_model.feature_importances_, palette='magma')
        plt.title('Feature Importance (Selected Features)')
        plt.ylabel('Importance Score')
        feature_importance_plot = plot_to_base64(plt)

        return render(request, 'ml_app/result.html', {
            'knn_pred': knn_pred,
            'svm_pred': svm_pred,
            'rf_pred': rf_pred,
            'insights': insights,
            'model_comparison_plot': model_comparison_plot,
            'rul_distribution_plot': rul_distribution_plot,
            'model_performance_plot': model_performance_plot,
            'feature_importance_plot': feature_importance_plot,
        })

    return render(request, 'ml_app/predict.html')

def training_results(request):
    global knn_mae, svm_mae, rf_mae, knn_rmse, svm_rmse, rf_rmse, knn_r2, svm_r2, rf_r2
    global y_pred_knn, y_pred_svm, y_pred_rf, X_test, y_test, df_train, df_test, feature_names, df
    global feature_correlation_plot, feature_importance_all_plot  # Include the new plots

    model_metrics = {
        'kNN': {'MAE': knn_mae, 'RMSE': knn_rmse, 'R2': knn_r2},
        'SVM': {'MAE': svm_mae, 'RMSE': svm_rmse, 'R2': svm_r2},
        'Random Forest': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2},
    }

    def plot_to_base64(plt):
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return image_base64

    metrics = ['MAE', 'RMSE', 'R2']
    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        sns.barplot(x=list(model_metrics.keys()), y=[model_metrics[model][metric] for model in model_metrics], palette='coolwarm')
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
    plt.tight_layout()
    model_performance_plot = plot_to_base64(plt)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_names, y=rf_model.feature_importances_, palette='magma')
    plt.title('Feature Importance (Selected Features)')
    plt.ylabel('Importance Score')
    feature_importance_plot = plot_to_base64(plt)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(y_pred_knn, label='kNN', fill=True)
    sns.kdeplot(y_pred_svm, label='SVM', fill=True)
    sns.kdeplot(y_pred_rf, label='Random Forest', fill=True)
    plt.title('Distribution of Predicted RUL')
    plt.xlabel('Predicted RUL')
    plt.ylabel('Density')
    plt.legend()
    predictions_distribution_plot = plot_to_base64(plt)

    residuals_knn = y_test - y_pred_knn
    residuals_svm = y_test - y_pred_svm
    residuals_rf = y_test - y_pred_rf

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred_knn, y=residuals_knn, label='kNN')
    sns.scatterplot(x=y_pred_svm, y=residuals_svm, label='SVM')
    sns.scatterplot(x=y_pred_rf, y=residuals_rf, label='Random Forest')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plots')
    plt.xlabel('Predicted RUL')
    plt.ylabel('Residuals')
    plt.legend()
    residuals_plot = plot_to_base64(plt)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred_knn, label='kNN', alpha=0.5)
    plt.scatter(y_test, y_pred_svm, label='SVM', alpha=0.5)
    plt.scatter(y_test, y_pred_rf, label='Random Forest', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Actual vs Predicted RUL')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.legend()
    actual_vs_predicted_plot = plot_to_base64(plt)

    dataset_stats = {
        'num_samples': len(df_train) + len(df_test),
        'num_features': len(feature_names),
        'mean_rul': df['RUL'].mean(),
        'std_rul': df['RUL'].std(),
    }

    feature_selection_explanation = """
    The two features, 'Max. Voltage Dischar. (V)' and 'Min. Voltage Charg. (V)', were selected based on their high correlation with the RUL (Remaining Useful Life) and their importance scores from the Random Forest model. 
    The correlation heatmap and feature importance plot above illustrate why these features were chosen.
    """

    return render(request, 'ml_app/training_results.html', {
        'model_metrics': model_metrics,
        'model_performance_plot': model_performance_plot,
        'feature_importance_plot': feature_importance_plot,
        'predictions_distribution_plot': predictions_distribution_plot,
        'residuals_plot': residuals_plot,
        'actual_vs_predicted_plot': actual_vs_predicted_plot,
        'feature_correlation_plot': feature_correlation_plot,
        'feature_importance_all_plot': feature_importance_all_plot,
        'feature_selection_explanation': feature_selection_explanation,
        'dataset_stats': dataset_stats,
    })

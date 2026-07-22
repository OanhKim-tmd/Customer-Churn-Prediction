import pandas as pd

from preprocessing import preprocess_data
from models import train_models
from evaluation import (
    evaluate_all_models,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance
)


df = pd.read_csv(
    "../data/processed/cleaned_data.csv"
)


X_train, X_test, y_train, y_test, scaler = preprocess_data(df)



trained_models = train_models(
    X_train,
    y_train
)



save_dir = "../outputs/figures"



results = evaluate_all_models(
    trained_models,
    X_test,
    y_test
)


results.to_csv(
    "../outputs/model_results.csv",
    index=False
)



for name, model in trained_models.items():

    name_clean = (
        name
        .replace(" ","_")
        .lower()
    )


    plot_confusion_matrix(
        model,
        X_test,
        y_test,
        name_clean,
        save_dir
    )


    plot_roc_curve(
        model,
        X_test,
        y_test,
        name_clean,
        save_dir
    )


    plot_precision_recall_curve(
        model,
        X_test,
        y_test,
        name_clean,
        save_dir
    )


plot_feature_importance(
    trained_models["Random Forest"],
    X_train.columns,
    save_dir
)
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Global style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def plot_target_distribution(df, target_col="Churn", save_path=None):
    """
    Plot target variable distribution.
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(
        data=df,
        x=target_col,
        palette="Blues_d",
        ax=ax
    )

    total = len(df)

    for p in ax.patches:
        percent = 100 * p.get_height() / total

        ax.annotate(
            f"{percent:.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title("Target Distribution")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_categorical_distribution(df, cat_cols, save_path=None):
    """
    Plot distribution of categorical variables.
    """

    n_rows = (len(cat_cols) + 1) // 2

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, 4 * n_rows)
    )

    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cat_cols):

        sns.countplot(
            data=df,
            x=col,
            palette="Set2",
            ax=axes[i]
        )

        axes[i].set_title(f"Distribution of {col}")
        axes[i].tick_params(axis="x", rotation=30)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_churn_rate_by_category(
    df,
    cat_cols,
    target_col="Churn",
    save_path=None
):
    """
    Plot churn rate by categorical features.
    """

    df_temp = df.copy()

    if df_temp[target_col].dtype == "object":
        df_temp["Target"] = df_temp[target_col].map(
            {"Yes": 1, "No": 0}
        )
    else:
        df_temp["Target"] = df_temp[target_col]

    n_rows = (len(cat_cols) + 1) // 2

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, 4 * n_rows)
    )

    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cat_cols):

        churn_rate = (
            df_temp.groupby(col)["Target"]
            .mean()
            .reset_index()
        )

        churn_rate["Target"] *= 100

        sns.barplot(
            data=churn_rate,
            x=col,
            y="Target",
            palette="Reds_d",
            ax=axes[i]
        )

        axes[i].set_title(f"Churn Rate by {col}")
        axes[i].set_ylabel("Churn Rate (%)")
        axes[i].tick_params(axis="x", rotation=30)

        for p in axes[i].patches:

            axes[i].annotate(
                f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=8
            )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_numerical_histograms(df, num_cols, save_path=None):
    """
    Plot histograms for numerical features.
    """

    fig, axes = plt.subplots(
        1,
        len(num_cols),
        figsize=(5 * len(num_cols), 4)
    )

    axes = np.atleast_1d(axes)

    for i, col in enumerate(num_cols):

        sns.histplot(
            data=df,
            x=col,
            bins=30,
            kde=True,
            color="steelblue",
            ax=axes[i]
        )

        axes[i].set_title(col)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_numerical_boxplots(
    df,
    num_cols,
    target_col="Churn",
    save_path=None
):
    """
    Plot boxplots for numerical features.
    """

    fig, axes = plt.subplots(
        1,
        len(num_cols),
        figsize=(5 * len(num_cols), 4)
    )

    axes = np.atleast_1d(axes)

    for i, col in enumerate(num_cols):

        sns.boxplot(
            data=df,
            x=target_col,
            y=col,
            palette="Set2",
            ax=axes[i]
        )

        axes[i].set_title(f"{col} vs {target_col}")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_correlation_heatmap(df, save_path=None):
    """
    Plot correlation heatmap.
    """

    numeric_df = df.select_dtypes(include=np.number)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


if __name__ == "__main__":

    from data_loader import load_and_clean_data

    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

        # Folder lưu hình
    figure_dir = os.path.join(
        base_dir,
        "outputs",
        "figures"
    )

    os.makedirs(figure_dir, exist_ok=True)

    data_path = os.path.join(
        base_dir,
        "data",
        "processed",
        "cleaned_data.csv"
    )

    try:

        df = load_and_clean_data(data_path)

        # Target distribution
        plot_target_distribution(
            df,
            save_path=os.path.join(
                figure_dir,
                "target_distribution.png"
            )
        )

        # Distribution of categorical variables
        categorical_cols = [
            "gender",
            "Partner",
            "Dependents",
            "SeniorCitizen",
            "Contract",
            "InternetService",
            "PaymentMethod",
            "PaperlessBilling"
        ]

        plot_categorical_distribution(
            df,
            categorical_cols,
            save_path=os.path.join(
                figure_dir,
                "categorical_distribution.png"
            )
        )

        # Churn rate by category
        plot_churn_rate_by_category(
            df,
            categorical_cols,
            save_path=os.path.join(
                figure_dir,
                "churn_rate_by_category.png"
            )
        )

        # Histograms
        numerical_cols = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        plot_numerical_histograms(
            df,
            numerical_cols,
            save_path=os.path.join(
                figure_dir,
                "numerical_histograms.png"
            )
        )

        # Boxplots
        plot_numerical_boxplots(
            df,
            numerical_cols,
            save_path=os.path.join(
                figure_dir,
                "numerical_boxplots.png"
            )
        )

        # Correlation heatmap
        plot_correlation_heatmap(
            df,
            save_path=os.path.join(
                figure_dir,
                "correlation_heatmap.png"
            )
        )

    except Exception as e:
        print(f"Error: {e}")
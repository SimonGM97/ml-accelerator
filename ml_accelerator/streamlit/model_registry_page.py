from ml_accelerator.config.params import Params
from ml_accelerator.utils.datasets.data_helper import DataHelper
from ml_accelerator.modeling.models.model import Model
from ml_accelerator.modeling.model_registry import ModelRegistry
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import auc
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from PIL import Image
import pandas as pd
import numpy as np
import os
from typing import List, Tuple


# @st.cache_resource
def find_models(
    _model_registry: ModelRegistry
) -> List[Model]:
    """
    Method utilized to extract the production, staging & development models from the ModelRegistry.

    :param `_model_registry`: (ModelRegistry) Instance from the ModelRegistry class.
        - Note: Streamlit will not hash arguments with an underscore in the argument's name in the 
          function signature.

    :return: (List[Model]) List of production, staging & development models.
    """
    return (
        [_model_registry.load_prod_model(light=True)] 
        + _model_registry.load_staging_models(light=True)
        # + _model_registry.load_dev_models(light=True)
    )


@st.cache_data
def load_test_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load DataHelper
    DH: DataHelper = DataHelper()

    # Load complete X & y datasets
    X = DH.load_dataset(df_name=f'X_{Params.TRANSFORMERS_STEPS[-1]}')
    y = DH.load_dataset(df_name=f'y_{Params.TRANSFORMERS_STEPS[-1]}')

    # Divide into X_train, X_test, y_train & y_test
    _, X_test, _, y_test = DH.divide_datasets(X=X, y=y)
    
    # Return test datasets
    return X_test, y_test


def extract_metric_col_name(metric_name: str) -> str:
    return metric_name.replace("_", " ").title()


def show_models_table(
    models: List[Model]
) -> pd.DataFrame:
    """
    Function that will render a DataFrame with summary information of the models stored in the ModelRegistry.

    :param `models`: (List[Model]) List of development, staging & production models.

    :return: (pd.DataFrame) DataFrame with summary information of the models stored in the ModelRegistry.
    """
    # Define row
    row0, row1, row2 = st.columns([1, 8, 1])

    # Find Summary Models DF
    models_df = pd.DataFrame({
        'Model ID': [model.model_id for model in models],
        'Version': [model.model_id for model in models],
        'Stage': [model.stage for model in models],
        'Algorithm': [model.algorithm for model in models]
    })

    # Add metrics
    metric_cols = []
    for metric_name in models[0].metric_names:
        col_name = extract_metric_col_name(metric_name)
        models_df[col_name] = [getattr(model, metric_name) for model in models]
        metric_cols.append(col_name)

    # Add validation column
    models_df['Optimization Metric'] = [extract_metric_col_name(model.optimization_metric) for model in models]
    models_df['Validation'] = [model.val_score for model in models]

    def highlight_row(row):
        color = 'background-color: white'
        if row['Stage'] == 'production':
            color = 'background-color: #93A3BC'
        return [color] * len(row)

    models_df[metric_cols + ['Validation']] = 100 * models_df[metric_cols + ['Validation']].round(3)

    row1.dataframe(
        models_df.style.apply(highlight_row, axis=1), 
        column_config={
            col: st.column_config.ProgressColumn(
                col,
                format="%.1f%%",
                min_value=0,
                max_value=100
            ) for col in metric_cols
        },
        use_container_width=True,
        hide_index=True
    )

    return models_df


def show_algorithm_fig(
    model: Model
) -> None:
    """
    Function that will render the logo for the model flavor/algorithm that the Model was built on.

    :param `model`: (Model) Model instance.
    """
    # Define columns
    col0, col1, col2 = st.columns([3, 3, 3])

    # Show Ed Machina Logo
    col1.image(Image.open(os.path.join("resources", "images", f"{model.algorithm}_logo.png")), use_column_width=True)

    # Write empty space
    st.write("#")


def show_metrics(
    model: Model,
    models_df: pd.DataFrame
) -> None:
    """
    Function the will utilize streamlit metrics to render: ROC AUC, F1 Score, Precision, Recall & 
    Accuracy from the inputed model.

    :param `model`: (Model) Model instance to extract metrics from.
    :param `models_df`: (pd.DataFrame) Summary results from ModelRegistry models.
    """
    def show_metric(
        col: DeltaGenerator,
        metric_name: str,
    ):
        # Extract metric column name
        col_name: str = extract_metric_col_name(metric_name)

        # Extract score
        score = round(100 * getattr(model, metric_name), 2)

        # Find mean score (for all models)
        mean_score = round(models_df[col_name].mean(), 2)

        # Claculate delta
        delta = round(score - mean_score, 2)

        # Show metric
        col.metric(
            label=col_name, 
            value=f"{score} %", 
            delta=f"{delta} %"
        )

    # Define cols
    cols = st.columns([1.5] + [2] * len(model.metric_names) + [1])

    # Show metrics
    for col, metric_name in zip(cols[1:-1], model.metric_names):
        show_metric(col=col, metric_name=metric_name)

    # Write empty space
    st.write("#")


def show_performance_plots(
    model: Model
) -> None:
    """
    Function that will build and render a confusion matrix heatmat and a ROC AUC plot.

    :param `model`: (Model) Model instance to extract metrics from.
    """
    # Define Columns
    col0, col1, col2, col3 = st.columns([1, 3, 3, 1])

    # Define X & Y axis names
    y = ['Real Fail', 'Real Pass']
    x = ['Predicted Pass', 'Predicted Fail']

    # Extract TN, FP, FN, TP 
    #   - TN: Predicted Pass - Real Pass
    #   - FP: Predicted Fail - Real Pass
    #   - FN: Predicted Pass - Real Fail (queremos minimizar)
    #   - TP: Predicted Fail - Real Fail (queremos maximizar)
    (tn, fp), (fn, tp) = model.confusion_matrix

    # Prepare Matrix
    plot_matrix = np.array([[fn, tp], [tn, fp]])
    # np.array([[tp, fn], [fp, tn]])

    # Find Confusion matrix
    cm_fig = ff.create_annotated_heatmap(
        z=plot_matrix, 
        x=x, 
        y=y, 
        # annotation_text=z_text, 
        colorscale='Blues',
    )

    cm_fig.update_layout(
        title=f'Confusion Matrix (cutoff: {round(100*model.cutoff, 2)} %)',
        # showlegend=False,
        height=500,
        width=500,
    )

    # Plot Confusion Matrix
    col1.plotly_chart(cm_fig)

    # Find ROC AUC Plot
    roc_auc_fig = px.area(
        x=model.fpr, y=model.tpr,
        title=f'ROC Curve (AUC={auc(model.fpr, model.tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    roc_auc_fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    roc_auc_fig.update_yaxes(scaleanchor="x", scaleratio=1)
    roc_auc_fig.update_xaxes(constrain='domain')

    # Plot ROC AUC Plot
    col2.plotly_chart(roc_auc_fig)

    # Write empty space
    st.write("#")


def show_feature_importance_plots(
    model: Model
) -> None:
    """
    Function that will build and render a feature importance barchart and a shap Beeswarm plot.

    :param `model`: (Model) Model instance to extract metrics from.
    """
    # Define Columns
    col0, col1, col2 = st.columns([1, 5, 1])
    
    # Filter Features
    importance_df = model.feature_importance_df.iloc[:20] # .sort_values(by='importance', ascending=True)

    # Plot feature importance
    fi_fig = px.bar(
        importance_df, 
        x='feature', 
        y='importance',
        color='importance',
        # orientation='h'
    )

    fi_fig.update_layout(
        title="Shap Feature Importance (for top 20 Features)",
        # xaxis={
        #     'title': '',
        #     'tickangle': 45
        # },
        height=600,
        width=1200
    )

    col1.plotly_chart(fi_fig, use_container_width=True)

    # Plot beeswarm
    # if model.shap_values is not None:
    #     # https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137
    #     fig = shap.plots.beeswarm(model.shap_values)
    #     col2.plotly_chart(fig)


def show_3d_predictions(
    model: Model
) -> None:
    """
    Function that will build and render a 3-D plot with inferences from the inputed Model, based on the main
    features, determined by the feature importance.

    :param `model`: (Model) Model instance to extract inferences from.
    """
    pass


def build_model_registry_page() -> None:
    """
    Function that will build and render the Model Registry page for the user to examine registered models.
    """
    # Instanciate Registry
    model_registry: ModelRegistry = ModelRegistry()

    # Load Models
    models: List[Model] = find_models(model_registry)

    # Write Model Registry title
    st.markdown(
        '<p style="font-family:sans-serif; color:#183f59; font-size: 25px; font-weight: bold; text-align: left;"'
        '>Model Registry',
        unsafe_allow_html=True
    )

    # Write a line
    st.write("-----")

    # Show models table
    models_df = show_models_table(models=models)

    # Write an empty space
    st.write("#")

    # Write new inference text
    st.markdown(
        '<p style="font-family:sans-serif; color:#183f59; font-size: 20px; font-weight: bold; text-align: left;"'
        '>Inspect Model',
        unsafe_allow_html=True
    )

    # Write a line
    st.write("-----")

    # Select model_id
    model_ids = [f"{model.model_id} ({model.stage})" for model in models]
    selection = st.selectbox(
        label='model_ids_selection', 
        options=model_ids,
        label_visibility='collapsed',
        placeholder='Choose a Model',
        index=None
    )

    if selection is not None:
        # Extract model id
        selected_model_id = selection.split(' ')[0]

        # Extract model
        selected_model = next(model for model in models if model.model_id == selected_model_id)

        # Show model flavor/algorithm centered fig
        show_algorithm_fig(model=selected_model)

        # Show Metrics
        show_metrics(
            model=selected_model,
            models_df=models_df
        )

        # Show Confusion matrix, ROC AUC plot
        # show_performance_plots(model=selected_model)

        # Show feature importance
        feature_importance_plots_toggle = st.toggle('Show Feature Importance Plots', value=False)
        if feature_importance_plots_toggle:
            show_feature_importance_plots(model=selected_model)

        # Show 3-D plot with top three featuers & color given by the prediction
        # prediction_plots_toggle = st.toggle('Show Prediction Plots', value=False)
        # if prediction_plots_toggle:
        #     show_3d_predictions(model=selected_model)
    
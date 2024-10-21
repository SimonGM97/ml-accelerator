from config.params import Params
from ed_ml.data_processing.data_cleaning import DataCleaner
from ed_ml.modeling.model_registry import ModelRegistry
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import requests
import warnings

warnings.filterwarnings("ignore")


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """
    Cached function that will load raw_df and leverage the DataCleaner to correct the 
    "periodo" column.

    :return: (pd.DataFrame) Raw DataFrame
    """
    # Load raw data    
    df = pd.read_csv(Params.raw_data_path, sep=';')

    # Instanciate DataCleaner
    DC = DataCleaner()

    # Correct periodo
    df = DC.correct_periodo(df=df)

    return df


@st.cache_data
def filter_df(
    df: pd.DataFrame, 
    course_name: str = None, 
    periodo: str = None
):
    """
    Cached function that will filter the input DataFrame, based on the course and and or periodo.

    :param `course_name`: (str) Course name to keep.
    :param `periodo`: (str) Periods to keep.

    :returns: (pd.DataFrame) Filtered DataFrame.
    """
    # Filter by course_name
    if course_name is not None:
        df = df.loc[df['course_name'] == course_name]

    # Filter by periodo
    if periodo is not None:
        df = df.loc[df['periodo'] == periodo]

    # Return filtered df
    return df


@st.cache_data
def generate_new_prediction(
    raw_df: pd.DataFrame
) -> pd. DataFrame:
    """
    Function that will ping the flask endpoint to generate new inferences for the specified raw_df.

    :param `raw_df`: (pd.DataFrame) Raw DataFrame with new observations to generate inferences on.

    :return: (pd.DataFrame) New inferences.
    """
    def first(col: pd.Series):
        return col.values[0]
    
    # Run a new request to served model
    predictions = requests.post(
        Params.request_url, 
        json=raw_df.replace(np.nan, 'nan').to_dict()
    ).json()
    
    # Create predictions DataFrame
    predictions_df = (
        pd.DataFrame(predictions)
        .rename(columns={'predicted_probability': 'Failing Probability [%]'})
        .sort_values(by=['Failing Probability [%]'], ascending=False)
        # .reset_index(drop=True)
    )

    # Round failing probability
    predictions_df['Failing Probability [%]'] = np.round(
        100 *  predictions_df['Failing Probability [%]'], 1
    )

    # Find Cutoff
    registry = ModelRegistry(load_from_local_registry=True)
    cutoff = registry.prod_model.cutoff

    # Find prediction class
    predictions_df['Predicted class'] = np.where(
        predictions_df['Failing Probability [%]'] >= cutoff * 100,
        'Fail', 'Pass'
    )

    # Define index name
    predictions_df.index.name = 'user_uuid'

    # Prepare additional information
    DC = DataCleaner(df=raw_df)

    cleaned_df = DC.run_cleaner_pipeline()

    append_df = (
        cleaned_df
        .groupby('user_uuid')
        .agg({
            'legajo': first,
            'course_name': first,
            'nota_parcial': 'mean',
            'score': 'mean'
        })
        .rename(columns={
            'legajo': 'Legajo',
            'course_name': 'Course Name',
            'nota_parcial': 'Average Parcial',
            'score': 'Average Activity'
        })
        .round(1)
    )
    
    # Concatenate DataFrames
    predictions_df = pd.concat([predictions_df, append_df], axis=1)

    # Remove null predictions
    predictions_df = predictions_df.loc[predictions_df['Failing Probability [%]'].notna()]

    return predictions_df


def find_donut_chart(
    predictions_df: pd.DataFrame
):
    """
    Function that will build the plotly figure of the donut chart containing counts of "Passes" &
    "Fail" students.

    :param `predictions_df`: (pd.DataFrame) Inferences outputted by the run_new_predictions function.

    :return: (go.Figure) Plotly donut chart figure.
    """
    vals = ['Pass', 'Fail']
    counts = predictions_df['Predicted class'].value_counts()
    tot = counts.sum()
    for v in vals:
        if v not in counts.index:
            counts.loc[v] = 0
        
        counts.loc[v] = 100 * counts.loc[v] / tot

    fig = go.Figure(go.Pie(
        labels=["Pass", "Fail"],
        values=[
            counts.loc['Pass'],
            counts.loc['Fail']
        ], 
        marker=dict(
            colors=['#62A64E', '#A64E4E'] # '#C60000', '#21C600', '#62A64E', '#A64E4E'
        ),
        sort=False,
        textinfo='percent+label',
        # textposition='outside',
        hole=.4,
        pull=[0.05, 0.05, 0, 0]
    ))

    fig.update_layout(
        title=f'Total Students: {predictions_df.shape[0]}',
        margin=dict(l=0, r=0, t=80, b=0),
        showlegend=False,
        height=220,
        width=220
    )

    return fig


def show_results(
    predictions_df: pd.DataFrame
) -> None:
    """
    Function that will:
        - Create a donut chart with the counts of students predicted as "Pass" and "Fail".
        - Create a DataFrame table with the individual predictions for each specified students.
    
    :param `predictions_df`: (pd.DataFrame) Inferences outputted by the run_new_predictions function.
    """
    # Define row
    row0, row1, row2, row3 = st.columns([0.25, 1.5, 4, 1])

    # Find donut chart
    fig = find_donut_chart(predictions_df)

    # Plot donut chart
    row1.plotly_chart(fig)

    # Calculate height
    if predictions_df.shape[0] < 10:
        height = None
    else:
        height = min([int(36.5 * predictions_df.shape[0]), 1095])

    # Configure Prediction & Predicted Category column
    ordered_cols = [
        'Legajo', 'Course Name', 'Average Parcial', 
        'Average Activity', 'Predicted class', 'Failing Probability [%]'
    ]

    # predictions_df['Predicted class'] = predictions_df['Predicted class'].apply(
    #     lambda pred: Image.open(os.path.join("docs", "tick.png")) if pred == 'Pass'
    #     else Image.open(os.path.join("docs", "cross.png"))
    # )

    def highlight_row(row):
        color = 'background-color: white'
        if row['Predicted class'] == 'Fail':
            color = 'background-color: #F6B0B0'
        return [color] * len(row)

    styled_df = predictions_df.style.apply(highlight_row, axis=1)

    row2.data_editor(
        styled_df, 
        column_config={
            "Failing Probability [%]": st.column_config.ProgressColumn(
                "Failing Probability [%]",
                help="Predicted probability of failing the course.",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            )
        },
        hide_index=True,
        height=height,
        column_order=ordered_cols
    )


def find_options(
    df: pd.DataFrame, 
    col_name: str
) -> list:
    """
    Function that will find the unique values in a specific column; returning the results in a 
    sorted list.

    :param `df`: (pd.DataFrame) DataFrame to find options from.
    :param `col_name`: (list) Column name where the options are to be found.

    :return: (list) List of unique options, sorted by the amount of observations.
    """
    return (
        df
        .groupby(col_name)
        [col_name]
        .count()
        .sort_values(ascending=False)
        .index
        .tolist()
    )


def build_inference_page() -> None:
    """
    Function that will build and render the Inference page for the user to make new inferences on specific
    courses, stundents and partitions.
    """
    # Load cleaned dataset from cache
    raw_df = load_raw_data()

    # Define second row
    row10, row11, row12, row13, row14 = st.columns([3, 3, 1, 3, 1])

    # Write course selection text
    row11.markdown(
        '<p style="font-family:sans-serif; color:#183f59; font-size: 20px; text-align: left;"'
        '>Select Course:',
        unsafe_allow_html=True
    )

    # Write period selection text
    row13.markdown(
        '<p style="font-family:sans-serif; color:#183f59; font-size: 20px; text-align: left;"'
        '>Select Period:',
        unsafe_allow_html=True
    )

    # Define third row
    row20, row21, row22, row23, row24 = st.columns([3, 3, 1, 3, 1])

    # Write new inference text
    row20.markdown(
        '<p style="font-family:sans-serif; color:#183f59; font-size: 30px; font-weight: bold; text-align: left;"'
        '>New Inference',
        unsafe_allow_html=True
    )

    # Select course_name
    course_names = find_options(df=raw_df, col_name='course_name')
    course_name = row21.selectbox(
        label='course_name_selection', 
        options=course_names,
        label_visibility='collapsed',
        placeholder='Select a course',
        index=None
    )

    # Select periodo
    periodos = find_options(df=raw_df, col_name='periodo')
    periodo = row23.selectbox(
        label='period_selection', 
        options=periodos, 
        label_visibility='collapsed',
        placeholder='Select a period (Optional)',
        index=None
    )

    # Write a line
    st.write("-----")

    # Filter raw_df
    raw_df = filter_df(df=raw_df, course_name=course_name, periodo=periodo)

    # Select Users
    toggle = st.toggle('Select specific students', value=False)
    if toggle:
        legajos = st.multiselect(
            label='selected_legajos', 
            options=raw_df['legajo'].unique().tolist(), 
            default=None, 
            label_visibility='collapsed'
        )
        
        # Filter raw_df, based on selected users
        raw_df = raw_df.loc[raw_df['legajo'].isin(legajos)]
    
    # Select particion
    toggle = st.toggle('Select specific partition', value=False)
    if toggle:
        min_p, max_p = int(raw_df['particion'].min()), int(raw_df['particion'].max())
        partition = st.slider(
            label='selected_partition',
            min_value=min_p,
            max_value=max_p,
            value=max_p,
            step=1,
            label_visibility='collapsed'
        )
        
        # Filter raw_df, based on selected partition
        raw_df = raw_df.loc[raw_df['particion'] <= partition]

    # Blank space
    st.write("#")

    # Define fifth row
    row40, row41, row42 = st.columns([3, 3, 10])

    # Inference button
    run_inference = row40.button(label='Run Inference')

    # Blank space
    st.write("#")

    # Run Inference Pipeline
    if run_inference:
        # Find predictions_df
        predictions_df = generate_new_prediction(raw_df=raw_df)
        
        # Show DataFrame
        show_results(predictions_df)

        # Download predictions
        row41.download_button(
            label='Download Predictions',
            data=predictions_df.to_csv(index=False).encode('utf-8'),
            file_name=f'{course_name}_{periodo}_inferences.csv'
        )

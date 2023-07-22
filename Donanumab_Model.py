# Import packages
from ASDM.ASDM import Structure
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
pd.options.display.max_columns = 100

# Function to load the model
def load_model(model_path):
    # Error handling for loading the model
    try:
        model = Structure(from_xmile=model_path)
    except FileNotFoundError:
        st.error(f"File {model_path} not found.")
        return None
    return model

# Function to simulate the model
def simulate_model(model, progression_val):
    # Clearing the last run and compiling exporting the result
    model.clear_last_run()

    # Now tweak a variable
    model.aux_equations['Risk_of_progressing_MCI_to_diagnosis_pa'] = progression_val

    model.simulate()
    results = model.export_simulation_result()

    # Convert the 'time' units
    results['Years'] = [year * 1 for year in results['Years']]

    return results


# Function to plot the results with custom x-axis ticks
def plot_results(df):
    df_outcomes = df[["Percent_change_in_late_stage_dementia_prevalence", "Percent_change_in_prevalence_of_diagnosed_dementia", "Percent_change_in_early_stage_diagnosed_dementia_prevalence"]]
    df_time = df[['Years']]

    # Calculate percent change from baseline (100)
    df_outcomes = df_outcomes - 100

    # Now specify the columns directly
    fig = go.Figure()

    # Add traces for each outcome
    for column in df_outcomes.columns:
        fig.add_trace(go.Scatter(x=df['Years'], y=df_outcomes[column], mode='lines', name=column))

    # Set the x-axis title
    fig.update_xaxes(title_text='Years', type='linear')

    # Set the y-axis title
    fig.update_yaxes(title_text='Percent Change (%) in Prevalence')

    # Customize x-axis ticks and labels in 10-year and 20-year increments
    max_years = int(df['Years'].max())
    x_ticks = list(range(0, max_years + 1, 3))
    x_labels = [f'{year} years' for year in x_ticks]

    fig.update_xaxes(ticks='outside', tickvals=x_ticks, ticktext=x_labels)

    # Move the legend below the plot
    fig.update_layout(legend=dict(orientation='h', x=0, y=-0.25))

    return fig



# Streamlit App
st.title('Donanumab Model')
st.subheader('Slide the slider to alter the yearly risk of progression')

# Load the model
model = load_model('models/Donanemab impact.stmx')

# Set up your sliders
if model is not None:
    # Set the default value of the progression slider to 0.1 using st.sidebar.write
    st.sidebar.write('Risk of progressing MCI to diagnosis per year')
    progression_slider = st.sidebar.slider('',
                                           min_value=0.05,
                                           max_value=0.95,
                                           value=0.25,
                                           step=0.05,
                                           format="%.2f",
                                           key="progression_slider")

    # Simulate the model and plot the results
    results = simulate_model(model, str(progression_slider))
    df = pd.DataFrame(results)

    # Display the results plot
    fig = plot_results(df)
    st.plotly_chart(fig)

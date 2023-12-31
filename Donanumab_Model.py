# Import packages
from ASDM.ASDM import Structure
import pandas as pd
import numpy as np
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

def _run_simulation(model, initial_time, current_time, dt, sim_time, time_units):
    # Set the simulation specs
    model.sim_specs['initial_time'] = initial_time
    model.sim_specs['current_time'] = current_time
    model.sim_specs['dt'] = dt
    model.sim_specs['simulation_time'] = sim_time
    model.sim_specs['time_units'] = time_units

    # Clear the previous run
    model.clear_last_run()

    # Run the simulation
    model.simulate()

    # Get the results
    results = model.export_simulation_result()
    results = pd.DataFrame.from_dict(results)
    results['Years'] = results.index * 1

    return results

def update_model(model, uptake_mci, uptake_early_ad):
    # Clear the previous run
    model.clear_last_run()

    # Update variables
    model.replace_element_equation('Uptake_of_Donanemab_MCI', uptake_mci)
    model.replace_element_equation('Uptake_of_Donanemab_early_stage_AD', uptake_early_ad)

    # Run the simulation for the first 5 years
    results = _run_simulation(model, initial_time=0, current_time=0, dt=0.25, sim_time=20, time_units='Years')

    return pd.DataFrame(results)

def plot_simulation(df):
    # Initialize the plot with the first set of data
    p = figure(width=800, height=250)
    p.line(x='Years', y='Diagnosed_early_stage_AD', source=df, line_width=2)

    # Update the plot with new data from each subsequent simulation
    def update_plot(attr, old, new):
        # Get new data
        df_new = update_model(model, uptake_mci_slider.value, uptake_ad_slider.value)

        # Update data source for plot
        p.line(x='Years', y='Diagnosed_early_stage_AD', source=df_new, line_width=2)

    # Add interactivity
    uptake_mci_slider.on_change('value', update_plot)
    uptake_ad_slider.on_change('value', update_plot)

    return p


# Function to plot the results with custom x-axis ticks
def plot_results(df, fig):
    df_outcomes = df[["Percent_change_in_late_stage_dementia_prevalence", "Percent_change_in_prevalence_of_diagnosed_dementia", "Percent_change_in_early_stage_diagnosed_dementia_prevalence"]]

    # Calculate percent change from baseline (100)
    df_outcomes = df_outcomes - 100

    # Adjust the 'Years' data
    df['Adjusted Years'] = df['Years'] + 3.5

    # Add traces for each outcome using the adjusted years
    for column in df_outcomes.columns:
        fig.add_trace(go.Scatter(x=df['Adjusted Years'], y=df_outcomes[column], mode='lines', name=column))

    # Set the x-axis title
    fig.update_xaxes(title_text='Adjusted Years (Original Year 0 as Year 3.5)', type='linear')

    # Set the y-axis title
    fig.update_yaxes(title_text='Percent Change (%) in Prevalence')

    # Customize x-axis ticks and labels to reflect the 3.5-year adjustment
    max_adjusted_years = int(df['Adjusted Years'].max())
    x_ticks = list(range(0, max_adjusted_years + 1, 3))
    x_labels = [f'{year} years' if year != 3.5 else '3.5 years' for year in x_ticks]

    fig.update_xaxes(ticks='outside', tickvals=x_ticks, ticktext=x_labels)

    # Set the x-axis range explicitly to cover the entire simulation period
    fig.update_xaxes(range=[3.5, max_adjusted_years])

    # Move the legend below the plot
    fig.update_layout(legend=dict(orientation='h', x=0, y=-0.25))

    # Add title to the chart
    fig.update_layout(title_text='Impact of Donanemab on Dementia Prevalence Over Time')

    return fig

# Streamlit App
st.title('RUM Donanumab Model')
st.subheader('Modelling the potential effects of the introduction of Donanumab on the UK population')

# Description of the App
st.write("Now that Donanemab is on the horizon, it raises questions about its implications for dementia prevalence and the planning of healthcare services. To gain a quick, high-level insight, we've created a simple System Dynamics model. System Dynamics offers a transparent approach using stocks, flows, and reasonable assumptions to conceptualize the progression of dementia and explore the potential impact of Donanemab.")

st.write("In this model, we simulate a population of 100,000 individuals, with the UK average prevalence for dementia (approximately 1.4%). The population is divided equally between mild cognitive impairment (MCI), diagnosed dementia with mild symptoms, and severe dementia. We've assumed a 50% risk of progression each year, resulting in an overall time from diagnosis to death of 4 years. Our initial model aims to explore the impact of introducing Donanemab under the assumption that other factors remain constant.")

# Load the model
model = load_model('models/Donanemab impact.stmx')

# Create the go.Figure() object once outside the plot_results function
fig = go.Figure()

# Set up your sliders
if model is not None:
    # Set the default value of the progression slider
    st.sidebar.title('Variable Sliders')

    # Set the default value of the uptake of Donanemab MCI slider
    st.sidebar.write('% Uptake of Donanemab for Mild Cognitive Impairment')
    uptake_mci_slider = st.sidebar.slider('',
                                          min_value=0,
                                          max_value=100,
                                          value=35,
                                          step=5,
                                          format="%.2f",
                                          key="uptake_mci_slider")

    # Set the default value of the uptake of Donanemab early stage AD slider
    st.sidebar.write('% Uptake of Donanemab for Early Stage Alzheimers Disease')
    uptake_ad_slider = st.sidebar.slider('',
                                         min_value=0,
                                         max_value=100,
                                         value=70,
                                         step=5,
                                         format="%.2f",
                                         key="uptake_ad_slider")

    # Simulate the model and plot the results
    results = update_model(model, str(uptake_mci_slider), str(uptake_ad_slider))
    df = pd.DataFrame(results)

    # Display the results plot
    plot_results(df, fig)  # Pass the fig object to the function
    st.plotly_chart(fig)

st.write("We simulate the population over 20 years, introducing Donanemab in year five. The model incorporates a 35% reduction in the risk of progression from MCI to mild and mild to severe dementia, as evidenced in the trial. Additionally, we consider that the drug only applies to Alzheimer's and may not achieve 100% coverage. The model reflects a gradual uptake of Donanemab, taking 3 years to treat 50% of all MCI patients and 70% of people with mild Alzheimer's.")

st.write("The model's initial results show a promising reduction of approximately 10% in the number of people with severe dementia within 3-5 years, as individuals are essentially 'held' in the early stages of the disease. However, over the 20-year period, the system returns to previous levels of severe dementia, and the total number of people with dementia continues to grow. While this model provides an initial insight into the potential impact, there are many refinements that can be made to further explore the implications of Donanemab on dementia prevalence.")

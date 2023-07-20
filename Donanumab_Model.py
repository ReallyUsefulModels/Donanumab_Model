# Import packages

from ASDM.ASDM import Structure
import pandas as pd
import plotly.express as px
import streamlit as st
pd.options.display.max_columns = 100

# Streamlit App
st.title('Donanumab Model')
st.subheader('Slide the slider to alter the yearly risk of progression')

# Now with compiled model
# Compile the model from XMILE

model = Structure(from_xmile='models/Donanemab impact.stmx')

# Set up your sliders
progression_slider = st.sidebar.slider('Risk_of_progressing_MCI_to_diagnosis_pa', 0.05, 0.1, 0.9)
progression_slider = str(progression_slider)

# Clearing the last run and compiling exporting the result
model.clear_last_run()

# Now tweak a variable
model.aux_equations['Risk_of_progressing_MCI_to_diagnosis_pa'] = progression_slider
print(model.aux_equations['Risk_of_progressing_MCI_to_diagnosis_pa'])

#Uptake_of_Donanemab_early_stage_AD = 50
model.simulate()
results = model.export_simulation_result()#

df = pd.DataFrame(results)
# df['Average_wait_for_diagnostic_test'] = pd.to_datetime(df['Average_wait_for_diagnostic_test'])  # Convert the 'time' column to datetime objects

df_outcomes = df[["Percent_change_in_late_stage_dementia_prevalence", "Percent_change_in_prevalence_of_diagnosed_dementia", "Percent_change_in_early_stage_diagnosed_dementia_prevalence"]]
df_time = df[['Years']]

# Now specify the columns directly
st.line_chart(df_outcomes)
# adaptive_threshold_candidate_ml_model

This repository provides a simple machine learning model that identifies which KPI 
datasets are "good candidates" for adaptive threshold training. The model is trained 
on synthetic KPI data and is intended for use with Splunk's Machine Learning Toolkit.

--------------------------------------------------------------------------------
FILE STRUCTURE
--------------------------------------------------------------------------------

adaptive_threshold_candidate_ml_model/
    README.md                - This plain-text readme
    SPL/                    - A much simpler way to make the model, however it's very SPL heavy. You need to tweak the search a lot to make it work with your data.
    data/                   - Synthetic KPI data or CSV files, as well as the python files needed to create the synthetic data.
    model/                  - Contains any saved models (e.g., ONNX files)

--------------------------------------------------------------------------------
OVERVIEW

1. Synthetic Data Generation:
   - Scripts (Python or otherwise) that produce synthetic KPI files labeled as 
     "consistent," "erratic," or "combination."

2. Feature Engineering:
   - SPL searches or Python code that compute metrics like:
     - Hour-of-day standard deviations
     - Weekend vs. weekday patterns
     - Rolling averages and day-of-week trends

3. Model Training:
   - A classification model (RandomForest, neural network, etc.) that learns to 
     classify each KPI into "fit," "maybe_fit," or "not_fit."

4. Splunk Integration:
   - The model can be trained directly in Splunk MLTK using the "fit" command, 
     or externally in Python and imported via ONNX.

--------------------------------------------------------------------------------
USAGE WITH SPLUNK MLTK

- Ingest synthetic CSV data into Splunk.
- Use SPL to transform/aggregate the data into per-KPI features. 
- Train the model with:
  | fit RandomForestClassifier <label> from <features> into <model_name>
- Apply the model to new KPI data:
  | apply <model_name>

--------------------------------------------------------------------------------
EXAMPLE SPL SNIPPET

index="adaptive-threshold_ml_training" sourcetype="csv"
| eval hour_of_day=tonumber(strftime(_time,"%H"))
| stats avg(value) as overall_avg, stdev(value) as overall_stdev by kpi_name
| fit RandomForestClassifier fit_label from overall_avg overall_stdev into "kpi_fitness_model"

--------------------------------------------------------------------------------
LICENSE AND CONTRIBUTIONS

Collaboration and feature requests are welcome.

--------------------------------------------------------------------------------

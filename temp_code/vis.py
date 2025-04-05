import plotly.express as px
import pandas as pd

# Example grid data converted to a DataFrame
data = pd.DataFrame([[-1.045, 2.0, 3.5, -4.890],
                     [-5.678, 3.2, 2.89, 5.78]])

# Create an interactive heatmap
fig = px.imshow(data.values,
                labels=dict(color="Value"))
fig.show()
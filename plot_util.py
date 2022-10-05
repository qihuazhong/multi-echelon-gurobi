import plotly.express as px
import pandas as pd


def plot_gurobi_variables(gurobi_variables, name=''):
    df = pd.DataFrame([[j, k, gurobi_variables[(j, k)].X] for (j, k) in gurobi_variables.keys()],
                      columns=['j', 'k', 'value'])
    fig = px.line(df, x='k', y='value', color='j', title=name)
    fig.show()

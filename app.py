# Dashboard (using Dash for Plotly)
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='age-maxhr-scatter'),
    dcc.Dropdown(id='age-group-filter', options=[
        {'label': '0-30', 'value': '0-30'},
        {'label': '31-45', 'value': '31-45'},
        {'label': '46-60', 'value': '46-60'},
        {'label': '61+', 'value': '61+'}
    ], value='0-30', multi=True, placeholder='Filter by age group')
])

@app.callback(
    Output('age-maxhr-scatter', 'figure'),
    Input('age-group-filter', 'value')
)
def update_graph(selected_age_groups):
    filtered_data = data[data['age_group'].isin(selected_age_groups)]
    fig = px.scatter(filtered_data, x='age', y='maximum_heart_rate', color='target')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
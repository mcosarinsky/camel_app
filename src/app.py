import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import webbrowser
from threading import Timer

def read_entries_from_file(file_path):
    # Initialize an empty list to store the tuples
    data = []

    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read each line in the file
        for line in file:
            # Strip the newline character and split the line by commas
            tuple_values = line.strip().split(',')
            # Convert the split string values to integers and store them as a tuple
            data.append(tuple(map(int, tuple_values)))

    return data


def fibo_retracement(low_swing, high_swing, fibo_level):
    diff = high_swing - low_swing
    
    # Calculate the retracement value for the given fibo_level
    retracement_value = high_swing - (diff * fibo_level)
    return retracement_value


def plot_strat(df, entry):
    A, B, C, D, E, F = entry
    
    start = max(A - 5, 0)
    end = min(F + 5, len(df))
    df = df.loc[start:end, :]
    df.loc[:, 'date'] = pd.to_datetime(df['date'])
    
    return df
    

Data = pd.read_csv('signals_processed.csv')
entries = read_entries_from_file('entries.txt')
entry = entries[0]
A, B, C, D, E, F = entry
df = plot_strat(Data, entry)
    
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    dcc.Dropdown(
        id='fibo-checklist',
        options=[
            {'label': 'fibo level AB 0.786', 'value': 'fibo_AB_786'},
            {'label': 'fibo level AB 0.696', 'value': 'fibo_AB_696'},
            {'label': 'fibo level AB 0.535', 'value': 'fibo_AB_535'},
            {'label': 'fibo level AB 0.618', 'value': 'fibo_AB_618'},
            {'label': 'fibo level CD 0.786', 'value': 'fibo_CD_786'},
            {'label': 'fibo level CD 0.696', 'value': 'fibo_CD_696'},
            {'label': 'fibo level CD 0.535', 'value': 'fibo_CD_535'},
            {'label': 'fibo level CD 0.618', 'value': 'fibo_CD_618'},
            {'label': 'fibo level CD 0.382', 'value': 'fibo_CD_382'}
        ],
        value=['fibo_AB_786'],
        multi=True
    ),
    dcc.Graph(id='candlestick-chart'),
])

def add_fibonacci_trace(fig, df, start_point, end_point, level, label):
    retracement_value = fibo_retracement(df.loc[start_point, 'low'], df.loc[end_point, 'high'], level)
    time_diff = (df['date'].iloc[1] - df['date'].iloc[0]).total_seconds()
    points_offset = 0.4 * time_diff
    fig.add_trace(go.Scatter(
        x=[df['date'].min(), df['date'].max() + 2*pd.Timedelta(seconds=points_offset*5)],
        y=[retracement_value, retracement_value],
        mode='lines',
        line=dict(dash='dot', color='black', width=1),
        opacity=0.5,
        name=label,
        showlegend=False
    ))
    fig.add_annotation(
        x=df['date'].iloc[-10],  
        y=retracement_value,
        text=label,
        showarrow=False,
        xanchor='left',
        yanchor='top',
        font=dict(
            color='black'
        )
    )

@app.callback(
    Output('candlestick-chart', 'figure'),
    [Input('fibo-checklist', 'value')]
)

def update_chart(fibo_levels):
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        showlegend=False
    )])

    # Set offset for placing fractals
    price_range = df['high'].max() - df['low'].min()
    offset = price_range * 0.03
    
    fig.add_trace(go.Scatter(
        x=df.loc[df['fractal_up'] == 1, 'date'],
        y=df.loc[df['fractal_up'] == 1, 'high'] + offset,  # Adjust to position the marker above the candle
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=10),
        name='Up Fractal'
    ))

    # Add red triangles for fractal_down
    fig.add_trace(go.Scatter(
        x=df.loc[df['fractal_down'] == 1, 'date'],
        y=df.loc[df['fractal_down'] == 1, 'low'] - offset,  # Adjust to position the marker below the candle
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=10),
        name='Down Fractal'
    ))

    # Set offset for placing points
    time_diff = (df['date'].iloc[1] - df['date'].iloc[0]).total_seconds()
    points_offset = 0.4 * time_diff
    
    # Add markers for A, C, E placed on low
    fig.add_trace(go.Scatter(
        x=[df.loc[A, 'date'] - pd.Timedelta(seconds=points_offset), df.loc[C, 'date'] - pd.Timedelta(seconds=points_offset), df.loc[E, 'date'] - pd.Timedelta(seconds=points_offset)],
        y=[df.loc[A, 'low'], df.loc[C, 'low'], df.loc[E, 'low']],
        text=['A', 'C', 'E'],
        mode='text',
        textposition='top center',
        showlegend=False
    ))

    # Add markers for B, D placed on high
    fig.add_trace(go.Scatter(
        x=[df.loc[B, 'date'] - pd.Timedelta(seconds=points_offset), df.loc[D, 'date'] - pd.Timedelta(seconds=points_offset)],
        y=[df.loc[B, 'high'], df.loc[D, 'high']],
        text=['B', 'D'],
        mode='text',
        textposition='bottom center',
        showlegend=False
    ))

    # Add marker for F placed on close
    fig.add_trace(go.Scatter(
        x=[df.loc[F, 'date'] - pd.Timedelta(seconds=points_offset)],
        y=[df.loc[F, 'close']],
        text=['F'],
        mode='text',
        textposition='top center',
        showlegend=False
    ))

    # Add Fibonacci levels
    fibo_mapping = {
        'fibo_AB_786': ((A, B), 0.786, 'fibo AB 0.786'),
        'fibo_AB_696': ((A, B), 0.696, 'fibo AB 0.696'),
        'fibo_AB_535': ((A, B), 0.535, 'fibo AB 0.535'),
        'fibo_AB_618': ((A, B), 0.618, 'fibo AB 0.618'),
        'fibo_CD_786': ((C, D), 0.786, 'fibo CD 0.786'),
        'fibo_CD_696': ((C, D), 0.696, 'fibo CD 0.696'),
        'fibo_CD_535': ((C, D), 0.535, 'fibo CD 0.535'),
        'fibo_CD_618': ((C, D), 0.618, 'fibo CD 0.618'),
        'fibo_CD_382': ((C, D), 0.382, 'fibo CD 0.382')
    }

    for fibo_level in fibo_levels:
        if fibo_level in fibo_mapping:
            points, level, label = fibo_mapping[fibo_level]
            add_fibonacci_trace(fig, df, points[0], points[1], level, label) 
    
    # Update layout for better visuals
    fig.update_layout(
        title='Camel strategy',
        yaxis_title='Price',
        xaxis_title='Date',
        xaxis_rangeslider_visible=True,
        height=800
    )

    return fig

if __name__ == "__main__":
    app.run_server(debug=False)


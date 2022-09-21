from statsmodels.tsa.statespace.mlemodel import PredictionResults
from time_series_v13 import main
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
app = dash.Dash(__name__)


def generate_timeline_component(data, name):
    '''
    This function generates a timeline component for the prediction timeline
    Takes in a dataframe and a name
    '''
    timelineFig = px.line(data, x="index",
                          y="Price", template="plotly_dark")
    # timelineFig.update_xaxes(range=[0, 50])

    timelineFig.update_layout(
        xaxis_title="Price Prediction Timeline for the Next 7 Days Sales", yaxis_title="Predicted Price")

    timelineComponent = html.Div(
        className="chart-container vh-60",
        children=[
            html.H3(children='Price Prediction of Selected Sneaker',),

            html.Div(id="subheader",
                     children=f'''Price prediction for the sales average over the next 7 days for the {name}''',),

            dcc.Graph(
                id='price-graph',
                figure=timelineFig,
            ),
            # dcc.RangeSlider(
            #     id='timeline-slider',
            #     min=0,
            #     max=len(data)-1,
            #     step=1,
            #     # value=[data.index[0], data.index[-1]],
            # )
        ])
    return timelineComponent


def generate_social_volume_component(data):
    '''
    This function generates a social volume component for the prediction timeline
    Takes in a dataframe
    '''
    socialFig = px.histogram(data, y="Count", x="index",
                             template="plotly_dark", labels={"Time(GMT)": "Time(GMT)", "Subreddit": ""})

    socialFig.update_yaxes(title="Social Media Volume")

    socialComponent = html.Div(
        className="chart-container vh-40",
        children=[
            html.Div(id="redditHeader",
                     children='''
            Twitter & Reddit Volume Data''',),

            dcc.Graph(
                className="chart",
                id='reddit-graph',
                figure=socialFig,
            ),
        ])

    return socialComponent


def generate_sentiment_component(data):
    '''
    This function generates a sentiment component for the prediction timeline
    Takes in a dataframe
    '''
    sentimentChart = px.pie(data, values="Sentiment",
                            names="Sentiment Type", template="plotly_dark")

    sentimentComponent = html.Div(
        className="chart-container vh-40",
        children=[
            html.Div(id="sentimentHeader",
                     children='''
            Social Media Sentiment''',),

            dcc.Graph(
                className="chart",
                id='sentiment-graph',
                figure=sentimentChart,
            ),
        ])

    return sentimentComponent


def generate_sentiment_callouts(most_positive, most_negative):

    positiveCompoent = html.Div(
        className="chart-container vh-10 w48",
        children=[html.Div(id="positiveHeader", children='''Most Positive Social Media Content'''), dcc.Markdown(f'''*{most_positive["text"]}* - {most_positive["user"]}''')])

    negativeCompoent = html.Div(
        className="chart-container vh-10 w48",
        children=[html.Div(id="negativeHeader", children='''Least Positive Social Media Content'''), dcc.Markdown(f'''*{most_negative["text"]}* - {most_negative["user"]}''')])

    sentimentCalloutContainer = html.Div(
        className="sentiment-row",
        children=[positiveCompoent, negativeCompoent])

    return sentimentCalloutContainer

# Core Components


def generate_sidebar(social_volume_data, sentiment_data):

    sidebarComponent = html.Div(
        className="sidebar",
        children=[
            generate_social_volume_component(social_volume_data),
            generate_sentiment_component(sentiment_data),
        ])
    return sidebarComponent


def generate_main_content(prediction_data, name, sentiment_callouts_data):

    mainContentComponent = html.Div(
        className="main-container two-thirds",
        children=[generate_timeline_component(prediction_data, name), generate_sentiment_callouts(sentiment_callouts_data["most_positive"], sentiment_callouts_data["most_negative"])])

    return mainContentComponent


def generate_content_component(prediction_data, name, sentiment_data, social_volume_data, sentiment_callout_data):

    contentComponent = html.Div(
        className="content",
        children=[generate_main_content(prediction_data, name, sentiment_callout_data), generate_sidebar(social_volume_data, sentiment_data)])

    return contentComponent


headerComponent = html.Div(
    className="header",
    children=[




        html.A(
            html.Img(
                src=app.get_asset_url("logo.jpg"),
                className="logo-img",
            ),
        ),

        html.H1(children='Sneaker Price Prediction Dashboard',),

        html.A(
            html.Img(
                src=app.get_asset_url("homeIcon.png"),
                className="logo-img invert",
            ),
        ),

    ]

)


# Dashboard


layout = html.Div(
    id="root",
    children=[headerComponent]
)

app = dash.Dash(__name__)

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


app.layout = url_bar_and_content_div


app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout,
])


homeLayout = html.Div(
    id="root",
    children=[headerComponent,
              html.Div(


                  [dcc.Input(
                      id="input",
                      type="text",
                      placeholder="Enter StockX URL",
                  )] + [html.Div(id="out-all-types"),
                        dcc.Link('Submit', id="submitbtn", href='/page-1', className="submit-button")],

                  className="input-container", ),

              ]

)


@ app.callback(
    Output("submitbtn", "href"),
    [Input("input", "value")],
)
def cb_render(value):
    print(value)
    return f"/prediction/{value.split('/')[-1]}"


# Index callbacks
@ app.callback(Output('page-content', 'children'),
               Input('url', 'pathname'))
def display_page(pathname):
    print("pathname", pathname)
    if pathname == '/':
        return homeLayout
    elif "prediction" in pathname:

        stockxUrl = f"https://stockx.com/{pathname.split('prediction/')[1]}"
        regressionResult = main(stockxUrl)
        print(regressionResult)
        initialPrice = 290
        pctChangeList = regressionResult[0].to_list()
        prices = []
        prices.append(initialPrice)
        for i in range(len(pctChangeList)):
            prices.append(prices[-1]*(1+pctChangeList[i]))

        predictions = regressionResult[0].to_frame().reset_index()
        predictions["Price"] = prices[1:]
        print(predictions)

        name = regressionResult[3]

        regressionResult[1]["created_at"] = pd.to_datetime(
            regressionResult[1]["created_at"])
        regressionResult[2]['Time(GMT)'] = pd.to_datetime(
            regressionResult[2]['Time(GMT)'])
        regressionResult[1].set_index("created_at", inplace=True)
        regressionResult[2].set_index("Time(GMT)", inplace=True)
        resampledR1 = regressionResult[1].resample('1D').count()[["Name"]]
        resampledR2 = regressionResult[2].resample('1D').count()[["Key"]]
        resampledR1.rename(columns={"Name": "Count"}, inplace=True)
        resampledR2.rename(columns={"Key": "Count"}, inplace=True)

        resampledR1 = resampledR1.tz_convert(None)
        combinedVolumeData = resampledR1.append(resampledR2)
        combinedVolumeData.reset_index(inplace=True)

        regressionResult[2].rename(columns={"Body": "text"}, inplace=True)
        combinedSocialData = regressionResult[1].append(regressionResult[2])
        combinedSocialData.reset_index(inplace=True)
        mostPositive = combinedSocialData.iloc[combinedSocialData["joy"].idxmax(
        )]["text"]
        leastPostive = combinedSocialData.iloc[combinedSocialData["sadness"].idxmax(
        )]["text"]
        most_positive_example = {
            'text': f'''{mostPositive}''', 'user': ''}

        most_negative_example = {
            'text': f'{leastPostive}', 'user': ''}

        combinedSocialData["IsPositive"] = combinedSocialData.apply(
            lambda x: 1 if x["joy"] > x["sadness"] else 0, axis=1)
        percentPositive = combinedSocialData["IsPositive"].sum(
        )/len(combinedSocialData)
        percentNegative = 1-percentPositive
        sentiment = pd.DataFrame([[percentPositive, "Positive"], [percentNegative, "Negative"]], columns=[
            "Sentiment", "Sentiment Type"])

    else:
        # Prediction Data
        predictions = pd.read_csv("PredictionData.csv")
        name = "adidas Yeezy Boost 350 V2 Beluga Reflective"

        # Social Volume Data
        redditData = pd.read_csv('redditData.csv')
        redditData["Time(GMT)"] = pd.to_datetime(redditData["Time(GMT)"])
        redditData = redditData.set_index("Time(GMT)")
        combinedVolumeData = (redditData.resample('1D').count()[["Subreddit"]])
        combinedVolumeData.reset_index(inplace=True)
        combinedVolumeData.rename(columns={"Subreddit": "Count"}, inplace=True)
        combinedVolumeData["Time(GMT)"] = pd.to_datetime(
            combinedVolumeData["Time(GMT)"])
        print("Combined Volume Dataa", combinedVolumeData)
        combinedVolumeData.reset_index(inplace=True)
        # Sentiment Callouts
        most_positive_example = {
            'text': '''I need those beluga reflective yeezys.''', 'user': 'Shayla.'}

        most_negative_example = {
            'text': 'That beluga reflective isn’t the same.', 'user': 'BeXar'}

    # Sentiment Data
        sentiment = pd.DataFrame([[0.81, "Positive"], [0.19, "Negative"]], columns=[
            "Sentiment", "Sentiment Type"])

    callout_data = {"most_positive": most_positive_example,
                    "most_negative": most_negative_example}

    newLayout = html.Div(
        id="root",
        children=[headerComponent, generate_content_component(predictions, name, sentiment, combinedVolumeData, callout_data)])

    return newLayout


app.title = "Sneaker Pricer"


if __name__ == '__main__':
    app.run_server(debug=False)

from dataclasses import dataclass
import json
from flask import Flask, request, render_template, make_response, Response
import numpy as np
import pandas as pd
from Preprocessor import Preprocessor
from SearchUtil import SearchUtil
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from Preprocessor import Preprocessor
from SearchUtil import SearchUtil
from transformers import AutoTokenizer, AutoModel


app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("LaBSE-en-ru")
model = AutoModel.from_pretrained("LaBSE-en-ru")
with open('matrix_embedding_norm.npy', 'rb') as f:
    matrix = np.load(f)
df = pd.read_csv('./full_data.csv', sep=',')
p = Preprocessor()
s = SearchUtil(matrix, model, tokenizer)


def sort_df(index: int, df: pd.DataFrame, matrix: np.ndarray, p: Preprocessor, s: SearchUtil, accuracy: float = 0.85):
    query = df['product_name'].fillna('')[index] + df['okpd2_name'].fillna(
        '')[index] + df['product_characteristics'].fillna('')[index]
    query = p.clean_text(query, remove_duplicates=True)
    matrix_df = pd.DataFrame(s.sort(query)[:, :2], columns=['id', 'cos_dist'])
    matrix_df = matrix_df.astype({"id": int})
    sdf = df.loc[matrix_df['id']]
    sdf['cos_dist'] = matrix_df['cos_dist']
    # sdf.loc[matrix_df['cos_dist'] > accuracy]
    #
    #
    return sdf[(sdf['cos_dist'] > accuracy)]


def top_providers(index: int, sdf: pd.DataFrame):
    providers = sdf.groupby('inn').count()
    providers = providers.sort_values(
        by='product_vat_rate', ascending=False)[0:10]
    ax = providers['product_vat_rate'].plot(kind='bar')

    plt.xlabel('Поставщики')
    plt.ylabel('Количество сделок')
    plt.title('Топ поставщиков по заказам')
    plt.xticks(rotation=90)
    file = StringIO()
    plt.savefig(file, format='svg', bbox_inches='tight')
    return file.getvalue()


def other_products(index: int, df: pd.DataFrame):
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Histogram(x=df[df['inn'] == df.loc[index]['inn']]
                  ['product_vat_rate'], histnorm='percent', name='Учет НДС'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df[df['inn'] == df.loc[index]['inn']]['okpd2_name'].value_counts(
    ).index.tolist()[:10], histnorm='percent', name='Название'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df[df['inn'] == df.loc[index]['inn']]
                  ['country_name'], histnorm='percent', name='Название'), row=1, col=1)

    fig.update_layout(
        title="Другие товары поставщика",
        title_x=0.5,
        legend=dict(x=0.5, y=0.5, xanchor="auto", orientation="h"),
        margin=dict(l=0, r=0, t=50, b=0))

#     fig.show()
    file = StringIO()
    fig.write_html(file)
    return file.getvalue()


def statistics(index: int, sdf: pd.DataFrame):
    statDict1 = {'MeanPrice': round(sdf[sdf['inn'] == sdf.loc[index]['inn']]['price'].mean(), 2),
                 'MedianPrice': round(sdf[sdf['inn'] == sdf.loc[index]['inn']]['price'].median(), 2),
                 'MinPrice': round(sdf[sdf['inn'] == sdf.loc[index]['inn']]['price'].min(), 2),
                 'MaxPrice': round(sdf[sdf['inn'] == sdf.loc[index]['inn']]['price'].max(), 2),
                 'StdPrice': round(sdf[sdf['inn'] == sdf.loc[index]['inn']]['price'].std(), 2)}

    new_df = pd.DataFrame()
    for i in range(0, 10):
        new_df = new_df.append(
            sdf[sdf['inn'] == sdf['inn'].value_counts().index.tolist()[:10][i]])

    statDict2 = {'MeanPrice': round(new_df['price'].mean(), 2),
                 'MedianPrice': round(new_df['price'].median(), 2),
                 'MinPrice': round(new_df['price'].min(), 2),
                 'MaxPrice': round(new_df['price'].max(), 2),
                 'StdPrice': round(new_df['price'].std(), 2)}

    return statDict1, statDict2


def concurent_graph(sdf: pd.DataFrame):
    new_df = pd.DataFrame()
    for i in range(0, min(10, sdf.shape[0])):
        new_df = new_df.append(sdf[sdf['inn'] == sdf['inn'].value_counts(
        ).index.tolist()[:min(10, sdf.shape[0])][i]])

    fig = make_subplots(rows=2, cols=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        title="Другие товары поставщика",
        title_x=0.5
    )
    fig = px.bar(new_df, x='price', y='inn',
                 hover_data=['okpd2_name', 'inn'], color='product_name',
                 labels={'inn': 'Provider', 'product_name': 'Product Name', 'price': 'Price'}, height=400)

    file = StringIO()
    fig.write_html(file)
    return file.getvalue()


@dataclass
class ProductInfo:
    id: int = 0
    product_name: str = ""
    price: str = ""
    product_vat_rate: str = ""
    product_msr: str = ""
    product_characteristics: str = ""
    okpd2_code: str = ""
    okpd2_name: str = ""
    inn: str = ""
    country_code: str = ""


@app.route("/search", methods=["get"])
def search():
    query = request.args.get("query")
    cleaned_query = p.clean_text(query)
    matrix = s.sort(cleaned_query)
    result = {
        "query": cleaned_query,
        "results": [],
        "analogs": [],
        "similar": []
    }
    step = 3
    for i in df.iloc[np.array(matrix[:step, 0], dtype=int)].index.to_list():
        pi = ProductInfo(
            id=int(df['Unnamed: 0'][i]),
            product_name=str(df['product_name'].fillna('')[i]),
            price=str(df['price'].fillna('')[i]),
            product_vat_rate=str(df['product_vat_rate'].fillna('')[i]),
            product_msr=str(df['product_msr'].fillna('')[i]),
            product_characteristics=str(
                df['product_characteristics'].fillna('')[i]),
            okpd2_code=str(df['okpd2_code'].fillna('')[i]),
            okpd2_name=str(df['okpd2_name'].fillna('')[i]),
            inn=str(df['inn'].fillna('')[i]),
            country_code=str(df['country_name'][i])
        )
        result['results'].append(pi)
    for i in df.iloc[np.array(matrix[step:2*step, 0], dtype=int)].index.to_list():
        pi = ProductInfo(
            id=int(df['Unnamed: 0'][i]),
            product_name=str(df['product_name'].fillna('')[i]),
            price=str(df['price'].fillna('')[i]),
            product_vat_rate=str(df['product_vat_rate'].fillna('')[i]),
            product_msr=str(df['product_msr'].fillna('')[i]),
            product_characteristics=str(
                df['product_characteristics'].fillna('')[i]),
            okpd2_code=str(df['okpd2_code'].fillna('')[i]),
            okpd2_name=str(df['okpd2_name'].fillna('')[i]),
            inn=str(df['inn'].fillna('')[i]),
            country_code=str(df['country_name'][i])
        )
        result['analogs'].append(pi)
    for i in df.iloc[np.array(matrix[2*step:3*step, 0], dtype=int)].index.to_list():
        pi = ProductInfo(
            id=int(df['Unnamed: 0'][i]),
            product_name=str(df['product_name'].fillna('')[i]),
            price=str(df['price'].fillna('')[i]),
            product_vat_rate=str(df['product_vat_rate'].fillna('')[i]),
            product_msr=str(df['product_msr'].fillna('')[i]),
            product_characteristics=str(df['product_characteristics'][i]),
            okpd2_code=str(df['okpd2_code'].fillna('')[i]),
            okpd2_name=str(df['okpd2_name'].fillna('')[i]),
            inn=str(df['inn'].fillna('')[i]),
            country_code=str(df['country_name'][i])
        )
        result['similar'].append(pi)
    # print(df.iloc[np.array(matrix[5:10, 0], dtype=int)])
    # print(df.iloc[np.array(matrix[10:15, 0], dtype=int)])
    # results, analogs, similar = append_other(matrix, df)

    response = make_response(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET, PUT, POST, DELETE')
    response.headers.add('Access-Control-Allow-Headers',
                         'Origin, X-Requested-With, Content-Type, Accept')
    return response


@app.route("/graph", methods=["get"])
def graph():
    idx = request.args.get('id')
    sdf = sort_df(int(idx), df, matrix, p, s)
    d1, d2 = statistics(int(idx), df)
    context = {
        "data": df.loc[int(idx)].to_string(),
        # "g1": top_providers(int(idx), sdf),
        "g1": other_products(int(idx), df),
        "d1": d1.__str__(),
        "d2": d2
        # "g3": concurent_graph(sdf)
    }
    # print(context)
    return render_template('graphs.html', **context)


if __name__ == "__main__":
    app.run(port=3001, debug=False)

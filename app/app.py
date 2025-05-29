import dash
from dash import dcc, html, Input, Output, State
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from embedding.Bert import Bert_Embedding

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 初始化模型与数据
model = Bert_Embedding()
embeddings = np.load('../public/paper_embeddings.npy')
titles = np.load('../public/paper_titles.npy')

# 创建 Dash App
app = dash.Dash(__name__)
server = app.server  # 用于部署

# 自定义背景图（可替换成你自己喜欢的）
background_image_url = "D:\Coding\project\myproject\competition\Lithub_EmbedMind\public\bg.png"

# 内联样式字典
styles = {
    'container': {
        'maxWidth': '1400px',
        'margin': 'auto',
        'padding': '40px',
        'fontFamily': '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
        'color': '#fff',
        'position': 'relative',
        'zIndex': '1'
    },
    'title': {
        'textAlign': 'center',
        'fontSize': '3rem',
        'marginBottom': '30px',
        'textShadow': '2px 2px 5px rgba(0,0,0,0.5)',
        'color': 'rgb(30, 150, 100)'
    },
    'input': {
        'width': '100%',
        'padding': '15px',
        'borderRadius': '10px',
        'border': 'none',
        'marginBottom': '20px',
        'fontSize': '16px',
        'boxSizing': 'border-box',
        'outline': 'none',
        'transition': 'all 0.3s ease-in-out',
        'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.2)'
    },
    'button': {
        'padding': '12px 24px',
        'border': 'none',
        'borderRadius': '8px',
        'cursor': 'pointer',
        'fontSize': '16px',
        'fontWeight': 'bold',
        'transition': 'all 0.3s ease',
        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',
        'marginRight': '10px'
    },
    'search_button': {
        'background': 'linear-gradient(to right, #007bff, #0056b3)',
        'color': 'white'
    },
    'visualize_button': {
        'background': 'linear-gradient(to right, #28a745, #1e7e34)',
        'color': 'white'
    },
    'graph_container': {
        'marginTop': '30px',
        'backgroundColor': 'rgba(255, 255, 255, 0.05)',
        'borderRadius': '10px',
        'boxShadow': '0 6px 12px rgba(0, 0, 0, 0.2)',
        'overflow': 'hidden',
        'color': "black"
    },
    'results_box': {
        'marginTop': '20px',
        'padding': '15px',
        'backgroundColor': 'rgba(255, 255, 255, 0.1)',
        'borderRadius': '10px',
        'whiteSpace': 'pre-line',
        'color': '#333'
    }
}

# 布局：带背景图的全屏容器
app.layout = html.Div([
    # 背景图层
    html.Div(style={
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'width': '100%',
        'height': '100%',
        'background-image': f'url({background_image_url})',
        'background-size': 'cover',
        'background-position': 'center',
        'bachground-repeat': 'no-repeat',
        'filter': 'blur(3px)',
        'zIndex': '0'
    }),

    # 内容容器
    html.Div([
        html.H1("论文语义搜索引擎", style=styles['title']),

        html.Label("输入论文摘要："),
        dcc.Input(
            id='query-input',
            type='text',
            placeholder='请输入论文摘要...',
            style={**styles['input'], 'color': '#333'}
        ),

        html.Div([
            html.Button('搜索相似论文', id='search-button', n_clicks=0,
                        style={**styles['button'], **styles['search_button']}),
            html.Button('可视化聚类', id='visualize-button', n_clicks=0,
                        style={**styles['button'], **styles['visualize_button']})
        ]),

        html.Hr(style={'borderColor': '#ccc'}),

        html.Div(id='results-output', style=styles['results_box']),

        html.Div(id='graph-output', children=[
            dcc.Graph(
                id='tsne-graph',
                figure=go.Figure().update_layout(margin=dict(l=0, r=0, t=40, b=0)),
                style={'height': '700px'}
            )
        ], style=styles['graph_container'])

    ], style= styles['container'])
], style={
    'minHeight': '100vh',
    'position': 'relative',
    'backgroundBlendMode': 'overlay'
})


# 回调函数 - 搜索论文
@app.callback(
    Output('results-output', 'children'),
    Input('search-button', 'n_clicks'),
    State('query-input', 'value')
)
def search_papers(n_clicks, query_text):
    if not query_text or n_clicks == 0:
        return ""
    query_vec = model.encode([query_text])
    similarities = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:5]
    results = [(titles[i], similarities[i]) for i in top_indices]
    result_str = "\n".join([f"{title} (Score: {score:.4f})" for title, score in results])
    return result_str

colors = px.colors.qualitative.Set2
# 回调函数 - 聚类可视化
@app.callback(
    Output('tsne-graph', 'figure'),
    Input('visualize-button', 'n_clicks'),
    State('query-input', 'value')
)
def visualize_clusters(n_clicks, query_text):
    if not query_text or n_clicks == 0:
        return go.Figure()

    try:
        query_vec = model.encode([query_text])
        combined = np.vstack((query_vec, embeddings))
        tsne = TSNE(n_components=3, random_state=42)
        reduced = tsne.fit_transform(combined)

        # 自动选择最佳聚类数（示例范围：2~10）
        max_k = min(10, len(reduced) - 1)  # 避免 k >= 样本数
        best_k = 3  # 默认值
        scores = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42).fit(reduced)
            score = silhouette_score(reduced, kmeans.labels_)
            scores.append((k, score))

        best_k = max(scores, key=lambda x: x[1])[0]

        # 聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42).fit(reduced)
        labels = kmeans.labels_

        # 构建图表
        fig = go.Figure()

        # 绘制每个 cluster
        for cluster_id in range(best_k):
            idxs = np.where(labels == cluster_id)[0]
            fig.add_scatter(
                x=reduced[idxs, 0],
                y=reduced[idxs, 1],
                mode='markers',
                marker=dict(color=colors[cluster_id % len(colors)], size=6),
                name=f'Cluster {cluster_id + 1}'
            )

        # 添加查询点（单独一类）
        fig.add_scatter(
            x=[reduced[0, 0]],
            y=[reduced[0, 1]],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='Query'
        )

        # 设置布局
        fig.update_layout(
            title=f"t-SNE Visualization (K-Means Clustering, K={best_k})",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='black',
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                color='black',
                title='t-SNE 1'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                color='black',
                title='t-SNE 2'
            )
        )

        return fig

    except Exception as e:
        print("Error during visualization:", str(e))
        return go.Figure().add_annotation(text="Error generating plot", showarrow=False)


# 启动服务
if __name__ == '__main__':
    app.run(debug=True, port=5000)
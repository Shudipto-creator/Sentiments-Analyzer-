import gradio as gr
from transformers import pipeline
import torch
import shutil
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

print("Loading model from Hugging Face Hub...")
MODEL_NAME = "facebook/bart-large-mnli"

emotion_classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1,
)
print(f"Model '{MODEL_NAME}' loaded from Hugging Face Hub")
cache_dir = Path(".gradio") / "cached_examples"
if cache_dir.exists():
    shutil.rmtree(cache_dir, ignore_errors=True)
EMOTIONS = [
    "admiration",
    "adoration",
    "aesthetic appreciation",
    "amusement",
    "anger",
    "anxiety",
    "awe",
    "awkwardness",
    "boredom",
    "calmness",
    "confusion",
    "craving",
    "disgust",
    "empathic pain",
    "entrancement",
    "excitement",
    "fear",
    "horror",
    "interest",
    "joy",
    "nostalgia",
    "relief",
    "romance",
    "sadness",
    "satisfaction",
    "sexual desire",
    "surprise",
]
def analyze_emotions(text: str):
    """Return percentage breakdown across the requested emotions plus chart."""
    if not text or not text.strip():
        return "Please enter some text", create_empty_table(), create_empty_chart()

    trimmed = text.strip()[:512]
    result = emotion_classifier(
        trimmed, candidate_labels=EMOTIONS, multi_label=True
    )

    scores = dict(zip(result["labels"], result["scores"]))
    total = sum(scores.values()) or 1.0
    normalized = {label: score / total for label, score in scores.items()}

    dominant = max(normalized, key=normalized.get)
    summary = f"Dominant emotion: {dominant.title()} ({normalized[dominant] * 100:.1f}%)"

    sorted_rows = sorted(
        ((label.title(), round(normalized.get(label, 0) * 100, 2)) for label in EMOTIONS),
        key=lambda row: row[1],
        reverse=True,
    )
    chart = build_emotion_chart(normalized)
    return summary, sorted_rows, chart


def create_empty_table():
    return [[label.title(), 0.0] for label in EMOTIONS]


def create_empty_chart():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(255,255,255,0.95)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        annotations=[
            dict(
                text="Emotion chart will appear here after analysis.",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#475569", family="Arial, sans-serif"),
            )
        ],
        height=420,
    )
    return fig


def build_emotion_chart(score_dict):
    if not score_dict:
        return create_empty_chart()

    sorted_pairs = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
    top_pairs = sorted_pairs[:6]
    labels = [label.title() for label, _ in top_pairs]
    values = [round(score * 100, 2) for _, score in top_pairs]

    colors = [
        "#2563eb", 
        "#1d4ed8", 
        "#3b82f6", 
        "#60a5fa", 
        "#93c5fd", 
        "#dbeafe", 
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "xy"}]],
        subplot_titles=("Radar View", "Bar View"),
    )
    # Radar chart
    radar_labels = labels + labels[:1]
    radar_values = values + values[:1]
    fig.add_trace(
        go.Scatterpolar(
            r=radar_values,
            theta=radar_labels,
            fill="toself",
            name="Top Emotions",
            fillcolor="rgba(37, 99, 235, 0.25)",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=6, color="#2563eb"),
        ),
        row=1,
        col=1,
    )
    # Bar chart
    fig.add_trace(
        go.Bar(
            x=values[::-1],
            y=labels[::-1],
            orientation="h",
            marker=dict(
                color=values[::-1],
                colorscale=[[0, "#dbeafe"], [0.5, "#60a5fa"], [1, "#2563eb"]],
                showscale=True,
                colorbar=dict(title="%", len=0.4, y=0.3),
                line=dict(color="#1d4ed8", width=1),
            ),
            name="Percent",
            text=[f"{v:.1f}%" for v in values[::-1]],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(255,255,255,0.95)",
        plot_bgcolor="rgba(255,255,255,0.95)",
        font=dict(family="Arial, sans-serif", size=11, color="#0f172a"),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(60, max(values) + 10)],
                tickfont=dict(size=10, color="#475569"),
                gridcolor="rgba(148,163,184,0.3)",
                linecolor="#cbd5e1",
            ),
            angularaxis=dict(
                direction="clockwise",
                tickfont=dict(size=10, color="#0f172a"),
                linecolor="#cbd5e1",
            ),
            bgcolor="rgba(255,255,255,0.95)",
        ),
        xaxis=dict(
            title=dict(text="Percent (%)", font=dict(size=12, color="#0f172a")),
            tickfont=dict(size=10, color="#475569"),
            gridcolor="rgba(148,163,184,0.2)",
            linecolor="#cbd5e1",
        ),
        yaxis=dict(
            tickfont=dict(size=10, color="#0f172a"),
            gridcolor="rgba(148,163,184,0.2)",
            linecolor="#cbd5e1",
        ),
    )

    return fig


def clear_fields():
    return "", "", create_empty_table(), create_empty_chart()


examples = [
    ["Zahir Knows a lot of things but he unables to share it with others. but he is a good person."],
    ["The weather is pleaseant in kunming."],
    ["Zahir always gives me a hard time."],
    ["I'm not sure how I feel about this. It has both good and bad aspects."],
    ["The customer service was terrible and the product arrived damaged."],
    ["I appreciate your help, but there is still some confusion."],
]


with gr.Blocks(title="Emotion Analysis") as demo:
    gr.Markdown(
        """
        # Emotion Analysis App

        Enter text and the model will estimate the percentage across 27 nuanced emotions
        (from admiration and amusement to romance, sexual desire, and surprise).
        """
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter text",
                placeholder="Type your text here...",
                lines=5,
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Column():
            sentiment_output = gr.Textbox(label="Summary", interactive=False)
            confidence_output = gr.Dataframe(
                headers=["Emotion", "Percent (%)"],
                datatype=["str", "number"],
                label="Emotion Breakdown (%)",
                interactive=False,
            )
            chart_output = gr.Plot(label="Emotion Comparison")

    gr.Examples(
        examples=examples,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, chart_output],
        fn=analyze_emotions,
        cache_examples=False,
    )

    gr.Markdown(
        """
        ---
        **Model**: `facebook/bart-large-mnli` (zero-shot classification)  
        **Emotions**: Admiration, Adoration, Aesthetic Appreciation, Amusement, Anger, Anxiety, Awe, Awkwardness, Boredom, Calmness, Confusion, Craving, Disgust, Empathic Pain, Entrancement, Excitement, Fear, Horror, Interest, Joy, Nostalgia, Relief, Romance, Sadness, Satisfaction, Sexual Desire, Surprise
        
        ---
        **Developed by**: SUDIPTA ROY
        """
    )

    analyze_btn.click(
        fn=analyze_emotions,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, chart_output],
    )

    text_input.submit(
        fn=analyze_emotions,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, chart_output],
    )

    clear_btn.click(
        fn=clear_fields,
        inputs=None,
        outputs=[text_input, sentiment_output, confidence_output, chart_output],
    )






    
if __name__ == "__main__":
    demo.launch(share=True)


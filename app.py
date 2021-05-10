"""
This script is extremely hacky to get a working visualization together so please don't judge me.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.graph_objs import Layout
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

colors = dict()


def select_color():
    global color_index
    color = cmap(color_index)
    color_index += 1
    return color


num_iters = 44320  # Number of samples in full shakespeare
num_entries = 20
cmap = plt.cm.get_cmap('plasma', num_entries)
color_index = 0


def create_model_size_vs_metric_figure(y_axis,
                                       y_axis_title,
                                       norm_y=None,
                                       scale_y=1,
                                       checkpointing=False):
    names = [250000000, 500000000, 1000000000, 4000000000]
    paths = ["results/250m.csv", "results/500m.csv", "results/1.csv", "results/4.1.csv"]

    data_frames = []
    for path in paths:
        data_frames.append(pd.read_csv(path))

    plugins = zip(data_frames[0]['Plugin'], data_frames[0]['Plugin Alias'])
    layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig = go.Figure(layout=layout)
    for plugin, alias in plugins:
        if checkpointing and 'checkpoint' not in plugin.lower():
            continue
        elif not checkpointing and 'checkpoint' in plugin.lower():
            continue
        x = []
        y = []
        for name, df in zip(names, data_frames):
            if (df['Plugin'] == plugin).any():
                x.append(name)
                val = df[df['Plugin'] == plugin][y_axis].item() / scale_y
                if norm_y:
                    val = norm_y / val
                y.append(val)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                text='plugins=' + alias,
                name=plugin,
                mode='markers+lines',
                textposition='top right',
                showlegend=True,
                line=dict(
                    width=1.25
                )
            )
        )

    fig.update_xaxes(
        title_text='Number of Model Parameters',
        showgrid=True,
        linewidth=0.1,
        linecolor='black',
        gridcolor='rgba(50,50,50,0.1)',
    )

    fig.update_yaxes(
        title_text=y_axis_title if y_axis_title else y_axis,
        title_standoff=25,
        showgrid=True,
        linewidth=0.1,
        linecolor='black',
        gridcolor='rgba(50,50,50,0.05)',
        tickfont_size=12,
    )

    return fig


def plot_model_size_vs_metric_figure(title, y_axis, scale_y, description, norm_y=None, y_axis_title=None):
    st.subheader(title)
    st.text(description)
    checkpointing = st.checkbox(
        'Activation Checkpointing',
        value=True,
        key=title,
        help="Activation Checkpointing, (or commonly known as Gradient Checkpointing) "
             "de-allocates activations computed in the forward pass"
             " to save memory, and are re-computed on demand in the backward pass. "
             "This saves memory for large models that produce large activations.\n"
             "This does require some code changes but provides a substantial memory improvement, "
             "thus worth applying in most large model cases.\n"
             "See [DeepSpeed Activation Checkpointing](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed-activation-checkpointing) or [FairScale Checkpointing](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#fairscale-activation-checkpointing)."
    )
    st.plotly_chart(
        create_model_size_vs_metric_figure(
            y_axis=y_axis,
            y_axis_title=y_axis_title,
            scale_y=scale_y,
            norm_y=norm_y,
            checkpointing=checkpointing
        ),
        use_container_width=True
    )


def create_model_size_specific_figure(
        path,
        x='Time per iteration (s)',
        y='Peak Memory (MiB)',
        scale_y=1024,
        x_axis_title=None,
        y_axis_title=None,
        baseline=None,
        display_plugin_text=('DDP',)):
    df = pd.read_csv(path)

    if baseline:
        # Use factor instead of absolute values
        baseline_df = df[df['Plugin'] == baseline]

        def change_x(df):
            df[x] = baseline_df[x][0] / df[x]
            return df

        def change_y(df):
            df[y] = baseline_df[y][0] / df[y]
            return df

        df = df.groupby('Plugin').apply(change_x)
        df = df.groupby('Plugin').apply(change_y)

    layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig = go.Figure(layout=layout)

    for index, row in df.iterrows():
        plugin = row['Plugin'].lower()
        is_checkpointing = 'checkpoint' in plugin
        text = 'plugins=' + row['Plugin Alias']
        if is_checkpointing:
            text += ' + checkpointing\u00b9'

        color_id = plugin.replace('w checkpointing', '').strip()
        color = colors.get(color_id, select_color())
        colors[color_id] = color
        fig.add_trace(
            go.Scatter(
                x=[row[x]],
                y=[row[y] / scale_y],
                text=text,
                marker_symbol=['star'] if is_checkpointing else None,
                name=row['Plugin'],
                mode='markers+text' if row['Plugin'] in display_plugin_text else 'markers',
                textposition='top right',
                showlegend=False,
                marker=dict(color=color)
            )
        )

    fig.update_xaxes(
        title_text=x_axis_title if x_axis_title else x,
        showgrid=True,
        linewidth=0.1,
        linecolor='black',
        gridcolor='rgba(50,50,50,0.1)',
        ticksuffix='x' if baseline else None
    )

    fig.update_yaxes(
        title_text=y_axis_title if y_axis_title else y,
        title_standoff=25,
        showgrid=True,
        linewidth=0.1,
        linecolor='black',
        gridcolor='rgba(50,50,50,0.05)',
        tickfont_size=12,
        ticksuffix='x' if baseline else None,
        tick0=1 if baseline else None
    )

    fig.update_traces(
        marker_size=10
    )
    return fig


@st.cache()
def create_model_specific_figure(input_path, ddp_relative_supported=True):
    peak_mem_vs_iteration_time_relative = None
    if ddp_relative_supported:
        peak_mem_vs_iteration_time_relative = create_model_size_specific_figure(
            path=input_path,
            x_axis_title='Time per Iteration Improvement Multiplier (higher is better)',
            y_axis_title='Peak Memory Reduction Multiplier (higher is better)',
            baseline='DDP',
            scale_y=1,
        )

    peak_mem_vs_iteration_time = create_model_size_specific_figure(
        path=input_path,
        y_axis_title='Peak Memory (GiB)'
    )

    batch_size_vs_epoch_time = create_model_size_specific_figure(
        path=input_path,
        x='Max Per GPU Batch Size',
        y='Batch Epoch Time (s)',
        y_axis_title='Epoch Time (s)',
        scale_y=1
    )
    return peak_mem_vs_iteration_time_relative, peak_mem_vs_iteration_time, batch_size_vs_epoch_time


checkpoint_message = """\u00b9 Activation checkpointing has been enabled.
            This requires code changes, described in
            [DeepSpeed Activation Checkpointing](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed-activation-checkpointing)
            or [FairScale Checkpointing](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#fairscale-activation-checkpointing).
    """


def plot_model_specific_figure(model_size,
                               peak_mem_vs_iteration_time_relative,
                               peak_mem_vs_iteration_time,
                               batch_size_vs_epoch_time):
    st.subheader(option)
    st.subheader("Peak Memory vs Iteration Time")
    st.text(
        "When your limiting factor is memory, "
        f"use these graph to find out which plugin to use to fit {model_size.lower()} on your GPUs.")
    c1, c2 = st.beta_columns(2)
    if peak_mem_vs_iteration_time_relative:
        c1.plotly_chart(peak_mem_vs_iteration_time_relative)
        st.text("Note the batch size has been fixed to 8.")
        st.write(checkpoint_message)
        c2.plotly_chart(peak_mem_vs_iteration_time)
    else:
        c1.plotly_chart(peak_mem_vs_iteration_time)
        st.text("Note the batch size has been fixed to 8.")
        st.write(checkpoint_message)

    st.subheader("Batch Size vs Epoch Time")
    st.text(
        "When your limiting factor is throughput, "
        "use this graph to find the largest batch size or fastest iteration time to train on large datasets.")

    st.plotly_chart(batch_size_vs_epoch_time)


def create_largest_model_size_figure(input_path):
    df = pd.read_csv(input_path)
    # Extreme hack to get colors for the bar chart
    color = [f'rgba({",".join(str(y) for y in select_color())})' for x in range(len(df))]

    fig = go.Figure(
        go.Bar(
            x=df['Parameters'],
            y=df['Plugin'],
            text=df['Plugin Alias'],
            marker=dict(color=color),
            orientation='h',
            width=[0.7] * len(df)
        ),
    )

    fig.update_xaxes(
        title_text="Maximum Parameter Size (Billion)",
        showgrid=True,
        linewidth=1, linecolor='black',
        gridcolor='rgb(235,240,240)'

    )

    fig.update_layout(plot_bgcolor='rgb(250,250,250)')

    spaces = ' ' * 4  # Gives some space on the y axis
    fig.update_yaxes(
        title_standoff=25,
        ticksuffix=spaces,
        showgrid=False,
        title_font={"size": 20},
        linewidth=1, linecolor='black',
        gridcolor='rgb(230,230,230)',
        tickfont_size=12,
    )
    st.text(
        "Largest model possible to fit with a batch size of 8, "
        "useful to see what can be physically trained on one 8 A100 GPU machine. "
        "This visualization does not account for the decrease in throughput."
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "Activation checkpointing enabled. "
        "See [DeepSpeed Activation Checkpointing](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed-activation-checkpointing) or [FairScale Checkpointing](https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#fairscale-activation-checkpointing)."
    )


st.set_page_config(
    page_title='Lightning minGPT',
    page_icon='https://raw.githubusercontent.com/PyTorchLightning/pytorch-lightning/master/docs/source/_static/images/icon.svg',
    layout="wide"
)
st.title("Lightning Multi-GPU Plugins Visualizations using minGPT")
st.markdown(
    "PyTorch Lightning supports various [multi-gpu plugins](https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html) "
    "that can be used to speed up, "
    "reduce memory and reach larger parameter sizes on multiple GPUs or machines.\n"
    "To get an idea of which plugin suites your model size, batch size and compute, "
    "we've created these visualizations to help gain understanding of how different "
    "plugins and configurations perform when training models."
)
st.markdown(
    "These results were collected using an [8 A100 GPU machine](https://lambdalabs.com/deep-learning/servers/hyperplane-a100).\n"
    "All plugins are available as a string argument to the PyTorch Lightning Trainer via ``plugins=x``."
)

st.code(
    """
from pytorch_lightning import Trainer

model = MyModel()

# replace plugins='deepspeed' with any plugin alias in the below graphs. 
trainer = Trainer(gpus=4, plugins='deepspeed', precision=16)
trainer.fit(model)
    """
)

st.markdown(
    "From this data, high level conclusions and guides can be found in the "
    "[Advanced GPU Optimized Training documentation](https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html).\n\n"
    "In addition, see [the benchmark code](https://github.com/SeanNaren/minGPT/tree/streamlit)."
)

plot_model_size_vs_metric_figure(
    title="Model Parameters vs Peak Memory (GiB)",
    description="Batch size has been fixed to 8. "
                " Useful for determining model size for your hardware limits.\n"
                "In general lower peak memory means less memory required per GPU, "
                "however note in some cases a trade off with speed is required, "
                "which is captured in other below graphs.\n\n"
                "As peak GPU Memory is reduced we can increase batch size, "
                "which is captured separately in the Model Parameters vs Max Per GPU Batch Size graph.",
    y_axis='Peak Memory (MiB)',
    y_axis_title='Average Per GPU Peak Memory (GiB)',
    scale_y=1024
)

plot_model_size_vs_metric_figure(
    title="Model Parameters vs Max Per GPU Batch Size",
    description="Largest batch size per GPU that can fit onto 8 A100 GPUs at multiples of 8, "
                "useful for training regimes that require larger batch sizes (SSL). "
                "This required manual tuning of the batch size so there may be small discrepancies.",
    y_axis='Max Per GPU Batch Size',
    scale_y=1
)

plot_model_size_vs_metric_figure(
    title="Model Parameters vs Throughput (samples per second)",
    description="We use the largest batch size that we could fit per plugin for these timings, "
                "measured in the above graph.\nNote that this means convergence may be affected, "
                "and in the future we will provide time to convergence graphs.",
    y_axis='Batch Epoch Time (s)',
    y_axis_title='Throughput: Samples per second',
    scale_y=1,
    norm_y=num_iters
)
st.header("Per Model Size Breakdown")

st.text(
    "Useful for drilling further into the data if you have a particular model size you'd like to train.")

model_size_plots = {
    "250M Parameters": dict(input_path='results/250m.csv'),
    "500M Parameters": dict(input_path='results/500m.csv'),
    "1 Billion Parameters": dict(input_path='results/1.csv'),
    "4.1 Billion Parameters": dict(input_path='results/4.1.csv', ddp_relative_supported=False),
    "16 Billion Parameters": dict(input_path='results/16.csv', ddp_relative_supported=False),
}

option = st.select_slider(
    'Choose Model Size',
    [
        *model_size_plots.keys(),
        "Largest Model Possible"
    ],
)

selection_kwargs = model_size_plots.get(option)

if selection_kwargs:
    plot_model_specific_figure(option, *create_model_specific_figure(**selection_kwargs))
else:
    st.subheader(option)
    create_largest_model_size_figure('results/size.csv')

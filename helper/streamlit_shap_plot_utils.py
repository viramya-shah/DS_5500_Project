import streamlit as st
import shap
import streamlit.components.v1 as components

COLOR = "black"
BACKGROUND_COLOR = "#fff"

"""
This file is for random functions that need global presence but can't be associated with any particular
class.
"""


def show_plot(plot,
              height=None,
              width=None):
    """
    This function embeds the explainability plots into streamlit supported widgets
    :param plot: Shap plots
    :param height: Height of the widget
    :param width: Width of the widget
    :return: None
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, width=width)


def set_block_container_style(padding_top: int = 5,
                              padding_right: int = 1,
                              padding_left: int = 1,
                              padding_bottom: int = 10):
    """
    This function sets the streamlit window configuration
    :param padding_top: Top padding
    :param padding_right: Right padding
    :param padding_left: Left padding
    :param padding_bottom: Bottom padding
    :return: None
    """
    max_width_str = f"max-width: 80%;"
    st.markdown(f"""
        <style>
            .reportview-container .main .block-container{{
                {max_width_str}
                padding-top: {padding_top}rem;
                padding-right: {padding_right}rem;
                padding-left: {padding_left}rem;
                padding-bottom: {padding_bottom}rem;
            }}
            .reportview-container .main {{
                color: {COLOR};
                background-color: {BACKGROUND_COLOR};
            }}
        </style>
        """,
                unsafe_allow_html=True)

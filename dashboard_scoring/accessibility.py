import streamlit as st
import plotly.graph_objects as go


# Palette WCAG-friendly
COLORS_WCAG = {
    "accepted": "#228B22",    # ForestGreen
    "refused": "#B22222",     # FireBrick
    "borderline": "#D35400",  # Dark orange
    "gauge_low": "#66bb6a",   # Vert accessible
    "gauge_high": "#ef5350",  # Rouge accessible
    "text": "#000000",        # Noir
}

def add_alt_text(text):
    """
    Ajoute un texte alternatif lisible par les lecteurs d'écran
    """
    st.markdown(
        f"<span style='font-size: 0.9em; color: gray;'>(Texte alternatif pour accessibilité) : {text}</span>",
        unsafe_allow_html=True
    )

def get_accessible_color_for_score(score, threshold=0.5):
    """
    Renvoie une couleur accessible avec label explicite selon le score
    """
    if score <= threshold * 0.9:
        return COLORS_WCAG["accepted"], "Accepté"
    elif score >= threshold * 1.1:
        return COLORS_WCAG["refused"], "Refusé"
    else:
        return COLORS_WCAG["borderline"], "Limite"

def apply_accessible_style(fig):
    """
    Applique un style accessible à un graphique Plotly (texte lisible, contraste élevé).
    Si la figure n'est pas un objet Plotly, la renvoie inchangée.
    """
    try:
        import plotly.graph_objects as go
        if isinstance(fig, go.Figure):
            fig.update_layout(
                font=dict(size=16, color=COLORS_WCAG["text"]),
                title_font=dict(size=20),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                legend=dict(font=dict(size=14))
            )
    except Exception:
        pass  # On ne fait rien si ce n'est pas une figure Plotly ou en cas d'erreur

    return fig
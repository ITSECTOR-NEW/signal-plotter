import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, arctan, arcsin, arccos, sinh, cosh, tanh
from numpy import pi, exp, log, log10, sqrt, abs, floor, ceil, round
from numpy import where, maximum, minimum, sign
from numpy import linspace, arange, fft, zeros, ones

# --- App Title
st.title("📈 Signal Plotter and FFT Viewer")

# --- Dropdown for predefined signals
signal_options = {
    "sin(2π5t)": "sin(2*pi*5*t)",
    "cos(2π3t)": "cos(2*pi*3*t)",
    "exp(-t)·cos(t)": "exp(-t)*cos(t)",
    "t² + sin(t)": "t**2 + sin(t)",
    "|sin(5t)|": "abs(sin(5*t))",
    "Unit Step (t > π)": "where(t > pi, 1, 0)",
    "Ramp (t)": "t",
    "Sinc(t)": "sin(t)/(t+1e-9)"
}

selected_label = st.selectbox("Choose a signal:", list(signal_options.keys()))
default_expr = signal_options[selected_label]

# --- Editable custom input
signal_input = st.text_input("Edit or enter your own expression (in terms of 't'):", default_expr)

# --- Enter button
if st.button("▶️ Plot Signal"):
    # --- Define time vector
    t = np.arange(0, 2*np.pi, 0.01)

    # --- Evaluation environment
    safe_env = {
        't': t,
        'sin': sin, 'cos': cos, 'tan': tan,
        'arctan': arctan, 'arcsin': arcsin, 'arccos': arccos,
        'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
        'pi': pi, 'exp': exp, 'log': log, 'log10': log10,
        'sqrt': sqrt, 'abs': abs, 'floor': floor, 'ceil': ceil, 'round': round,
        'where': where, 'maximum': maximum, 'minimum': minimum, 'sign': sign,
        'linspace': linspace, 'arange': arange, 'fft': fft, 'zeros': zeros, 'ones': ones,
        'np': np
    }

    # --- Signal evaluation and plotting
    try:
        y = eval(signal_input, {"__builtins__": None}, safe_env)

        if isinstance(y, np.ndarray) and len(y) == len(t):
            # Time-domain plot
            fig1, ax1 = plt.subplots()
            ax1.plot(t, y)
            ax1.set_title("⏱ Time-Domain Signal")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            st.pyplot(fig1)

            # FFT plot
            Y = np.abs(np.fft.fft(y))
            f = np.fft.fftfreq(len(t), d=0.01)
            f = f[:len(f)//2]
            Y = Y[:len(Y)//2]

            fig2, ax2 = plt.subplots()
            ax2.plot(f, Y)
            ax2.set_title("📊 Frequency Spectrum (FFT)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Amplitude")
            st.pyplot(fig2)

        else:
            st.error("❗ Output must be a NumPy array of same length as 't'.")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

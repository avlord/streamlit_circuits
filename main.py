import streamlit as st
from PIL import Image
# from utils import load_model
import joblib
import numpy as np
import os


def load_model(path):
    """Load model"""
    path_to_model = os.path.join(path, "model.joblib")
    model = joblib.load(path_to_model)
    scaler = joblib.load(os.path.join(path, "scaler.joblib"))
    return model, scaler

def setup_model(vals):
    print(vals)

    bw = vals["bw"]
    pw = vals["pw"]
    gain = vals["a0"]
    model, scaler = load_model("models_trained")
    vals = np.array([bw, pw, gain]).reshape(1,3)
    perf = np.hstack([np.array([0,0,0]).reshape(1,3),vals])
    # perf = np.array([3.17e9,3.81e-04,22]).reshape(1,3)
    # perf = np.hstack([np.array([0,0,0]).reshape(1,3),perf])

    perf = scaler.transform(perf)[:,3:]

    # print('perf after', perf)

    params = model.predict(perf)

    data = np.hstack((params, perf))
    
    # print(data)

    inverse_trans = scaler.inverse_transform(data)
    #params, perf
    print(inverse_trans[:,0:3],inverse_trans[:,3:6])
    print([str(x) for x in inverse_trans[:,0:3]])
    return [str(x) for x in inverse_trans[:,0:3]]
    #bandwidth, power, gain
#     [2.175997e+09 3.816326e-04 2.094111e+01]
# [8.501272e+09 5.610970e-04 2.823009e+01]



circuit = st.radio(
    "Select a circuit for simulation",
    ('Cascode', 'LNA', 'Mixer', 'Common Source Amplifier', 'PA', 'Two Stage', 'VCO'))


if circuit == 'Cascode':
    image = Image.open('images/cascode.png')
    perf_req = ['bw', 'pw', 'a0']
if circuit == 'LNA':
    image = Image.open('images/lna.png')
    perf_req = ['Gt', 'S11', 'Nf']
if circuit == 'Mixer':
    image = Image.open('images/mixer.png')
    perf_req = ['PowerConsumption', 'Swing', 'Conversion_Gain']
if circuit == 'Common Source Amplifier':
    image = Image.open('images/nmos.png')
    perf_req = ['bw', 'pw', 'a0']
if circuit == 'PA':
    image = Image.open('images/pa.png')
    perf_req = ['gain1', 'PAE1', 'DE1']
if circuit == 'Two Stage':
    image = Image.open('images/2st.png')
    perf_req = ['bw', 'pw', 'a0']
if circuit == 'VCO':
    image = Image.open('images/vco.png')
    perf_req = ['power_consumption', 'out_power', 'tuningrange']

st.image(image, caption='Circuit Scheme')

st.markdown('Please, type performance requirements for the circuit. In the following format: bw=3.17e9, pw=3.81e-04, a0=22')

input_values = {}
for req in perf_req:
    input_values[req] = st.text_input(req, key=req)

if st.button('Simulate'):
    st.write('Simulating...')
    st.write(setup_model(input_values))
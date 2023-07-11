import streamlit as st
from PIL import Image
# from utils import load_model
import joblib
import numpy as np
import os


def load_model(path,circuit_name):
    """Load model"""
    circuit_name = circuit_name.lower()
    path_to_model = os.path.join(path, f"{circuit_name}.joblib")
    model = joblib.load(path_to_model)
    scaler = joblib.load(os.path.join(path, f"scaler_{circuit_name}.joblib"))
    return model, scaler

def setup_model(vals,circuit_name,perf_req,num_params):
    print(vals)
    print(circuit_name)
    vals = [vals[key]  for key in  perf_req]

    model, scaler = load_model("models_trained",circuit_name)
    vals = np.array(vals).reshape(1,len(perf_req))
    perf = np.hstack([np.array([0]*num_params).reshape(1,num_params),vals])
    print(perf.shape,vals.shape)
    # perf = np.array([3.17e9,3.81e-04,22]).reshape(1,3)
    # perf = np.hstack([np.array([0,0,0]).reshape(1,3),perf])

    perf = scaler.transform(perf)[:,num_params:]

    # print('perf after', perf)

    params = model.predict(perf)

    data = np.hstack((params, perf))
    
    # print(data)

    inverse_trans = scaler.inverse_transform(data)
    #params, perf
    # print(inverse_trans[:,0:3],inverse_trans[:,3:6])
    # print([str(x) for x in inverse_trans[:,0:3]])
    return [str(x) for x in inverse_trans[:,0:num_params]]
    #bandwidth, power, gain
#     [2.175997e+09 3.816326e-04 2.094111e+01]
# [8.501272e+09 5.610970e-04 2.823009e+01]



circuit = st.radio(
    "Select a circuit for simulation",
    ('Cascode', 'LNA', 'Mixer', 'CS', 'PA', 'Two_Stage', 'VCO'))


if circuit == 'Cascode':
    image = Image.open('images/cascode.png')
    perf_req = ['bw', 'pw', 'a0']
    num_params = 3
if circuit == 'LNA':
    image = Image.open('images/lna.png')
    perf_req = ['Gt', 'S11', 'Nf']
    num_params = 4
if circuit == 'Mixer':
    image = Image.open('images/mixer.png')
    perf_req = ['PowerConsumption', 'Swing', 'Conversion_Gain']
    num_params = 4
if circuit == 'CS':
    image = Image.open('images/nmos.png')
    perf_req = ['bw', 'pw', 'a0']
    num_params =  2
if circuit == 'PA':
    image = Image.open('images/pa.png')
    perf_req = ['gain1', 'PAE1', 'DE1']
    num_params = 4
if circuit == 'Two_Stage':
    image = Image.open('images/2st.png')
    perf_req = ['bw', 'pw', 'a0']
    num_params = 3
if circuit == 'VCO':
    image = Image.open('images/vco.png')
    perf_req = ['power_consumption', 'out_power', 'tuningrange']
    num_params = 4
st.image(image, caption='Circuit Scheme')

st.markdown('Please, type performance requirements for the circuit. In the following format: bw=3.17e9, pw=3.81e-04, a0=22')

input_values = {}
for req in perf_req:
    input_values[req] = st.text_input(req, key=req)

if st.button('Simulate'):
    st.write('Simulating...')
    st.write(setup_model(input_values, circuit, perf_req,num_params))
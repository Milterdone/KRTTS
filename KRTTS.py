from sympy.codegen.ast import String
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile

model_name_k = "Beka-pika/mms_kaz_tts_neutral"
model_k     = VitsModel.from_pretrained(model_name_k)
tokenizer_k = AutoTokenizer.from_pretrained(model_name_k)
rate_k = model_k.config.sampling_rate

model_name_r = "facebook/mms-tts-rus"
model_r     = VitsModel.from_pretrained(model_name_r)
tokenizer_r = AutoTokenizer.from_pretrained(model_name_r)
rate_r = model_r.config.sampling_rate

def synthesize_kk(text :String):
    inputs = tokenizer_k(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model_k(**inputs).waveform
    scipy.io.wavfile.write("output_temp/output_KZ.wav", rate_k, waveform.squeeze().cpu().numpy())

def synthesize_ru(text :String):
    inputs = tokenizer_r(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model_r(**inputs).waveform
    scipy.io.wavfile.write("output_temp/output_RU.wav", rate_r, waveform.squeeze().cpu().numpy())
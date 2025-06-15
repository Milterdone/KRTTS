from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile

model_name = "facebook/mms-tts-rus"
model     = VitsModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
rate = model.config.sampling_rate

text = ("Вот и пришла долгожданная весна."
        "Засветило яркое солнышко, на улице стало тепло."
        "Побежали весёлые ручейки."
        "С юга прилетели птицы, которые так звонко чирикают,"
        "как будто радуются тому, что вернулись в родные края."
        "На полянках расцвели первые подснежники — нежные вестники весны.").replace('.', '.   ')

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    waveform = model(**inputs).waveform
scipy.io.wavfile.write("output_examples/output_RU.wav", rate, waveform.squeeze().cpu().numpy())
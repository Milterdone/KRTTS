from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile

# Лучшая модель для казахского которую я нашёл
model_name = "Beka-pika/mms_kaz_tts_neutral"
model     = VitsModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
rate = model.config.sampling_rate

# Если делать как функцию, то просто текст из ЛЛМки сюда как стринг
text = ("Қазақ халқы қонақжайлылығымен танымал. "
        "Қонақасы – арнайы келген қонақты құрметпен қарсы алып, "
        "дастарханға дәстүрлі тағамдар қою рәсімі. "
        "Көрімдік – маңызды қуанышты бөлісу үшін сыйлық ұсыну дәстүрі. "
        "Қазақ тілі – түркінің қыпшақ тобына жататын тіл, "
        "Қазақстанның мемлекеттік тілі болып табылады. "
        "Жазбаша қазақ әліпбиі кириллица негізінде құрастырылған және 42 әріптен тұрады.").replace('.', '. ')
        # Без добавления whitespace после точек ТТС не останавливается
        # А с ними есть артефакты в речи, небольшие

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    waveform = model(**inputs).waveform
scipy.io.wavfile.write("output_examples/output_KZ.wav", rate, waveform.squeeze().cpu().numpy())

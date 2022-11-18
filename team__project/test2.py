import librosa
import librosa.display
import numpy as np, cv2
import matplotlib.pyplot as plt
import wave
import pyaudio
import librosa
import os
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression #텐서플로우로 바꿀예정
import speech_recognition as speech
import pymysql

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "output.wav"

DATA_PATH = "./password/"
train_data=[]#train_date 저장할 공간
train_label=[]#train_label 저장할 공간
test_data=[]#train_date 저장할 공간
test_label=[]#train_label 저장할 공간

pass_score = 90#비밀번호 인증 기준 90%

def load_wave_generator(path):
    batch_waves = []
    labels = []
    # input_width=CHUNK*6 # wow, big!!
    folders = os.listdir(path)
    # while True:
    # print("loaded batch of %d files" % len(files))
    for folder in folders:
        if not os.path.isdir(path): continue  # 폴더가 아니면 continue
        files = os.listdir(path + "/" + folder)
        print("Foldername :", folder, "-", len(files))  # 폴더 이름과 그 폴더에 속하는 파일 갯수 출력
        for wav in files:
            if not wav.endswith(".wav"):
                continue
            else:
                global train_data, train_label  # 전역변수를 사용하겠다.
                print("Filename :", wav)  # .wav 파일이 아니면 continue
                y, sr = librosa.load(path + "/" + folder + "/" + wav)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr * 0.01), n_fft=int(sr * 0.02)).T
                if (len(train_data) == 0):
                    train_data = mfcc
                    train_label = np.full(len(mfcc), int(folder))
                else:
                    train_data = np.concatenate((train_data, mfcc), axis=0)
                    train_label = np.concatenate((train_label, np.full(len(mfcc), int(folder))), axis=0)
                    print("mfcc :", mfcc.shape)


load_wave_generator(DATA_PATH)

p = pyaudio.PyAudio() # 오디오 객체 생성

stream = p.open(format=FORMAT, # 16비트 포맷
                channels=CHANNELS, #  모노로 마이크 열기
                rate=RATE, #비트레이트
                input=True,
                frames_per_buffer=CHUNK) # CHUNK만큼 버퍼가 쌓인다.

print("Start to record the audio.")

frames = [] # 음성 데이터를 채우는 공간

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    #지정한  100ms를 몇번 호출할 것인지 10 * 3 = 30  100ms 버퍼 30번채움 = 3초
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording is finished.")

stream.stop_stream() # 스트림닫기
stream.close() # 스트림 종료
p.terminate() # 오디오객체 종료

# WAVE_OUTPUT_FILENAME의 파일을 열고 데이터를 쓴다.
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

spf = wave.open(WAVE_OUTPUT_FILENAME,'r')

signal = spf.readframes(-1)
signal = np.frombuffer(signal, dtype=np.int16)

#시간 흐름에 따른 그래프를 그리기 위한 부분
Time = np.linspace(0, len(signal)/RATE, num=len(signal))
# plt.figure(1)
# plt.title('Voice Signal Wave...')
# #plt.plot(signal) // 음성 데이터의 그래프
# plt.plot(Time, signal)
# plt.show()

print("train_data.shape :", train_data.shape, type(train_data))
print("train_label.shape :", train_label.shape, type(train_label))
clf = LogisticRegression(solver='lbfgs', max_iter=100)
clf.fit(train_data, train_label)

y, sr = librosa.load(WAVE_OUTPUT_FILENAME)  # y=signal, sr=sample_rate
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.show()
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr * 0.01), n_fft=int(sr * 0.02)).T

y_test_estimated = clf.predict(mfcc)
print(y_test_estimated)
test_label = np.full(len(mfcc), 0)
print(test_label)

ac_score = metrics.accuracy_score(y_test_estimated, test_label)
print("정답률 =", ac_score)
print(pd.value_counts(pd.Series(y_test_estimated)))

most_frequency = max((y_test_estimated).tolist(), key=(y_test_estimated).tolist().count)
if most_frequency == 0:
    name = 'min'
if most_frequency == 1:
    name = 'song'
print("가장 근접한 사용자 이름 : ", name)

score = round(ac_score*100, 2)
print(score)
if score >= pass_score:
    r = speech.Recognizer()
    record = speech.AudioFile(WAVE_OUTPUT_FILENAME)
    with record as source:
        aud = r.record(source)
    try:
        audio = r.recognize_google(aud, language="ko-KR")
        print("입력한 비밀번호 : ", audio)
    except speech.UnknownValueError:
        print("Your speech can not understand")
    except speech.RequestError as e:
        print("Request Error!; {0}".format(e))

    # 저장된 비밀번호를 반환하기 위해 데이터베이스에서 비밀번호를 꺼냄
    conn = pymysql.connect(host='localhost', user='root', password='1234', db='save_password', charset='utf8')
    cursor = conn.cursor()

    sql = '''SELECT password from user WHERE name = %s'''
    cursor.execute(sql, name)
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql, name)
            result = cur.fetchall()
            for data in result:
                password = data[0]

    if audio == password:
        print("잠금이 해제됩니다..")
    else:
        print("비밀번호가 다릅니다..")
else :
    print("등록된 사용자와 일치하지 않습니다..")
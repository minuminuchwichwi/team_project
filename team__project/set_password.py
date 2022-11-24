import glob
import time, librosa, wave, pyaudio, os, pymysql
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import speech_recognition as speech

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
k = 0  # 음성녹음의 초기 횟수

DATA_PATH = "./password/"
train_data=[]#train_date 저장할 공간
train_label=[]#train_label 저장할 공간
test_data=[]#train_date 저장할 공간
test_label=[]#train_label 저장할 공간

# 새로운 사용자의 폴더가 없으면 폴더 생성
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
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

print('입력하실 비밀번호를 10번 반복하여 녹음합니다.')
name = input('name : ')
if "min" in name:
    folder = '0'
    createDirectory(DATA_PATH + folder)
elif "song" in name:
    folder = '1'
    createDirectory(DATA_PATH + folder)
else:
    folder = str(len(os.listdir(DATA_PATH)) + 1)
    createDirectory(DATA_PATH + folder)

while k < 10:
    time.sleep(1)
    WAVE_OUTPUT_FILENAME = "password/" + folder + '/' + name + str(k) + ".wav"
    print("저장될 음성파일 : ", WAVE_OUTPUT_FILENAME)

    p = pyaudio.PyAudio()  # 오디오 객체 생성
    r = speech.Recognizer()
    stream = p.open(format=FORMAT,  # 16비트 포맷
                    channels=CHANNELS,  # 모노로 마이크 열기
                    rate=RATE,  # 비트레이트
                    input=True,
                    frames_per_buffer=CHUNK)  # CHUNK만큼 버퍼가 쌓인다.

    print("비밀번호를 말씀하십시오..")

    frames = []  # 음성 데이터를 채우는 공간

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # 지정한  100ms를 몇번 호출할 것인지 10 * 5 = 50  100ms 버퍼 50번채움 = 5초
        data = stream.read(CHUNK)
        frames.append(data)

    print("%d회 녹음이 완료되었습니다.." % (k + 1))

    stream.stop_stream()  # 스트림닫기
    stream.close()  # 스트림 종료
    p.terminate()  # 오디오객체 종료

    # WAVE_OUTPUT_FILENAME의 파일을 열고 데이터를 쓴다.
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    record = speech.AudioFile(WAVE_OUTPUT_FILENAME)
    with record as source:
        aud = r.record(source)
    try:
        globals()['audio{}'.format(k)] = r.recognize_google(aud, language="ko-KR")
        print("녹음된 비밀번호 : " + globals()['audio{}'.format(k)])
        if k == 0:
            audio = globals()['audio{}'.format(k)]
    except speech.UnknownValueError:
        print("Your speech can not understand")
    except speech.RequestError as e:
        print("Request Error!; {0}".format(e))

    if k == 9:
        print("비밀번호 설정이 완료되었습니다..")
    else : print("1초 뒤, 시작합니다...")

    if globals()['audio{}'.format(k)] == audio:
        if k == 9:
            password = audio
        k += 1
    else:
        print("비밀번호가 다릅니다.")
        break

print("설정된 비밀번호는 : " + password)

conn = pymysql.connect(host='localhost', user='root', password='1234', db='save_password', charset='utf8')
cursor = conn.cursor()

sql = '''INSERT INTO user (name, password) VALUES (%s, %s)'''

cursor.execute(sql, (name, password))
conn.commit()

sql = '''DELETE FROM user WHERE name = %s and id !=''' + str(int(folder) + 1)
cursor.execute(sql, name)
conn.commit()

print("----------------------------------")
print("데이터베이스 테이블 출력")
sql = '''SELECT * from user'''
cursor.execute(sql)
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        result = cur.fetchall()
        for data in result:
            print(data)

# npy파일이 존재한다면 덮어쓰기에 관한 오류가 발생하여 미리 npy파일을 지움.
for f in glob.glob('password/*.npy'):
    os.remove(f)

load_wave_generator(DATA_PATH)

# 훈련된 데이터는 numpy array로 저장되기에 .npz 파일로 저장
np.save('password/train_data.npy', train_data)
np.save('password/train_label.npy', train_label)
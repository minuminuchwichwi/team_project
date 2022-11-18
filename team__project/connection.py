import pymysql

mode = int(input('0 : 데이터베이스 생성\n1 : 테이블 생성\n2 : 종료\n3 : 테이블 삭제\n'))

while mode != 2:
    # db 생성
    if mode == 0:
        conn = pymysql.connect(host='localhost', user='root', password='1234', charset='utf8')
        cursor = conn.cursor()

        sql = "CREATE DATABASE save_password"
        cursor.execute(sql)
        conn.commit()
        conn.close()

    # 테이블 생성
    if mode == 1:
        conn = pymysql.connect(host='localhost', user='root', password='1234', db='save_password', charset='utf8')
        cursor = conn.cursor()

        sql = '''CREATE TABLE user ( 
        id int(11) NOT NULL AUTO_INCREMENT PRIMARY KEY, 
        name varchar(255), 
        password varchar(255) 
        )
        '''
        cursor.execute(sql)
        conn.commit()
        conn.close()

    if mode == 2:
        break

    # 테이블 삭제
    if mode == 3:
        conn = pymysql.connect(host='localhost', user='root', password='1234', db='save_password', charset='utf8')
        cursor = conn.cursor()

        sql = '''DROP TABLE user'''

        cursor.execute(sql)
        conn.commit()
        conn.close()

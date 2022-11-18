import pymysql

conn = pymysql.connect(host='localhost', user='root', password='1234', db='save_password', charset='utf8')
cursor = conn.cursor()

sql = '''SELECT password from user WHERE name = 'min' '''
cursor.execute(sql)
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        result = cur.fetchall()
        print(result)
        for data in result:
            print(data[0])

password = data[0]
print(password)
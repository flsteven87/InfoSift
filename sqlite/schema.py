import sqlite3

# 連接到 SQLite 數據庫
# 如果數據庫不存在，那麼它就會被創建
# 如果已經存在，那麼就會連接到這個數據庫
conn = sqlite3.connect('infosift.db')

# 創建一個 Cursor 物件並調用其 execute() 方法來執行 SQL 命令
c = conn.cursor()

# 創建一個表
c.execute("""
    CREATE TABLE video (
        video_id TEXT PRIMARY KEY,
        url TEXT, 
        title TEXT, 
        channel TEXT, 
        video_length INTEGER, 
        video_size INTEGER, 
        views_count INTEGER,
        created_time TEXT, 
        updated_time TEXT
    )
""")

# 關閉與數據庫的連接
conn.close()

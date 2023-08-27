import sqlite3
import pandas as pd

# 連接到 SQLite 數據庫
# 如果數據庫不存在，那麼它就會被創建
# 如果已經存在，那麼就會連接到這個數據庫
conn = sqlite3.connect('./sqlite/infosift.db')

# 創建一個 Cursor 物件並調用其 execute() 方法來執行 SQL 命令
c = conn.cursor()

# Write the query
query = "SELECT video_id, title, channel FROM video"

# Use pandas to pass sql query using connection from SQLite3
df = pd.read_sql_query(query, conn)

# Show the resulting DataFrame
print(df)


# 關閉與數據庫的連接
conn.close()

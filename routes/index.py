from flask import Blueprint, render_template

# 這邊的功能為 Flask 應用程式定義一個「藍圖」（Blueprint），並將一個路由與對應的視圖函數（View Function）連結。
# Blueprint 是 Flask 提供的一種模組化工具，用於將應用程式拆分為更小的組件。

# Blueprint('index', __name__)：
# 'index' 是這個 Blueprint 的名稱，通常用於區分不同的模組或功能區域。
# __name__ 是當前模組的名稱，用於告訴 Flask 這個 Blueprint 是在哪個模組中定義的。
# index_bp 是這個 Blueprint 的實例，之後會被註冊到 Flask 主應用程式中（通常在主程式 app.py 中）。

# 1.藍圖名稱 (Blueprint Name)：
# 每個 Blueprint 都有一個名稱，這是定義藍圖時的第一個參數。
# 例如：
# upload_bp = Blueprint('upload', __name__)
# 此處 upload 是這個 Blueprint 的名稱。

# 2.視圖函數名稱：
# Blueprint 中每個路由的視圖函數名稱。
# 例如：
# @upload_bp.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     ...
# upload_file 是這個路由的視圖函數名稱。

# 3.合併結果：
# Flask 將藍圖名稱與視圖函數名稱組合在一起，形成完整的路由名稱：<藍圖名稱>.<視圖函數名稱>。

index_bp = Blueprint('index', __name__)

@index_bp.route('/')
def index():
    return render_template('index.html')

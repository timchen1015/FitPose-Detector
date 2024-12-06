import requests
import os

# 替換為你的 API 金鑰
API_KEY = '47485650-628003982870b961d61121b91'
BASE_URL = 'https://pixabay.com/api/'

# 設定儲存資料夾（使用原始字串避免反斜線轉義問題）
output_dir = r"C:\Users\User\Desktop\Improvements-in-Fitness-Pose-Detection-for-Exercise-1\exercise_dataset\push_up"
os.makedirs(output_dir, exist_ok=True)

# 每頁最多可下載 200 張圖片，最多支持 5 頁
total_images = 500
images_per_page = 200
pages = total_images // images_per_page + (1 if total_images % images_per_page > 0 else 0)

for page in range(1, pages + 1):
    # 向 API 發送請求
    params = {
        'key': API_KEY,
        'q': 'push up',  # 搜尋關鍵字
        'image_type': 'photo',
        'per_page': images_per_page,
        'page': page
    }

    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hits = data['hits']
        
        for idx, hit in enumerate(hits):
            # 檢查標題、標籤是否包含 "push up"
            if 'push up' in hit['tags'].lower() or 'pushup' in hit['tags'].lower():
                img_url = hit['webformatURL']  # 圖片 URL
                try:
                    img_data = requests.get(img_url).content
                    img_filename = os.path.join(output_dir, f"pushup_{(page - 1) * images_per_page + idx + 1}.jpg")
                    with open(img_filename, 'wb') as f:
                        f.write(img_data)
                    print(f"下載完成: {img_filename}")
                except Exception as e:
                    print(f"下載失敗: {img_url}，原因: {e}")
            else:
                print(f"忽略圖片: {hit['tags']}")  # 顯示忽略的圖片標籤
    else:
        print(f"無法抓取圖片，HTTP 狀態碼: {response.status_code}")

import requests
import os

# 替換為你的 API 金鑰
API_KEY = 'MUd6V9McS6Ex7PZRzOVGpywpzPRYs1dwbFh7ZMWg90SPaX3k6baB4MjK'
BASE_URL = 'https://api.pexels.com/v1/search'

# 設定儲存資料夾
output_dir = r"C:\Users\User\Desktop\Improvements-in-Fitness-Pose-Detection-for-Exercise-1\exercise_dataset\push_up"
os.makedirs(output_dir, exist_ok=True)

# 每頁最多可下載 15 張圖片
images_per_page = 15
total_images = 500
pages = total_images // images_per_page + (1 if total_images % images_per_page > 0 else 0)

for page in range(1, pages + 1):
    headers = {
        'Authorization': API_KEY  # 使用你的 API 金鑰
    }
    params = {
        'query': 'push up',  # 搜尋關鍵字
        'per_page': images_per_page,
        'page': page
    }

    response = requests.get(BASE_URL, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        photos = data['photos']

        for idx, photo in enumerate(photos):
            img_url = photo['src']['large']  # 圖片 URL
            try:
                img_data = requests.get(img_url).content
                img_filename = os.path.join(output_dir, f"pushup_{(page - 1) * images_per_page + idx + 1}.jpg")
                with open(img_filename, 'wb') as f:
                    f.write(img_data)
                print(f"下載完成: {img_filename}")
            except Exception as e:
                print(f"下載失敗: {img_url}，原因: {e}")
    else:
        print(f"無法抓取圖片，HTTP 狀態碼: {response.status_code}")

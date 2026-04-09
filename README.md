# ngaplut_all — Pipeline Dự Báo Ngập Lụt

Pipeline tự động hoá hoàn chỉnh gồm 2 bước:

1. **`code_ai.py`** — Huấn luyện & dự báo mực nước tại hai trạm (**Phú An** và **Nhà Bè**) bằng 9 model Machine Learning / Deep Learning.
2. **`code_process_spatial.py`** — Nội suy IDW + xác định vùng ngập + xuất các file không gian địa lý.
3. **`main.py`** — Script điều phối chạy tuần tự 2 bước trên qua một lệnh duy nhất.
4. **`requirements.txt`** — Danh sách thư viện cần cài đặt.

---

## Cấu trúc thư mục

```
ngaplut_all/
├── main.py                        # Script pipeline chính
├── code_ai.py                     # Bước 1: Dự báo AI
├── code_process_spatial.py        # Bước 2: Xử lý không gian
├── requirements.txt               # Danh sách thư viện Python
│
├── preprocessed_flood_data3.csv   # Dữ liệu chuỗi thời gian mực nước (đầu vào AI)
├── validation_2023.csv            # Dữ liệu validate năm 2023
│
├── data/
│   ├── dem_process_32648.csv      # DEM đầu vào (cột X, Y, VALUE — EPSG:32648)
│   ├── nn2025.geojson             # Polygon vùng nông nghiệp cần kiểm tra ảnh hưởng
│   └── bandodem_epsg32648.tif     # Raster tham chiếu (EPSG:32648)
│
├── results/
│   └── flood_all_models/          # ← OUTPUT của code_ai.py
│       ├── best_predict.csv                          # Dự báo của model có NSE tốt nhất
│       ├── best_model_by_nse.csv                     # Bảng tóm tắt model tốt nhất mỗi trạm
│       ├── combined_flood_data.csv                   # Dữ liệu tổng hợp tất cả model
│       ├── model_metrics.csv                         # Đánh giá trên tập test
│       ├── validation_metrics.csv                    # Đánh giá trên tập validation 2023
│       ├── {model}_predictions.csv                   # Dự báo riêng từng model
│       ├── plot_H_Phú_An_Max_m_{model}.png           # Biểu đồ trạm Phú An từng model
│       ├── plot_H_Nhà_Bè_Max_m_{model}.png           # Biểu đồ trạm Nhà Bè từng model
│       └── validation_compare_{target}_{model}.csv  # So sánh validate chi tiết từng model
│
└── ketqua_spatial/                # ← OUTPUT của code_process_spatial.py
    ├── results.csv
    ├── flood.csv
    ├── flood_xy_value.csv
    ├── ngap_point.geojson
    ├── bandongap_cogiatri.gpkg
    ├── vung_ngap_anhhuong_sx_nongnghiep.geojson
    └── vung_ngap_anhhuong_sx_nongnghiep.tif
```

> **Lưu ý:** `preprocessed_flood_data3.csv` và `validation_2023.csv` mặc định được đọc từ thư mục gốc dự án (cùng cấp với `code_ai.py`). Nếu bạn đặt chúng ở `data/`, hãy thêm tham số `--data-path data/preprocessed_flood_data3.csv --validation-path data/validation_2023.csv` khi chạy.

---

## Bước 1 — `code_ai.py` (Dự báo AI)

Huấn luyện 9 model ML/DL trên dữ liệu mực nước lịch sử, tự động chọn model tốt nhất theo NSE và xuất file dự báo.

### Các model hỗ trợ

| Tên tham số | Model |
|-------------|-------|
| `xgb` | XGBoost |
| `lgbm` | LightGBM |
| `catboost` | CatBoost |
| `rf` | Random Forest |
| `etr` | Extra Trees |
| `hgbr` | HistGradientBoosting |
| `ridge` | Ridge Regression |
| `lstm` | LSTM (TensorFlow/Keras) |
| `stack` | Stacking Ensemble |

### File đầu vào
| File | Mô tả |
|------|-------|
| `preprocessed_flood_data3.csv` | Dữ liệu chuỗi thời gian mực nước lịch sử tại trạm Phú An & Nhà Bè |
| `validation_2023.csv` | Dữ liệu năm 2023 dùng để validate model sau huấn luyện |

### File đầu ra (`results/flood_all_models/`)
| File | Mô tả |
|------|-------|
| `best_predict.csv` | Dự báo tương lai của model có NSE tốt nhất (đầu vào của bước 2) |
| `best_model_by_nse.csv` | Model được chọn cho mỗi trạm và nguồn NSE (test / validation) |
| `combined_flood_data.csv` | Toàn bộ dữ liệu tổng hợp — thực đo + dự báo của tất cả model |
| `model_metrics.csv` | Chỉ tiêu đánh giá tập **test** (NSE, RMSE, MAE, R², RSR…) cho mỗi model/trạm |
| `validation_metrics.csv` | Chỉ tiêu đánh giá tập **validation 2023** |
| `{model}_predictions.csv` | Dự báo tương lai riêng của từng model (ví dụ: `xgb_predictions.csv`) |
| `plot_H_Phú_An_Max_m_{model}.png` | Biểu đồ thực đo vs dự báo tại trạm **Phú An** cho từng model |
| `plot_H_Nhà_Bè_Max_m_{model}.png` | Biểu đồ thực đo vs dự báo tại trạm **Nhà Bè** cho từng model |
| `validation_compare_{target}_{model}.csv` | So sánh chi tiết thực đo vs dự báo trên tập validation |
| `model_{target}_{model}` *(nếu `--save-pkl true`)* | File model đã huấn luyện (joblib / keras) |

### Cú pháp & ví dụ

```bash
# Chạy trực tiếp code_ai.py — tất cả model, dự báo 7 ngày
python code_ai.py --models all --forecast-days 7 \
    --train-end-date 2021-12-31 --test-end-date 2022-12-31

# Chỉ chạy một số model
python code_ai.py --models xgb etr catboost --forecast-days 7 \
    --train-end-date 2021-12-31 --test-end-date 2022-12-31

# Chỉ định đường dẫn dữ liệu tùy chỉnh
python code_ai.py --models all \
    --data-path data/preprocessed_flood_data3.csv \
    --validation-path data/validation_2023.csv
```

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--models` | `all` hoặc tên model cách nhau dấu cách | `all` |
| `--forecast-days` | Số ngày dự báo tương lai | `7` |
| `--train-end-date` | Ngày kết thúc tập huấn luyện (`YYYY-MM-DD`) | `2021-12-31` |
| `--test-end-date` | Ngày kết thúc tập kiểm tra (`YYYY-MM-DD`) | `2022-12-31` |
| `--save-pkl` | Lưu model dạng file (`true`/`false`) | `false` |
| `--data-path` | Đường dẫn file dữ liệu huấn luyện | `preprocessed_flood_data3.csv` |
| `--validation-path` | Đường dẫn file validation | `validation_2023.csv` |
| `--output-folder` | Thư mục lưu kết quả | `results/flood_all_models` |

---

## Bước 2 — `code_process_spatial.py` (Xử lý không gian)

Đọc kết quả dự báo từ `best_predict.csv`, nội suy mực nước IDW lên toàn bộ lưới DEM, tính độ sâu ngập, xác định vùng ảnh hưởng và xuất các file GIS.

### File đầu vào
| Đường dẫn | Mô tả |
|-----------|-------|
| `results/flood_all_models/best_predict.csv` | Dự báo mực nước (output của bước 1) |
| `data/dem_process_32648.csv` | Lưới DEM — cột `X`, `Y`, `VALUE` (EPSG:32648) |
| `data/nn2025.geojson` | Polygon vùng nông nghiệp cần xét |
| `data/bandodem_epsg32648.tif` | Raster tham chiếu để rasterize |

### File đầu ra (`ketqua_spatial/`)
| File | Mô tả |
|------|-------|
| `results.csv` | Toàn bộ lưới DEM với cột bổ sung: `H_estimate` (mực nước IDW), `Flood_Depth` (độ sâu ngập), `Is_Flooded` (0/1) |
| `flood.csv` | Chỉ các điểm có `Is_Flooded = 1` — đầy đủ cột |
| `flood_xy_value.csv` | Các điểm ngập — chỉ cột `X`, `Y`, `VALUE` |
| `ngap_point.geojson` | Point GeoJSON các điểm ngập (EPSG:32648) |
| `bandongap_cogiatri.gpkg` | GeoPackage với layer `flood_points` — điểm ngập kèm `Flood_Depth` |
| `vung_ngap_anhhuong_sx_nongnghiep.geojson` | Điểm ngập nằm bên trong polygon nông nghiệp (`nn2025.geojson`) |
| `vung_ngap_anhhuong_sx_nongnghiep.tif` | Raster nhị phân (0/1) vùng ngập ảnh hưởng, nén LZW (EPSG:32648) |

### Sơ đồ xử lý
```
best_predict.csv  ──►  Lấy H tại Phú An & Nhà Bè (dòng đầu tiên)
                             │
                             ▼
dem_process_32648.csv ──►  Nội suy IDW  ──►  H_estimate tại mỗi điểm DEM
                             │
                             ▼
                        Tính Flood_Depth = H_estimate − VALUE
                             │
                    ┌────────┴─────────┐
                    ▼                  ▼
              results.csv        flood.csv / flood_xy_value.csv
                                       │
                    ┌──────────────────┤
                    ▼                  ▼
             ngap_point.geojson   bandongap_cogiatri.gpkg
                    │
                    ▼  (spatial join với nn2025.geojson)
       vung_ngap_anhhuong_sx_nongnghiep.geojson
                    │
                    ▼  (rasterize theo bandodem_epsg32648.tif)
       vung_ngap_anhhuong_sx_nongnghiep.tif
```

---

## Bước 3 — `main.py` (Pipeline hoàn chỉnh)

Gọi tuần tự `code_ai.py` → `code_process_spatial.py` qua một lệnh duy nhất.

### Cú pháp
```bash
python main.py [--models MODEL [MODEL ...]]
               [--forecast-days N]
               [--train-end-date YYYY-MM-DD]
               [--test-end-date YYYY-MM-DD]
               [--save-pkl {true,false}]
               [--skip-ai]
               [--skip-spatial]
```

### Ví dụ

```bash
# Chạy toàn bộ pipeline với tất cả model
python main.py --models all --forecast-days 7 \
    --train-end-date 2021-12-31 --test-end-date 2022-12-21 --save-pkl false

# Chạy toàn bộ pipeline với model cụ thể
python main.py --models xgb etr catboost --forecast-days 7 \
    --train-end-date 2021-12-31 --test-end-date 2022-12-21 --save-pkl false

# Chỉ chạy xử lý không gian (bỏ qua AI, dùng best_predict.csv đã có)
python main.py --skip-ai

# Chỉ chạy AI, bỏ qua xử lý không gian
python main.py --models all --forecast-days 7 \
    --train-end-date 2021-12-31 --test-end-date 2022-12-21 --skip-spatial
```

| Tham số | Mô tả |
|---------|-------|
| `--skip-ai` | Bỏ qua `code_ai.py` (dùng khi `best_predict.csv` đã tồn tại) |
| `--skip-spatial` | Bỏ qua `code_process_spatial.py` |

---

## Cài đặt thư viện

Tất cả thư viện cần thiết được liệt kê trong `requirements.txt`. Chạy lệnh sau để cài đặt:

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
# Core scientific stack
pip install numpy pandas matplotlib scikit-learn joblib optuna openpyxl

# Gradient boosting
pip install xgboost lightgbm catboost

# Deep learning (LSTM)
pip install tensorflow

# GIS / không gian
pip install geopandas rasterio pyproj
```

> **Lưu ý:** Đảm bảo GDAL đã được cài đặt trước khi cài `rasterio` và `geopandas`.  
> Trên Google Cloud Linux VM, dùng `pip install tensorflow` (không phải `tensorflow-gpu`).

---

## Lưu ý đường dẫn

- Chạy tất cả script từ **thư mục gốc** của dự án (`ngaplut_all/`).
- `code_ai.py` mặc định đọc file dữ liệu tại thư mục gốc; dùng `--data-path` / `--validation-path` nếu file nằm trong `data/`.
- Tất cả đường dẫn đầu vào trong `code_process_spatial.py` là tương đối so với thư mục chạy script.
- Thư mục đầu ra `ketqua_spatial/` và `results/flood_all_models/` sẽ được tạo tự động.
- File `best_predict.csv` phải nằm tại `results/flood_all_models/best_predict.csv` trước khi chạy bước 2.

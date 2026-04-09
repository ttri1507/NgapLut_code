### module 2 + 3 + 4 + 5 tích hợp hoàn chỉnh ###
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from pathlib import Path

# =========================================
# 0. KHAI BÁO ĐƯỜNG DẪN
# =========================================
predictions_file = "results/flood_all_models/best_predict.csv"
input_file = "data/dem_process_32648.csv"

polygon_path = "data/nn2025.geojson"          # polygon vùng cần xác định ảnh hưởng
ref_raster_path = "data/bandodem_epsg32648.tif"  # raster tham chiếu

output_dir = Path("/ketqua_spatial")
output_dir.mkdir(parents=True, exist_ok=True)

results_file = output_dir / "results.csv"
flood_file = output_dir / "flood.csv"
flood_xy_value_file = output_dir / "flood_xy_value.csv"
geojson_file = output_dir / "ngap_point.geojson"
gpkg_file = output_dir / "bandongap_cogiatri.gpkg"
affected_geojson_file = output_dir / "vung_ngap_anhhuong_sx_nongnghiep.geojson"
out_tif = output_dir / "vung_ngap_anhhuong_sx_nongnghiep.tif"

# =========================================
# TÙY CHỈNH
# =========================================
idw_power = 2
buffer_radius = 3   # mét; để 0 nếu muốn giữ nguyên point -> pixel

# =========================================
# 1. KHAI BÁO 2 TRẠM ĐO (tọa độ gốc lon/lat)
#    H sẽ được đọc từ file etr_predictions.csv
# =========================================
stations = {
    'Phu_An': {'lon': 106.718, 'lat': 10.793, 'H': None},
    'Nha_Be': {'lon': 106.732, 'lat': 10.686, 'H': None}
}

# =========================================
# 2. CONVERT TỌA ĐỘ TRẠM TỪ EPSG:4326 -> EPSG:32648
# =========================================
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)

for name, data in stations.items():
    x_proj, y_proj = transformer.transform(data['lon'], data['lat'])
    stations[name]['x'] = x_proj
    stations[name]['y'] = y_proj

# =========================================
# 3. HÀM PHỤ TRỢ TÌM CỘT
# =========================================
def find_column_by_keywords(columns, keyword_groups):
    """
    Tìm tên cột dựa theo nhiều bộ từ khóa.
    Chỉ cần 1 group match hoàn toàn là nhận.
    """
    for col in columns:
        col_lower = str(col).strip().lower()
        for group in keyword_groups:
            if all(keyword in col_lower for keyword in group):
                return col
    return None

# =========================================
# 4. ĐỌC FILE DỰ BÁO AI VÀ LẤY MỰC NƯỚC
#    -> LÀM TRÒN 2 CHỮ SỐ THẬP PHÂN
# =========================================
try:
    try:
        pred_df = pd.read_csv(predictions_file, encoding="utf-8-sig")
    except Exception:
        pred_df = pd.read_csv(predictions_file, encoding="latin1")

    pred_df.columns = pred_df.columns.str.strip()

    if pred_df.empty:
        raise ValueError("File etr_predictions.csv không có dữ liệu.")

    # Dò cột Phú An
    phu_an_col = find_column_by_keywords(
        pred_df.columns,
        [
            ["phú", "an", "dự", "đoán"],
            ["phu", "an", "du", "doan"],
            ["phu", "an", "pred"],
            ["h", "an", "max"],
            ["phãº", "an", "dá»±", "ä‘oã¡n"],
            ["phãº", "an", "max"]
        ]
    )

    # Dò cột Nhà Bè
    nha_be_col = find_column_by_keywords(
        pred_df.columns,
        [
            ["nhà", "bè", "dự", "đoán"],
            ["nha", "be", "du", "doan"],
            ["nha", "be", "pred"],
            ["nhã", "bã¨", "dá»±", "ä‘oã¡n"],
            ["nhã", "bã¨", "max"],
            ["bè", "dự", "đoán"]
        ]
    )

    # fallback theo vị trí cột
    # cột 0 = ngày
    # cột 1 = Phú An dự đoán
    # cột 2 = giờ xuất hiện
    # cột 3 = Nhà Bè dự đoán
    if phu_an_col is None or nha_be_col is None:
        if pred_df.shape[1] >= 4:
            phu_an_col = pred_df.columns[1]
            nha_be_col = pred_df.columns[3]
            print("Không dò được tên cột chuẩn, dùng fallback theo vị trí cột.")
        else:
            raise ValueError("Không tìm được cột dự đoán Phú An / Nhà Bè trong file etr_predictions.csv")

    first_row = pred_df.iloc[0]

    phu_an_h = pd.to_numeric(first_row[phu_an_col], errors="coerce")
    nha_be_h = pd.to_numeric(first_row[nha_be_col], errors="coerce")

    if pd.isna(phu_an_h) or pd.isna(nha_be_h):
        raise ValueError(
            f"Không đọc được giá trị mực nước từ dòng đầu tiên. "
            f"Phu_An={phu_an_h}, Nha_Be={nha_be_h}"
        )

    # Làm tròn 2 chữ số theo yêu cầu
    stations['Phu_An']['H'] = round(float(phu_an_h), 2)
    stations['Nha_Be']['H'] = round(float(nha_be_h), 2)

    print("=== Giá trị mực nước đọc từ etr_predictions.csv (dòng đầu tiên, làm tròn 2 số) ===")
    print(f"Phu_An H = {stations['Phu_An']['H']:.2f}")
    print(f"Nha_Be H = {stations['Nha_Be']['H']:.2f}")

except Exception as e:
    print(f"Lỗi khi đọc file dự báo AI: {e}")
    raise

# =========================================
# 5. KIỂM TRA TỌA ĐỘ + MỰC NƯỚC SAU KHI NẠP
# =========================================
print("\n=== Tọa độ trạm sau khi chuyển sang EPSG:32648 ===")
for name, data in stations.items():
    print(f"{name}: X = {data['x']:.3f}, Y = {data['y']:.3f}, H = {data['H']:.2f}")

# =========================================
# 6. ĐỌC DỮ LIỆU DEM ĐẦU VÀO
#    File đã có dạng X, Y, VALUE
# =========================================
try:
    df = pd.read_csv(input_file)
except Exception as e:
    print(f"Lỗi khi đọc file DEM: {e}")
    raise

required_columns = ['X', 'Y', 'VALUE']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Thiếu cột trong file đầu vào: {missing_cols}")

for col in ['X', 'Y', 'VALUE']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['X', 'Y', 'VALUE']).copy()

# =========================================
# 7. TÍNH IDW (BẢN VECTORIZE CHO NHANH)
# =========================================
print("\nĐang tính H_estimate theo IDW...")

x = df["X"].to_numpy()
y = df["Y"].to_numpy()

x1, y1, h1 = stations["Phu_An"]["x"], stations["Phu_An"]["y"], stations["Phu_An"]["H"]
x2, y2, h2 = stations["Nha_Be"]["x"], stations["Nha_Be"]["y"], stations["Nha_Be"]["H"]

dist1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
dist2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)

# Tránh chia cho 0
eps = 1e-12
dist1_safe = np.where(dist1 == 0, eps, dist1)
dist2_safe = np.where(dist2 == 0, eps, dist2)

w1 = 1.0 / (dist1_safe ** idw_power)
w2 = 1.0 / (dist2_safe ** idw_power)

h_est = (w1 * h1 + w2 * h2) / (w1 + w2)

# Nếu trùng đúng vị trí trạm thì lấy đúng H của trạm
h_est = np.where(dist1 == 0, h1, h_est)
h_est = np.where(dist2 == 0, h2, h_est)

df["H_estimate"] = h_est

# =========================================
# 8. TÍNH ĐỘ SÂU NGẬP
# =========================================
df["Flood_Depth"] = df["H_estimate"] - df["VALUE"]
df["Is_Flooded"] = (df["Flood_Depth"] > 0).astype(int)
df.loc[df["Is_Flooded"] == 0, "Flood_Depth"] = 0

# Có thể làm tròn nhẹ để file gọn hơn
df["H_estimate"] = df["H_estimate"].round(4)
df["Flood_Depth"] = df["Flood_Depth"].round(4)

# =========================================
# 9. XUẤT CSV
# =========================================
print("\nĐang lưu các file CSV...")

# file tổng
df.to_csv(results_file, index=False, encoding="utf-8-sig")
print(f"Đã lưu file tổng hợp: {results_file}")

# file chỉ chứa điểm ngập, giữ toàn bộ cột
df_flood = df[df["Is_Flooded"] == 1].copy()
df_flood.to_csv(flood_file, index=False, encoding="utf-8-sig")
print(f"Đã lưu file chỉ chứa điểm ngập: {flood_file}")

# file chỉ chứa X, Y, VALUE của các điểm ngập
df_flood_xy_value = df_flood[["X", "Y", "VALUE"]].copy()
df_flood_xy_value.to_csv(flood_xy_value_file, index=False, encoding="utf-8-sig")
print(f"Đã lưu file X, Y, VALUE của điểm ngập: {flood_xy_value_file}")

print(f"Tổng số điểm khảo sát: {len(df):,}")
print(f"Số điểm bị ngập: {len(df_flood):,}")

# =========================================
# 10. MODULE 3 - TẠO GEOJSON TỪ flood_xy_value.csv
# =========================================
print("\nĐang tạo GeoJSON từ flood_xy_value.csv...")

if len(df_flood_xy_value) == 0:
    print("Không có điểm ngập nên dừng quy trình tại đây.")
else:
    geo_df = pd.read_csv(flood_xy_value_file)
    geo_df.columns = geo_df.columns.str.strip()

    geo_df["X"] = pd.to_numeric(geo_df["X"], errors="coerce")
    geo_df["Y"] = pd.to_numeric(geo_df["Y"], errors="coerce")
    geo_df["VALUE"] = pd.to_numeric(geo_df["VALUE"], errors="coerce")

    geo_df = geo_df.dropna(subset=["X", "Y"]).copy()

    gdf_geojson = gpd.GeoDataFrame(
        geo_df[["VALUE"]],
        geometry=gpd.points_from_xy(geo_df["X"], geo_df["Y"]),
        crs="EPSG:32648"
    )

    gdf_geojson.to_file(geojson_file, driver="GeoJSON")
    print(f"Đã lưu file GeoJSON: {geojson_file}")
    print(f"Số điểm trong GeoJSON: {len(gdf_geojson):,}")

    # =========================================
    # 11. TẠO GPKG TỪ flood.csv
    # =========================================
    print("\nĐang tạo GeoPackage (.gpkg) từ flood.csv...")

    gpkg_df = pd.read_csv(flood_file)
    gpkg_df.columns = gpkg_df.columns.str.strip()

    gpkg_df["X"] = pd.to_numeric(gpkg_df["X"], errors="coerce")
    gpkg_df["Y"] = pd.to_numeric(gpkg_df["Y"], errors="coerce")
    gpkg_df["Flood_Depth"] = pd.to_numeric(gpkg_df["Flood_Depth"], errors="coerce")

    gpkg_df = gpkg_df.dropna(subset=["X", "Y", "Flood_Depth"]).copy()

    gpkg_df["Flood_Depth"] = gpkg_df["Flood_Depth"].round(2)
    gpkg_df["X"] = gpkg_df["X"].round(2)
    gpkg_df["Y"] = gpkg_df["Y"].round(2)

    gdf_gpkg = gpd.GeoDataFrame(
        gpkg_df[["Flood_Depth"]],
        geometry=gpd.points_from_xy(gpkg_df["X"], gpkg_df["Y"]),
        crs="EPSG:32648"
    )

    gdf_gpkg.to_file(gpkg_file, driver="GPKG", layer="flood_points")
    print(f"✅ Đã lưu file GPKG: {gpkg_file}")
    print(f"Tổng số điểm trong GPKG: {len(gdf_gpkg):,}")

    # =========================================
    # 12. MODULE 4 - XÁC ĐỊNH VÙNG NGẬP
    # =========================================
    print("\nĐang xác định vùng ngập nằm trong polygon...")

    gdf_a = gpd.read_file(polygon_path)   # polygon
    gdf_b = gpd.read_file(geojson_file)   # point

    if gdf_a.empty:
        raise ValueError("File polygon nn2025.geojson không có dữ liệu.")

    if gdf_b.empty:
        raise ValueError("File ngap_point.geojson không có dữ liệu.")

    if gdf_a.crs != gdf_b.crs:
        print(f"-> Chuyển CRS của điểm từ {gdf_b.crs} sang {gdf_a.crs}")
        gdf_b = gdf_b.to_crs(gdf_a.crs)

    result = gpd.sjoin(gdf_b, gdf_a, how="inner", predicate="within")

    # chỉ giữ geometry cho nhẹ
    result = result[["geometry"]].copy()

    result.to_file(affected_geojson_file, driver="GeoJSON")
    print(f"Đã lưu: {affected_geojson_file}")
    print(f"Số điểm nằm trong vùng: {len(result):,}")

    # =========================================
    # 13. MODULE 5 - CHUYỂN VÙNG NGẬP RA TIFF
    # =========================================
    print("\nĐang convert vùng ngập sang file .tif ...")

    gdf = gpd.read_file(affected_geojson_file)

    if gdf.empty:
        raise ValueError("File vùng ngập ảnh hưởng không có dữ liệu.")

    if not all(gdf.geometry.geom_type == "Point"):
        raise ValueError("Toàn bộ geometry trong file vùng ngập phải là Point.")

    with rasterio.open(ref_raster_path) as src:
        # đồng bộ CRS theo raster tham chiếu
        if gdf.crs != src.crs:
            print(f"-> Chuyển đổi hệ tọa độ từ {gdf.crs} sang {src.crs}...")
            gdf = gdf.to_crs(src.crs)

        # buffer để giảm lốm đốm
        if buffer_radius > 0:
            print(f"-> Đang tạo buffer bán kính {buffer_radius} m...")
            gdf["geometry"] = gdf.geometry.buffer(buffer_radius)

        transform = src.transform
        out_shape = (src.height, src.width)
        meta = src.meta.copy()

        print("-> Đang rasterize...")
        raster = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True
        )

        meta.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="lzw"
        )

        print("-> Đang lưu file TIFF...")
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(raster, 1)

    print(f"✅ HOÀN TẤT! Đã lưu file TIFF tại: {out_tif}")

print("\n===== XONG TOÀN BỘ QUY TRÌNH =====")
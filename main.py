"""
main.py — Pipeline hoàn chỉnh: Dự báo AI lũ lụt → Xử lý không gian

Chạy tuần tự:
  1. code_ai.py  (huấn luyện + dự báo mực nước → results/flood_all_models/)
  2. code_process_spatial.py  (nội suy IDW + xuất GeoJSON/GPKG/TIF)

Ví dụ:
  # Chạy tất cả model, dự báo 7 ngày:
  python main.py --models all --forecast-days 7 \
      --train-end-date 2021-12-31 --test-end-date 2022-12-21 --save-pkl false

  # Chỉ chạy một số model:
  python main.py --models xgb etr catboost --forecast-days 7 \
      --train-end-date 2021-12-31 --test-end-date 2022-12-21 --save-pkl false

  # Bỏ qua bước AI (chỉ chạy spatial với file best_predict.csv đã có sẵn):
  python main.py --skip-ai
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────
# ĐƯỜNG DẪN CÁC FILE
# ──────────────────────────────────────────────
ROOT = Path(__file__).parent
CODE_AI = ROOT / "code_ai.py"
CODE_SPATIAL = ROOT / "code_process_spatial.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline dự báo lũ: AI model → xử lý không gian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Tham số cho code_ai.py ──
    ai_group = parser.add_argument_group("Tham số AI (code_ai.py)")
    ai_group.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        metavar="MODEL",
        help=(
            "Danh sách model cần chạy, hoặc 'all'. "
            "Ví dụ: --models xgb etr catboost  |  mặc định: all"
        ),
    )
    ai_group.add_argument(
        "--forecast-days",
        type=int,
        default=7,
        metavar="N",
        help="Số ngày dự báo (mặc định: 7)",
    )
    ai_group.add_argument(
        "--train-end-date",
        default="2021-12-31",
        metavar="YYYY-MM-DD",
        help="Ngày kết thúc tập huấn luyện (mặc định: 2021-12-31)",
    )
    ai_group.add_argument(
        "--test-end-date",
        default="2022-12-21",
        metavar="YYYY-MM-DD",
        help="Ngày kết thúc tập kiểm tra (mặc định: 2022-12-21)",
    )
    ai_group.add_argument(
        "--save-pkl",
        default="false",
        choices=["true", "false"],
        help="Lưu model dạng .pkl (mặc định: false)",
    )

    # ── Tùy chọn pipeline ──
    pipe_group = parser.add_argument_group("Tùy chọn pipeline")
    pipe_group.add_argument(
        "--skip-ai",
        action="store_true",
        help=(
            "Bỏ qua bước chạy code_ai.py "
            "(dùng khi best_predict.csv đã có sẵn)"
        ),
    )
    pipe_group.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Bỏ qua bước chạy code_process_spatial.py",
    )

    return parser.parse_args()


def run_step(script: Path, extra_args: list[str], step_name: str) -> None:
    """Chạy một script Python con, dừng pipeline nếu lỗi."""
    cmd = [sys.executable, str(script)] + extra_args
    print(f"\n{'='*60}")
    print(f"  BƯỚC: {step_name}")
    print(f"  Lệnh: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(
            f"\n[LỖI] {step_name} kết thúc với mã lỗi {result.returncode}. "
            "Dừng pipeline."
        )
        sys.exit(result.returncode)
    print(f"\n[OK] {step_name} hoàn tất.")


def main():
    args = parse_args()

    # ──────────────────────────────────────────
    # BƯỚC 1 — code_ai.py
    # ──────────────────────────────────────────
    if not args.skip_ai:
        if not CODE_AI.exists():
            print(
                f"[CẢNH BÁO] Không tìm thấy {CODE_AI}. "
                "Vui lòng upload lại file code_ai.py."
            )
            sys.exit(1)

        models_arg = args.models  # list, ví dụ ["all"] hoặc ["xgb","etr"]
        ai_args = [
            "--models", *models_arg,
            "--forecast-days", str(args.forecast_days),
            "--train-end-date", args.train_end_date,
            "--test-end-date", args.test_end_date,
            "--save-pkl", args.save_pkl,
        ]
        run_step(CODE_AI, ai_args, "Dự báo AI (code_ai.py)")
    else:
        print("\n[SKIP] Bỏ qua bước code_ai.py theo yêu cầu.")

    # ──────────────────────────────────────────
    # BƯỚC 2 — code_process_spatial.py
    # ──────────────────────────────────────────
    if not args.skip_spatial:
        if not CODE_SPATIAL.exists():
            print(f"[LỖI] Không tìm thấy {CODE_SPATIAL}.")
            sys.exit(1)
        run_step(CODE_SPATIAL, [], "Xử lý không gian (code_process_spatial.py)")
    else:
        print("\n[SKIP] Bỏ qua bước code_process_spatial.py theo yêu cầu.")

    print("\n" + "=" * 60)
    print("  PIPELINE HOÀN TẤT!")
    print("=" * 60)


if __name__ == "__main__":
    main()
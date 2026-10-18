"""Local Feodosyev interface; all mechanics remain in the existing solvers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import mimetypes
from pathlib import Path
import re
import sys
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
CAT_DIR = PROJECT_DIR / "cat meme"
# Capture the launch working directory, before any request or path changes.
DEFAULT_OUTPUT_DIR = Path.cwd().resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from feodosyev_curve import solve_uniform_curve, write_curve_csv
from feodosyev_single_point import GrantaSS316TensionLaw
from feodosyev_temperature_study import _PointEvaluator, _zero_pressure_origin
from interpolation import T_E, T_alpha, T_hard, thermal_strain

TEMPERATURE_MIN_C = max(float(T_E.min()), float(T_alpha.min()), float(T_hard.min()))
TEMPERATURE_MAX_C = min(float(T_E.max()), float(T_alpha.max()), float(T_hard.max()))
MAX_SAMPLE_COUNT = 5000
CONFIG_FORMAT = "feodosyev_section_9_3_settings"
UNITS = {"length": "mm", "temperature": "degC", "strain": "%", "k": "dimensionless"}
DEFAULTS = {
    "radius_mm": 100.0, "inner_thickness_mm": 1.0, "outer_thickness_mm": 2.0,
    "inner_temperature_c": 538.0, "outer_temperature_c": 20.0, "k": 0.5,
    "auto_strain_range": True, "start_strain_percent": 0.025,
    "max_strain_percent": 3.0, "sample_count": 481,
}
STATIC_FILES = {
    "/": APP_DIR / "index.html", "/index.html": APP_DIR / "index.html",
    "/styles.css": APP_DIR / "styles.css", "/app.js": APP_DIR / "app.js",
    "/assets/generating.gif": APP_DIR / "assets" / "generating.gif",
    "/assets/cat-ok.png": CAT_DIR / "cat_ok.png",
    "/assets/cat-vegetable.jpg": CAT_DIR / "cat_vegetable.jpg",
}
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
STATE_LOCK = threading.Lock()
CALCULATION_LOCK = threading.Lock()
EXPORT_LOCK = threading.Lock()
LATEST_RESULT: dict[str, Any] | None = None


class UserFacingError(Exception):
    """A validation/solver message that can be shown in the browser."""


def validate_settings(raw: Any) -> dict[str, Any]:
    """Validate everything before returning a new, independent settings object."""
    if not isinstance(raw, dict):
        raise UserFacingError("settings 必須是 JSON 物件。")
    if set(raw) != set(DEFAULTS):
        raise UserFacingError("settings 必須包含全部模型與掃描欄位，且不可含未知欄位。")
    result = {}
    for key in DEFAULTS:
        value = raw[key]
        if key == "auto_strain_range":
            if not isinstance(value, bool):
                raise UserFacingError("Auto Strain Range 必須是布林值。")
            result[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UserFacingError(f"{key} 必須是 JSON 數字。")
        try:
            number = float(value)
        except OverflowError as exc:
            raise UserFacingError(f"{key} 必須是有限數值。") from exc
        if not math.isfinite(number):
            raise UserFacingError(f"{key} 必須是有限數值。")
        result[key] = number
    if min(result[key] for key in ("radius_mm", "inner_thickness_mm", "outer_thickness_mm")) <= 0:
        raise UserFacingError("半徑與兩個厚度都必須大於 0 mm。")
    for key in ("inner_temperature_c", "outer_temperature_c"):
        if not TEMPERATURE_MIN_C <= result[key] <= TEMPERATURE_MAX_C:
            raise UserFacingError(f"{key} 超出材料溫度範圍：{TEMPERATURE_MIN_C:g} 至 {TEMPERATURE_MAX_C:g} °C。")
    count = result["sample_count"]
    if not count.is_integer() or not 2 <= count <= MAX_SAMPLE_COUNT:
        raise UserFacingError(f"Sample Count 必須是 2 至 {MAX_SAMPLE_COUNT} 的整數（包含兩端）。")
    result["sample_count"] = int(count)
    if not result["auto_strain_range"] and result["max_strain_percent"] <= result["start_strain_percent"]:
        raise UserFacingError("End Strain (%) 必須大於 Start Strain (%)。")
    return result


def import_settings(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("format") != CONFIG_FORMAT:
        raise UserFacingError(f"JSON format 必須是 {CONFIG_FORMAT}；不接受 1D 塑性初始狀態。")
    if raw.get("units") != UNITS:
        raise UserFacingError("JSON units 必須是 length=mm、temperature=degC、strain=%、k=dimensionless。")
    return validate_settings(raw.get("settings"))


def generate_result(payload: dict[str, Any]) -> dict[str, Any]:
    global LATEST_RESULT
    if not CALCULATION_LOCK.acquire(blocking=False):
        raise UserFacingError("已有一個計算正在執行，請等待完成。")
    try:
        with STATE_LOCK:
            LATEST_RESULT = None
        settings = validate_settings(payload)
        try:
            arguments = dict(
                radius_m=settings["radius_mm"] * 1e-3,
                inner_thickness_m=settings["inner_thickness_mm"] * 1e-3,
                outer_thickness_m=settings["outer_thickness_mm"] * 1e-3,
                inner_material=GrantaSS316TensionLaw(settings["inner_temperature_c"]),
                outer_material=GrantaSS316TensionLaw(settings["outer_temperature_c"]),
                inner_thermal_strain=thermal_strain(settings["inner_temperature_c"]),
                outer_thermal_strain=thermal_strain(settings["outer_temperature_c"]),
                k=settings["k"],
            )
            origin = None
            start = settings["start_strain_percent"] / 100
            if settings["auto_strain_range"]:
                origin = _zero_pressure_origin(
                    _PointEvaluator(arguments), arguments["inner_thermal_strain"],
                    arguments["outer_thermal_strain"], 17,
                )
                start = float(origin["zero_pressure_hoop_strain"])
            curve = solve_uniform_curve(
                start_hoop_strain=start,
                end_hoop_strain=settings["max_strain_percent"] / 100,
                point_count=settings["sample_count"], **arguments,
            )
        except (ValueError, ArithmeticError) as exc:
            raise UserFacingError(f"無法取得可靠解（k={settings['k']:g}）：{exc}") from exc
        summary = {**asdict(curve.diagnostics), "zero_pressure_origin": origin,
                   "actual_start_strain_percent": curve.start_hoop_strain * 100,
                   "actual_end_strain_percent": curve.end_hoop_strain * 100}
        result_id = uuid.uuid4().hex
        configuration = {"format": CONFIG_FORMAT, "units": dict(UNITS),
                         "settings": settings, "summary": summary}
        with STATE_LOCK:
            LATEST_RESULT = {"result_id": result_id, "curve": curve, "configuration": configuration}
        return {"result_id": result_id, "point_count": len(curve.points),
                "strain_percent": [p.solution.common_total_hoop_strain * 100 for p in curve.points],
                "pressure_mpa": [p.solution.gas_pressure_pa / 1e6 for p in curve.points],
                "summary": summary}
    finally:
        CALCULATION_LOCK.release()


def _safe_export_target(directory_text: Any, filename_text: Any, extension: str) -> Path:
    if not isinstance(directory_text, str) or not directory_text.strip():
        raise UserFacingError("File Path 不可留白。")
    if not isinstance(filename_text, str) or not filename_text.strip():
        raise UserFacingError(f"{extension.upper()} 檔名不可留白。")

    filename = filename_text.strip()
    if filename.lower().endswith(f".{extension}"):
        filename = filename[: -(len(extension) + 1)]
    if (
        not filename
        or filename in {".", ".."}
        or INVALID_FILENAME_CHARS.search(filename)
        or Path(filename).name != filename
    ):
        raise UserFacingError(
            f"{extension.upper()} 檔名無效；請只輸入檔名，不要包含路徑或特殊字元。"
        )

    directory = Path(directory_text.strip()).expanduser()
    if not directory.is_dir():
        raise UserFacingError(f"File Path 不存在或不是資料夾：{directory}")

    target = directory / f"{filename}.{extension}"
    if target.exists():
        raise UserFacingError(f"為避免覆寫既有研究結果，檔案已存在：{target}")
    return target


def _save_plot(history: list[dict[str, Any]], target: Path) -> None:
    x_values = [row["epsilon_theta"] * 100.0 for row in history]
    y_values = [row["p_g"] / 1e6 for row in history]
    figure = Figure(figsize=(9, 5.5), dpi=180)
    axes = figure.subplots()
    axes.plot(x_values, y_values, color="#287fbd", linewidth=2.0)
    axes.scatter([x_values[0]], [y_values[0]], color="#287fbd", s=28, zorder=3)
    axes.set_xlabel(r"Total Hoop Strain $\Delta R/R$ (%)")
    axes.set_ylabel(r"Calculated Gas Pressure $p_g$ (MPa)")
    axes.grid(True, alpha=0.35)
    figure.tight_layout()
    figure.savefig(target, format="png")
    figure.clear()


def save_result(payload: dict[str, Any], extension: str) -> Path:
    if extension not in {"png", "csv", "json"}:
        raise UserFacingError("不支援的輸出格式。")
    result_id = payload.get("result_id")
    with STATE_LOCK:
        result = LATEST_RESULT
        if result is None or result_id != result["result_id"]:
            raise UserFacingError("目前沒有可儲存的成功計算結果，請先按 Generate。")
        curve = result["curve"]

    target = _safe_export_target(
        payload.get("file_path"), payload.get("filename"), extension
    )
    with EXPORT_LOCK:
        if target.exists():
            raise UserFacingError(f"為避免覆寫既有研究結果，檔案已存在：{target}")
        try:
            if extension == "csv":
                write_curve_csv(curve, target)
            elif extension == "json":
                target.write_text(
                    json.dumps(result["configuration"], ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
            else:
                _save_plot([{"epsilon_theta": p.solution.common_total_hoop_strain, "p_g": p.solution.gas_pressure_pa} for p in curve.points], target)
        except OSError as exc:
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
            raise UserFacingError(f"無法寫入 {extension.upper()}：{exc}") from exc
    return target


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "FeodosyevLocal"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise UserFacingError("無效的請求長度。") from exc
        if length <= 0 or length > 4 * 1024 * 1024:
            raise UserFacingError("請求內容為空或超過 4 MB。")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UserFacingError("請求不是有效的 JSON。") from exc
        if not isinstance(payload, dict):
            raise UserFacingError("請求內容必須是 JSON 物件。")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/api/config":
            self._send_json(
                HTTPStatus.OK,
                {
                    "default_output_path": str(DEFAULT_OUTPUT_DIR),
                    "temperature_min_c": TEMPERATURE_MIN_C,
                    "temperature_max_c": TEMPERATURE_MAX_C,
                    "defaults": DEFAULTS,
                    "max_sample_count": MAX_SAMPLE_COUNT,
                },
            )
            return

        file_path = STATIC_FILES.get(path)
        if file_path is None or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/import":
                settings = import_settings(payload)
                self._send_json(HTTPStatus.OK, {"ok": True, "settings": settings})
                return
            if path == "/api/generate":
                response = generate_result(payload)
                self._send_json(HTTPStatus.OK, {"ok": True, **response})
                return
            if path in {"/api/save/png", "/api/save/csv", "/api/save/json"}:
                extension = path.rsplit("/", 1)[-1]
                target = save_result(payload, extension)
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": True, "saved_path": str(target)},
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "找不到 API。"})
        except UserFacingError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except (FileNotFoundError, PermissionError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": f"檔案或路徑錯誤：{exc}"},
            )
        except Exception:
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "ok": False,
                    "error": (
                        "Numerical calculation failed: 程式發生未預期錯誤；"
                        "技術細節已寫入後端終端機。"
                    ),
                },
            )

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feodosyev section 9.3 local HTML interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Feodosyev section 9.3 interface: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

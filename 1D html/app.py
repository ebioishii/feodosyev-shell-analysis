"""Local web interface for the existing 1D double-shell solver."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import sys
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
CAT_DIR = PROJECT_DIR / "cat meme"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from heat_then_pressurize import (  # noqa: E402
    AUTO_MAX_POINTS,
    AUTO_MAX_TOTAL_STRAIN,
    HEATING_MAX_STEP_C,
    INITIAL_TEMPERATURE_C,
    admissible_epsilon_range,
    automatic_strain_history,
    shell_step,
    strain_history,
    validate_geometry,
    write_pressure_csv,
)
from interpolation import T_E, T_alpha, T_hard, E_at, flow_stress, thermal_strain  # noqa: E402
from known_initial_pressurize import check_initial_equilibrium  # noqa: E402


TEMPERATURE_MIN_C = max(float(T_E.min()), float(T_alpha.min()), float(T_hard.min()))
TEMPERATURE_MAX_C = min(float(T_E.max()), float(T_alpha.max()), float(T_hard.max()))

STATIC_FILES = {
    "/": APP_DIR / "index.html",
    "/index.html": APP_DIR / "index.html",
    "/styles.css": APP_DIR / "styles.css",
    "/app.js": APP_DIR / "app.js",
    "/assets/generating.gif": CAT_DIR / "oiiaioiiiai.gif",
    "/assets/cat-ok.png": CAT_DIR / "cat_ok.png",
    "/assets/cat-vegetable.jpg": CAT_DIR / "cat_vegetable.jpg",
}

INITIAL_STATE_FIELDS = (
    "R_m",
    "h_i_m",
    "h_o_m",
    "T_i_C",
    "T_o_C",
    "p_start_Pa",
    "epsilon_theta0",
    "epsilon_p_i",
    "kappa_i",
    "epsilon_p_o",
    "kappa_o",
)

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

STATE_LOCK = threading.Lock()
CALCULATION_LOCK = threading.Lock()
EXPORT_LOCK = threading.Lock()
LATEST_RESULT: dict[str, Any] | None = None


class UserFacingError(Exception):
    """An error whose message is safe and useful in the browser."""


def _number(payload: dict[str, Any], key: str, label: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise UserFacingError(f"{label} 必須是數字。")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise UserFacingError(f"{label} 必須是數字。") from exc
    if not math.isfinite(result):
        raise UserFacingError(f"{label} 必須是有限數值。")
    return result


def _validate_inputs(payload: dict[str, Any]) -> tuple[float, float, float, float, float, bool]:
    radius_mm = _number(payload, "radius_mm", "Common Radius")
    inner_thickness_mm = _number(payload, "inner_thickness_mm", "Inner Thickness")
    outer_thickness_mm = _number(payload, "outer_thickness_mm", "Outer Thickness")
    inner_temperature_c = _number(payload, "inner_temperature_c", "Inner Temperature")
    outer_temperature_c = _number(payload, "outer_temperature_c", "Outer Temperature")

    if min(radius_mm, inner_thickness_mm, outer_thickness_mm) <= 0:
        raise UserFacingError("Invalid geometry: 半徑與兩個厚度都必須大於 0 mm。")

    for label, temperature in (
        ("Inner Temperature", inner_temperature_c),
        ("Outer Temperature", outer_temperature_c),
    ):
        if not TEMPERATURE_MIN_C <= temperature <= TEMPERATURE_MAX_C:
            raise UserFacingError(
                "Temperature out of material data range: "
                f"{label} 必須介於 {TEMPERATURE_MIN_C:g}°C 與 "
                f"{TEMPERATURE_MAX_C:g}°C。"
            )

    heating = payload.get("heating_from_20")
    if not isinstance(heating, bool):
        raise UserFacingError("Heating from 20°C 的開關值無效。")

    radius_m = radius_mm * 1e-3
    inner_thickness_m = inner_thickness_mm * 1e-3
    outer_thickness_m = outer_thickness_mm * 1e-3
    validate_geometry(radius_m, inner_thickness_m, outer_thickness_m)
    return (
        radius_m,
        inner_thickness_m,
        outer_thickness_m,
        inner_temperature_c,
        outer_temperature_c,
        heating,
    )


def _check_material_temperature(temperature_c: float) -> None:
    E_at(temperature_c)
    thermal_strain(temperature_c)
    flow_stress(temperature_c, 0.0)


def _run_strain_sweep(
    initial_state: dict[str, Any], options: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    automatic = options.get("auto_strain_range", True)
    if not isinstance(automatic, bool):
        raise UserFacingError("Auto Strain Range 的開關值無效。")

    data = initial_state
    state_i = (data["epsilon_p_i"], data["kappa_i"])
    state_o = (data["epsilon_p_o"], data["kappa_o"])
    arguments = (
        data["T_i_C"], data["T_o_C"],
        data["R_m"], data["h_i_m"], data["h_o_m"], state_i, state_o,
    )
    if automatic:
        return automatic_strain_history(data["epsilon_theta0"], *arguments)

    end_strain = _number(options, "max_strain_percent", "Max Strain (%)") / 100.0
    count_value = _number(options, "sample_count", "Sample Count")
    if not count_value.is_integer() or not 2 <= count_value <= AUTO_MAX_POINTS:
        raise UserFacingError(
            f"Sample Count 必須是 2 至 {AUTO_MAX_POINTS} 之間的整數（包含起點與終點）。"
        )
    start_strain = data["epsilon_theta0"]
    if end_strain <= start_strain:
        raise UserFacingError(
            f"Max Strain (%) 必須大於加壓前初始應變 {start_strain * 100:.9g}%。"
        )
    material_cap = min(
        admissible_epsilon_range(data["T_i_C"], state_i)[1] - 1e-9,
        admissible_epsilon_range(data["T_o_C"], state_o)[1] - 1e-9,
    )
    if end_strain > AUTO_MAX_TOTAL_STRAIN:
        raise UserFacingError(
            f"Max Strain (%) 不可超過目前模型的 {AUTO_MAX_TOTAL_STRAIN * 100:g}% 小應變研究上限。"
        )
    if end_strain > material_cap:
        raise UserFacingError(
            "Material curve range exceeded: Max Strain (%) 不可超過此初始狀態的"
            f"材料有效上限 {material_cap * 100:.9g}%。"
        )
    history = strain_history(
        start_strain, end_strain, int(count_value), *arguments
    )
    return history, f"使用指定的 {int(count_value)} 個取樣點，達到總環向應變 {end_strain * 100:g}%"


def _calculate_from_heating(
    radius_m: float,
    inner_thickness_m: float,
    outer_thickness_m: float,
    inner_temperature_c: float,
    outer_temperature_c: float,
    options: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    for temperature in (
        INITIAL_TEMPERATURE_C,
        inner_temperature_c,
        outer_temperature_c,
    ):
        _check_material_temperature(temperature)

    maximum_temperature_change = max(
        abs(inner_temperature_c - INITIAL_TEMPERATURE_C),
        abs(outer_temperature_c - INITIAL_TEMPERATURE_C),
    )
    heating_points = max(
        2,
        math.ceil(maximum_temperature_change / HEATING_MAX_STEP_C) + 1,
    )
    state_i = (0.0, 0.0)
    state_o = (0.0, 0.0)
    initial: dict[str, Any] | None = None

    for index in range(heating_points):
        fraction = index / (heating_points - 1)
        temperature_i = INITIAL_TEMPERATURE_C + fraction * (
            inner_temperature_c - INITIAL_TEMPERATURE_C
        )
        temperature_o = INITIAL_TEMPERATURE_C + fraction * (
            outer_temperature_c - INITIAL_TEMPERATURE_C
        )
        try:
            initial = shell_step(
                0.0,
                temperature_i,
                temperature_o,
                radius_m,
                inner_thickness_m,
                outer_thickness_m,
                state_i,
                state_o,
            )
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(
                f"升溫步驟 {index + 1}/{heating_points}，"
                f"Ti={temperature_i:g}°C、To={temperature_o:g}°C：{exc}"
            ) from exc
        state_i = initial["state_i"]
        state_o = initial["state_o"]

    if initial is None:
        raise ArithmeticError("升溫計算沒有產生初始狀態")

    initial_state = {
        "format": "double_shell_initial_state_v1",
        "description": "State after heating from 20 C at zero gas pressure, before pressurization.",
        "R_m": radius_m, "h_i_m": inner_thickness_m, "h_o_m": outer_thickness_m,
        "T_i_C": inner_temperature_c, "T_o_C": outer_temperature_c,
        "p_start_Pa": initial["p_g"], "epsilon_theta0": initial["epsilon_theta"],
        "epsilon_p_i": state_i[0], "kappa_i": state_i[1],
        "epsilon_p_o": state_o[0], "kappa_o": state_o[1],
    }
    history, stop_reason = _run_strain_sweep(initial_state, options)
    if abs(history[0]["p_g"] - initial["p_g"]) > 100.0:
        raise ArithmeticError("第一個應變點沒有重現升溫後的零壓平衡")
    return history, stop_reason, initial_state


def _read_initial_state(raw_state: Any) -> dict[str, float]:
    if not isinstance(raw_state, dict):
        raise UserFacingError("Invalid JSON initial state: 請先用 Load JSON 選擇狀態檔。")
    if raw_state.get("format") != "double_shell_initial_state_v1":
        raise UserFacingError(
            "Invalid JSON initial state: format 必須是 double_shell_initial_state_v1。"
        )

    result: dict[str, float] = {}
    for field in INITIAL_STATE_FIELDS:
        value = raw_state.get(field)
        if isinstance(value, bool):
            raise UserFacingError(f"Invalid JSON initial state: {field} 必須是數字。")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise UserFacingError(
                f"Invalid JSON initial state: 缺少 {field} 或其值不是數字。"
            ) from exc
        if not math.isfinite(number):
            raise UserFacingError(
                f"Invalid JSON initial state: {field} 必須是有限數值。"
            )
        result[field] = number
    return result


def _assert_compatible_state(
    state: dict[str, float],
    radius_m: float,
    inner_thickness_m: float,
    outer_thickness_m: float,
    inner_temperature_c: float,
    outer_temperature_c: float,
) -> None:
    expected = {
        "R_m": radius_m,
        "h_i_m": inner_thickness_m,
        "h_o_m": outer_thickness_m,
        "T_i_C": inner_temperature_c,
        "T_o_C": outer_temperature_c,
    }
    labels = {
        "R_m": "Common Radius",
        "h_i_m": "Inner Thickness",
        "h_o_m": "Outer Thickness",
        "T_i_C": "Inner Temperature",
        "T_o_C": "Outer Temperature",
    }
    mismatches: list[str] = []
    for field, current in expected.items():
        absolute_tolerance = 1e-9 if field.startswith("T_") else 1e-12
        if not math.isclose(
            state[field], current, rel_tol=1e-9, abs_tol=absolute_tolerance
        ):
            mismatches.append(labels[field])
    if mismatches:
        raise UserFacingError(
            "Invalid JSON initial state: JSON 與目前輸入不一致："
            + "、".join(mismatches)
            + "。"
        )


def _calculate_from_known_state(
    raw_state: Any,
    radius_m: float,
    inner_thickness_m: float,
    outer_thickness_m: float,
    inner_temperature_c: float,
    outer_temperature_c: float,
    options: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    state = _read_initial_state(raw_state)
    _assert_compatible_state(
        state,
        radius_m,
        inner_thickness_m,
        outer_thickness_m,
        inner_temperature_c,
        outer_temperature_c,
    )
    check_initial_equilibrium(state)

    initial_state = {"format": "double_shell_initial_state_v1", **state}
    if isinstance(raw_state.get("description"), str):
        initial_state["description"] = raw_state["description"]
    history, stop_reason = _run_strain_sweep(initial_state, options)
    pressure_error = abs(history[0]["p_g"] - state["p_start_Pa"])
    pressure_tolerance = max(
        100.0,
        1e-5 * max(abs(state["p_start_Pa"]), abs(history[0]["p_g"])),
    )
    if pressure_error > pressure_tolerance:
        raise ArithmeticError("第一個應變點沒有重現 JSON 指定的初始壓力")
    return history, stop_reason, initial_state


def _friendly_solver_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "溫度" in message and "範圍" in message:
        return f"Temperature out of material data range: {message}"
    if "硬化" in message or "granta" in lowered or "塑性增量" in message:
        return f"Material curve range exceeded: {message}"
    if isinstance(exc, ArithmeticError):
        return f"Numerical calculation failed: {message}"
    return f"Numerical calculation failed: {message}"


def generate_result(payload: dict[str, Any]) -> dict[str, Any]:
    global LATEST_RESULT

    if not CALCULATION_LOCK.acquire(blocking=False):
        raise UserFacingError("已有一個計算正在執行，請等待完成。")

    try:
        with STATE_LOCK:
            LATEST_RESULT = None

        radius_m, inner_h_m, outer_h_m, inner_t_c, outer_t_c, heating = (
            _validate_inputs(payload)
        )
        try:
            if heating:
                history, stop_reason, initial_state = _calculate_from_heating(
                    radius_m,
                    inner_h_m,
                    outer_h_m,
                    inner_t_c,
                    outer_t_c,
                    payload,
                )
            else:
                history, stop_reason, initial_state = _calculate_from_known_state(
                    payload.get("initial_state"),
                    radius_m,
                    inner_h_m,
                    outer_h_m,
                    inner_t_c,
                    outer_t_c,
                    payload,
                )
        except UserFacingError:
            raise
        except (ValueError, ArithmeticError) as exc:
            raise UserFacingError(_friendly_solver_error(exc)) from exc

        result_id = uuid.uuid4().hex
        result = {
            "result_id": result_id,
            "history": history,
            "initial_state": initial_state,
            "stop_reason": stop_reason,
        }
        with STATE_LOCK:
            LATEST_RESULT = result

        return {
            "result_id": result_id,
            "strain_percent": [row["epsilon_theta"] * 100.0 for row in history],
            "pressure_mpa": [row["p_g"] / 1e6 for row in history],
            "point_count": len(history),
            "initial_strain": history[0]["epsilon_theta"],
            "stop_reason": stop_reason,
        }
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
        history = result["history"]

    target = _safe_export_target(
        payload.get("file_path"), payload.get("filename"), extension
    )
    with EXPORT_LOCK:
        if target.exists():
            raise UserFacingError(f"為避免覆寫既有研究結果，檔案已存在：{target}")
        try:
            if extension == "csv":
                write_pressure_csv(history, target)
            elif extension == "json":
                target.write_text(
                    json.dumps(result["initial_state"], ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
            else:
                _save_plot(history, target)
        except OSError as exc:
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
            raise UserFacingError(f"無法寫入 {extension.upper()}：{exc}") from exc
    return target


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "DoubleShellLocal/1.0"

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
                    "max_total_strain_percent": AUTO_MAX_TOTAL_STRAIN * 100.0,
                    "max_sample_count": AUTO_MAX_POINTS,
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
    parser = argparse.ArgumentParser(description="1D double-shell local HTML interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"1D double-shell interface: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

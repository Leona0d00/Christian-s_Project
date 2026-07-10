#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主脚本：实时中文语音转文字 GUI 应用

启动后显示一个窗口：
- 点击"开始识别"按钮启动实时语音转录
- 实时字幕显示在文本框中
- 点击"停止识别"结束转录并显示标点校正后的完整文本

用法:
    conda run -n christian python script/main.py
"""

import sys
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.Transcribe import (
    RealTimeTranscriber,
    list_microphones,
    auto_detect_microphone,
    select_device,
    list_available_devices,
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    POLL_INTERVAL_MS,
)

# 尝试获取合适的字体
FONT_FAMILY = "Microsoft YaHei"
FONT_SIZE = 14


class SpeechToTextApp:
    """实时语音转文字 GUI 应用。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(500, 400)

        # 状态
        self._transcriber = None
        self._is_recording = False
        self._recording_start_time = None
        self._loading_dots = 0

        # 检测可用设备
        self._available_devices = list_available_devices()
        self._device = select_device("auto")
        print(f"[信息] 使用设备: {self._device}")
        if self._available_devices.get("cuda"):
            print(f"[信息] CUDA 可用: {self._available_devices.get('cuda_device_name', 'Unknown')}")

        # 配置样式
        self._setup_style()

        # 构建 UI
        self._build_ui()

        # 加载麦克风设备列表
        self._refresh_microphones()

        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================= 样式配置 =================

    def _setup_style(self):
        """配置 ttk 自定义样式。"""
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        if "clam" in available_themes:
            self.style.theme_use("clam")

        # 配色方案
        BG = "#F5F6FA"
        PRIMARY = "#2C3E50"
        ACCENT = "#3498DB"
        SUCCESS = "#27AE60"
        DANGER = "#E74C3C"
        TEXT = "#2C3E50"

        self._colors = {
            "bg": BG, "primary": PRIMARY, "accent": ACCENT,
            "success": SUCCESS, "danger": DANGER, "text": TEXT,
        }

        # Frame
        self.style.configure("TFrame", background=BG)
        self.style.configure("Control.TFrame", background=BG)
        self.style.configure("Status.TFrame", background=PRIMARY)

        # Label
        self.style.configure("TLabel", background=BG, foreground=TEXT)
        self.style.configure("Status.TLabel", background=PRIMARY, foreground="white")
        self.style.configure("Info.TLabel", background=PRIMARY, foreground="#BDC3C7")

        # Button base
        self.style.configure("TButton",
            borderwidth=0, focusthreshold=0, padding=(15, 6))
        self.style.map("TButton",
            background=[("disabled", "#BDC3C7")],
            foreground=[("disabled", "#ECF0F1")])

        # Primary (start)
        self.style.configure("Primary.TButton",
            background=SUCCESS, foreground="white", padding=(15, 8))
        self.style.map("Primary.TButton",
            background=[("active", "#219A52"), ("disabled", "#BDC3C7")])

        # Danger (stop)
        self.style.configure("Danger.TButton",
            background=DANGER, foreground="white", padding=(15, 8))
        self.style.map("Danger.TButton",
            background=[("active", "#C0392B"), ("disabled", "#BDC3C7")])

        # Secondary (clear)
        self.style.configure("Secondary.TButton",
            background="#95A5A6", foreground="white", padding=(15, 8))
        self.style.map("Secondary.TButton",
            background=[("active", "#7F8C8D"), ("disabled", "#BDC3C7")])

        # Combobox
        self.style.configure("TCombobox",
            fieldbackground="white", background="white", arrowcolor=PRIMARY)

        # Separator
        self.style.configure("TSeparator", background="#D5D8DC")

    # ================= UI 构建 =================

    def _build_ui(self):
        """构建 GUI 界面。"""
        # 全局字体
        default_font = (FONT_FAMILY, FONT_SIZE)
        self.root.option_add("*Font", default_font)
        self.root.configure(bg=self._colors["bg"])

        # ===== 顶部控制栏 =====
        control_frame = ttk.Frame(self.root, padding="15", style="Control.TFrame")
        control_frame.pack(fill=tk.X)

        # 开始按钮
        self._start_btn = ttk.Button(
            control_frame, text="开始识别",
            command=self._start_recording,
            style="Primary.TButton",
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 8))

        # 停止按钮
        self._stop_btn = ttk.Button(
            control_frame, text="停止识别",
            command=self._stop_recording,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self._stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        # 清空按钮
        self._clear_btn = ttk.Button(
            control_frame, text="清空",
            command=self._clear_text,
            state=tk.DISABLED,
            style="Secondary.TButton",
        )
        self._clear_btn.pack(side=tk.LEFT, padx=(0, 15))

        # 录音状态指示器（右侧）
        indicator_frame = ttk.Frame(control_frame, style="Control.TFrame")
        indicator_frame.pack(side=tk.RIGHT, padx=(20, 0))

        self._indicator_canvas = tk.Canvas(
            indicator_frame, width=14, height=14,
            bg=self._colors["bg"], highlightthickness=0,
        )
        self._indicator_dot = self._indicator_canvas.create_oval(
            1, 1, 13, 13, fill="#BDC3C7", outline="")

        self._timer_label = ttk.Label(
            indicator_frame, text="00:00",
            font=(FONT_FAMILY, FONT_SIZE - 2), background=self._colors["bg"],
        )

        # 设备选择（控制栏中间）
        ttk.Label(control_frame, text="麦克风:").pack(side=tk.LEFT, padx=(10, 5))
        self._device_var = tk.StringVar()
        self._device_combo = ttk.Combobox(
            control_frame, textvariable=self._device_var,
            state="readonly", width=28,
        )
        self._device_combo.pack(side=tk.LEFT, padx=(0, 10))
        self._device_combo.bind("<<ComboboxSelected>>", self._on_device_selected)

        ttk.Label(control_frame, text="计算:").pack(side=tk.LEFT, padx=(10, 5))
        self._compute_var = tk.StringVar(value=self._device)
        self._compute_combo = ttk.Combobox(
            control_frame, textvariable=self._compute_var,
            state="readonly", width=5,
            values=["cpu", "cuda"] if self._available_devices.get("cuda") else ["cpu"],
        )
        self._compute_combo.pack(side=tk.LEFT)

        # ===== 分隔线 =====
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)

        # ===== 字幕显示区域 =====
        subtitle_frame = ttk.Frame(self.root, padding="10")
        subtitle_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(subtitle_frame, text="实时字幕:").pack(anchor=tk.W)

        self._text_area = scrolledtext.ScrolledText(
            subtitle_frame,
            wrap=tk.WORD,
            font=(FONT_FAMILY, FONT_SIZE + 4),
            padx=15, pady=15,
            bg="white", fg=self._colors["text"],
            insertbackground=self._colors["accent"],
            selectbackground=self._colors["accent"],
            selectforeground="white",
            borderwidth=1, relief=tk.SOLID,
        )
        self._text_area.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # ===== 分隔线 =====
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=10, side=tk.BOTTOM)

        # ===== 状态栏 =====
        status_frame = ttk.Frame(self.root, padding="8", style="Status.TFrame")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self._status_label = ttk.Label(
            status_frame, text="状态：就绪",
            font=(FONT_FAMILY, FONT_SIZE - 2),
            style="Status.TLabel",
        )
        self._status_label.pack(side=tk.LEFT)

        self._info_label = ttk.Label(
            status_frame, text="",
            font=(FONT_FAMILY, FONT_SIZE - 2),
            style="Info.TLabel",
        )
        self._info_label.pack(side=tk.RIGHT)

    # ================= 麦克风管理 =================

    def _refresh_microphones(self):
        """刷新麦克风设备列表。"""
        devices = list_microphones()
        if devices:
            self._mic_devices = devices
            device_names = []
            for d in devices:
                default_mark = " (默认)" if d.get("is_default") else ""
                device_names.append(f"[{d['index']}] {d['name']}{default_mark}")
            self._device_combo["values"] = device_names

            default = auto_detect_microphone()
            if default:
                for i, d in enumerate(devices):
                    if d["index"] == default["index"]:
                        self._device_combo.current(i)
                        break
            else:
                self._device_combo.current(0)
        else:
            self._mic_devices = []
            self._device_combo["values"] = ["(无可用麦克风)"]
            self._device_combo.current(0)
            self.set_status("状态：未检测到麦克风设备")

    def _on_device_selected(self, event=None):
        """设备选择变更。"""
        idx = self._device_combo.current()
        if idx >= 0 and idx < len(self._mic_devices):
            mic = self._mic_devices[idx]
            self.set_status(f"状态：已选择设备 [{mic['index']}] {mic['name']}")

    def _get_selected_device_index(self):
        """获取当前选中的麦克风设备索引。"""
        idx = self._device_combo.current()
        if 0 <= idx < len(self._mic_devices):
            return self._mic_devices[idx]["index"]
        return None

    # ================= 录音控制 =================

    def _start_recording(self):
        """开始录音识别。"""
        if self._is_recording:
            return

        device_index = self._get_selected_device_index()
        device = self._compute_var.get()

        # 根据设备自适应选择低延迟 chunk 尺寸
        if device == "cuda":
            chunk_size = [0, 5, 3]   # CUDA 低延迟: 300ms chunk + 180ms lookahead
        else:
            chunk_size = [0, 8, 4]   # CPU 低延迟: 480ms chunk + 240ms lookahead

        self.set_status("状态：正在加载模型，请稍候")
        self._set_indicator_state("loading")
        self._start_btn.config(state=tk.DISABLED)
        self._loading_dots = 0
        self._animate_loading()
        self.root.update()

        def _init_and_start():
            try:
                self._transcriber = RealTimeTranscriber(
                    device=device,
                    device_index=device_index,
                    chunk_size=chunk_size,
                )

                if not self._transcriber.start():
                    self.root.after(0, lambda: self._on_start_failed(
                        "无法启动转录，请检查麦克风或模型。"))
                    return

                self._is_recording = True
                self.root.after(0, self._on_recording_started)
            except Exception as e:
                self.root.after(0, lambda: self._on_start_failed(str(e)))

        thread = threading.Thread(target=_init_and_start, daemon=True)
        thread.start()

    def _on_recording_started(self):
        """录音开始后的 UI 更新。"""
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._text_area.delete(1.0, tk.END)
        self._update_clear_button_state()
        self.set_status("状态：录音中")
        self._info_label.config(text=f"设备: {self._device}")

        # 启动指示器和计时器
        self._set_indicator_state("recording")
        self._recording_start_time = time.time()
        self._update_timer()

        # 开始轮询结果
        self._poll_results()

    def _on_start_failed(self, error_msg: str):
        """启动失败处理。"""
        self._is_recording = False
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._set_indicator_state("idle")
        self.set_status("状态：启动失败")
        messagebox.showerror("转录启动失败", f"无法启动实时转录:\n\n{error_msg}")

    def _stop_recording(self):
        """停止录音识别。"""
        if not self._is_recording or self._transcriber is None:
            return

        self.set_status("状态：正在停止...")
        self._stop_btn.config(state=tk.DISABLED)
        self.root.update()

        def _do_stop():
            try:
                final_text = self._transcriber.stop()
                self._is_recording = False
                self.root.after(0, lambda: self._on_recording_stopped(final_text))
            except Exception as e:
                self.root.after(0, lambda: self._on_stop_failed(str(e)))

        thread = threading.Thread(target=_do_stop, daemon=True)
        thread.start()

    def _on_recording_stopped(self, final_text: str):
        """录音停止后的 UI 更新。"""
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._transcriber = None
        self._set_indicator_state("idle")

        # 停止计时器
        self._recording_start_time = None
        self._timer_label.config(text="00:00")

        # 显示最终校正字幕
        if final_text:
            existing = self._text_area.get(1.0, tk.END).strip()
            if existing:
                self._text_area.insert(tk.END, "\n\n" + "═" * 40 + "\n")
            self._text_area.insert(tk.END, f"最终校正字幕:\n{final_text}")
            self._text_area.see(tk.END)

        self.set_status("状态：就绪（可重新开始或清空文本）")
        self._info_label.config(text="")
        self._update_clear_button_state()

    def _on_stop_failed(self, error_msg: str):
        """停止失败处理。"""
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._set_indicator_state("idle")
        self.set_status("状态：停止时出错")
        messagebox.showwarning("停止异常", f"停止转录时出错:\n\n{error_msg}")

    # ================= 结果轮询 =================

    def _poll_results(self):
        """轮询结果队列（定时由 GUI 线程调用）。"""
        if not self._is_recording or self._transcriber is None:
            return

        try:
            results = self._transcriber.get_results()
            for text in results:
                if text:
                    self._text_area.insert(tk.END, text)
                    self._text_area.see(tk.END)
            if results:
                self._update_clear_button_state()
        except Exception as e:
            print(f"[警告] 轮询结果异常: {e}")

        if self._is_recording:
            self.root.after(POLL_INTERVAL_MS, self._poll_results)

    # ================= 清空按钮 =================

    def _clear_text(self):
        """清空文本区域。"""
        self._text_area.delete(1.0, tk.END)
        self._update_clear_button_state()

    def _update_clear_button_state(self):
        """根据文本内容和录音状态更新清空按钮。"""
        if self._is_recording:
            self._clear_btn.config(state=tk.DISABLED)
        else:
            content = self._text_area.get(1.0, tk.END).strip()
            self._clear_btn.config(state=tk.NORMAL if content else tk.DISABLED)

    # ================= 状态指示器与计时器 =================

    def _set_indicator_state(self, state: str):
        """设置录音指示器状态。"""
        colors = {
            "idle": "#BDC3C7",
            "loading": "#F39C12",
            "recording": "#E74C3C",
        }
        color = colors.get(state, "#BDC3C7")
        self._indicator_canvas.itemconfig(self._indicator_dot, fill=color)

        if state == "recording":
            self._indicator_canvas.pack(side=tk.LEFT, padx=(0, 5))
            self._timer_label.pack(side=tk.LEFT)
        elif state == "loading":
            self._indicator_canvas.pack(side=tk.LEFT, padx=(0, 5))
            self._timer_label.pack_forget()
        else:
            self._indicator_canvas.pack_forget()
            self._timer_label.pack_forget()

    def _update_timer(self):
        """更新计时器显示。"""
        if not self._is_recording or self._recording_start_time is None:
            return
        elapsed = int(time.time() - self._recording_start_time)
        mins, secs = elapsed // 60, elapsed % 60
        self._timer_label.config(text=f"{mins:02d}:{secs:02d}")
        if self._is_recording:
            self.root.after(1000, self._update_timer)

    def _animate_loading(self):
        """模型加载时的动画指示。"""
        if self._is_recording:
            return
        if self._start_btn.cget("state") == tk.NORMAL:
            return
        dots = "." * ((self._loading_dots % 3) + 1)
        self.set_status(f"状态：正在加载模型，请稍候{dots}")
        self._loading_dots += 1
        self.root.after(500, self._animate_loading)

    # ================= 工具方法 =================

    def set_status(self, text: str):
        """更新状态栏文字（线程安全）。"""
        try:
            self._status_label.config(text=text)
        except Exception:
            pass

    def _on_close(self):
        """窗口关闭事件处理。"""
        if self._is_recording:
            if messagebox.askyesno("确认退出", "转录正在进行中，确定要退出吗？"):
                if self._transcriber:
                    self._transcriber.stop()
                self._is_recording = False
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """主入口函数。"""
    print("=" * 50)
    print("  实时中文语音转文字")
    print("  基于 FunASR paraformer-zh-streaming")
    print("=" * 50)

    devices = list_available_devices()
    print(f"\n可用计算设备: {devices}")
    print(f"将使用设备: {select_device('auto')}")

    mics = list_microphones()
    if mics:
        print(f"\n可用麦克风设备 ({len(mics)}):")
        for m in mics:
            default = " [默认]" if m.get("is_default") else ""
            print(f"  [{m['index']}] {m['name']}{default}")
    else:
        print("\n警告: 未检测到麦克风设备。")

    print("\n正在启动 GUI...")

    root = tk.Tk()
    app = SpeechToTextApp(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        print("应用已退出。")

    return 0


if __name__ == "__main__":
    sys.exit(main())

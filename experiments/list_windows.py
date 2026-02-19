import ctypes
import ctypes.wintypes as wintypes


def list_visible_windows():
    user32 = ctypes.windll.user32
    foreground = user32.GetForegroundWindow()
    windows = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def enum_windows_proc(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.IsIconic(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if not title:
            return True

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        windows.append(
            {
                "hwnd": int(hwnd),
                "title": title,
                "pid": int(pid.value),
                "is_foreground": int(hwnd) == int(foreground),
            }
        )
        return True

    user32.EnumWindows(enum_windows_proc, 0)
    return windows


def main():
    windows = list_visible_windows()
    if not windows:
        print("No visible windows found.")
        return

    print("Visible windows (use HWND hex or title substring):")
    print("-" * 90)
    for w in windows:
        mark = "*" if w["is_foreground"] else " "
        print(f"{mark} HWND=0x{w['hwnd']:08X}  PID={w['pid']:>6}  TITLE={w['title']}")

    print("-" * 90)
    print("* = foreground window")


if __name__ == "__main__":
    main()


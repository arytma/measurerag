"""The chip's temperature on an Apple Silicon Mac, read without sudo through IOKit's HID event system.

    .venv/bin/python scripts/mac_temps.py      # every sensor, in degrees Celsius

`die_max()` is the hottest CPU die sensor (the PMU's tdie sensors). scripts/sweep_queue.py waits for it to fall before a
calibration probe, instead of waiting a fixed ten minutes."""
import ctypes, ctypes.util
from ctypes import c_void_p, c_int32, c_int64, c_double, c_char_p, byref
io = ctypes.cdll.LoadLibrary(ctypes.util.find_library("IOKit"))
cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
for f, res, args in (("CFDictionaryCreateMutable", c_void_p, [c_void_p, c_int64, c_void_p, c_void_p]), ("CFNumberCreate", c_void_p, [c_void_p, c_int32, c_void_p]),
                     ("CFStringCreateWithCString", c_void_p, [c_void_p, c_char_p, c_int32]), ("CFDictionarySetValue", None, [c_void_p, c_void_p, c_void_p]),
                     ("CFArrayGetCount", c_int64, [c_void_p]), ("CFArrayGetValueAtIndex", c_void_p, [c_void_p, c_int64]),
                     ("CFStringGetCString", ctypes.c_bool, [c_void_p, c_char_p, c_int64, c_int32]), ("CFRelease", None, [c_void_p])):
    fn = getattr(cf, f); fn.restype, fn.argtypes = res, args
for f, res, args in (("IOHIDEventSystemClientCreate", c_void_p, [c_void_p]), ("IOHIDEventSystemClientSetMatching", c_int32, [c_void_p, c_void_p]),
                     ("IOHIDEventSystemClientCopyServices", c_void_p, [c_void_p]), ("IOHIDServiceClientCopyProperty", c_void_p, [c_void_p, c_void_p]),
                     ("IOHIDServiceClientCopyEvent", c_void_p, [c_void_p, c_int64, c_int32, c_int64]), ("IOHIDEventGetFloatValue", c_double, [c_void_p, c_int32])):
    fn = getattr(io, f); fn.restype, fn.argtypes = res, args
UTF8 = 0x08000100
def s(x): return cf.CFStringCreateWithCString(None, x.encode(), UTF8)
def n(v): return cf.CFNumberCreate(None, 3, byref(c_int32(v)))
keycb = ctypes.addressof(ctypes.c_char.in_dll(cf, "kCFTypeDictionaryKeyCallBacks"))
valcb = ctypes.addressof(ctypes.c_char.in_dll(cf, "kCFTypeDictionaryValueCallBacks"))

def read() -> dict[str, float]:
    d = cf.CFDictionaryCreateMutable(None, 0, keycb, valcb)
    cf.CFDictionarySetValue(d, s("PrimaryUsagePage"), n(0xff00)); cf.CFDictionarySetValue(d, s("PrimaryUsage"), n(5))
    client = io.IOHIDEventSystemClientCreate(None)
    io.IOHIDEventSystemClientSetMatching(client, d)
    services = io.IOHIDEventSystemClientCopyServices(client)
    out = {}
    if not services:
        return out
    for i in range(cf.CFArrayGetCount(services)):
        sv = cf.CFArrayGetValueAtIndex(services, i)
        name = io.IOHIDServiceClientCopyProperty(sv, s("Product"))
        buf = ctypes.create_string_buffer(128)
        label = buf.value.decode() if name and cf.CFStringGetCString(name, buf, 128, UTF8) else f"sensor{i}"
        ev = io.IOHIDServiceClientCopyEvent(sv, 15, 0, 0)
        if ev:
            out[label] = io.IOHIDEventGetFloatValue(ev, 15 << 16)
            cf.CFRelease(ev)
    return out

def die_max() -> float | None:
    """The hottest CPU die sensor, or None where the sensors cannot be read (another machine, another OS)."""
    try:
        t = [v for k, v in read().items() if " tdie" in k and 0 < v < 150]
    except (OSError, AttributeError, ValueError):
        return None
    return max(t) if t else None


if __name__ == "__main__":
    t = read()
    print(len(t), "sensors")
    for k, v in sorted(t.items()): print(f"  {k:28} {v:6.1f}")

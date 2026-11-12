import time, shutil, subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright
out = Path("/Users/ashrit/Music/Logic/21M/project/Muse capture"); fr = Path("/tmp/pwframes"); shutil.rmtree(fr, ignore_errors=True); fr.mkdir()
FPS = 8; n = 0
def snap_for(pg, seconds):
    global n
    end = time.time() + seconds
    while time.time() < end:
        t = time.time(); pg.screenshot(path=str(fr / f"{n:05d}.png")); n += 1
        time.sleep(max(0, 1/FPS - (time.time() - t)))
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 600, "height": 1180}, device_scale_factor=2)
    pg.goto("http://localhost:8770/"); pg.wait_for_timeout(1500)
    t0 = time.time(); snap_for(pg, 5)
    pg.fill("#label", "midterm scratch session"); snap_for(pg, 1.5)
    pg.click("#big"); snap_for(pg, 9)
    pg.fill("#marktext", "section B"); snap_for(pg, 0.8); pg.click("#markbtn"); snap_for(pg, 9)
    pg.click("#big"); snap_for(pg, 5)
    real = time.time() - t0; b.close()
rate = n / real
mp4 = out / "recorder_page_capture.mp4"
subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",f"{rate:.3f}","-i",str(fr/"%05d.png"),"-vf","scale=900:-2:flags=lanczos,fps=24","-c:v","libx264","-preset","veryfast","-crf","20","-pix_fmt","yuv420p",str(mp4)], check=True)
print(mp4, n, "frames", round(real,1), "s", round(rate,2), "fps")

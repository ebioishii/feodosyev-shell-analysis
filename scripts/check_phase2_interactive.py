import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


async def check():
    edge = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    results = {}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, executable_path=edge)
        page = await browser.new_page(viewport={'width': 1200, 'height': 900})
        await page.goto('http://127.0.0.1:8765/curves_2d.html')
        await page.locator('.js-plotly-plot').wait_for()
        await page.wait_for_timeout(700)
        results['2d_loaded'] = await page.locator('.scatterlayer .trace').count() > 0
        await page.evaluate("""() => {
            const g = document.querySelector('.js-plotly-plot');
            window.hoverEvents = 0;
            g.on('plotly_hover', () => window.hoverEvents++);
        }""")
        points = page.locator('.scatterlayer .trace .points path')
        print('2D marker count:', await points.count(), flush=True)
        marker = await points.nth(100).bounding_box()
        await page.mouse.move(marker['x'] + marker['width'] / 2,
                              marker['y'] + marker['height'] / 2)
        await page.wait_for_timeout(800)
        results['2d_hover'] = await page.evaluate('window.hoverEvents > 0')
        await page.mouse.wheel(0, -400)
        await page.wait_for_timeout(400)
        results['2d_zoomed'] = await page.evaluate("""() => {
            const g = document.querySelector('.js-plotly-plot');
            return g.layout.xaxis?.range?.length === 2;
        }""")
        await page.goto('http://127.0.0.1:8765/surface_3d.html')
        await page.locator('.js-plotly-plot').wait_for()
        await page.wait_for_timeout(1200)
        results['3d_loaded'] = await page.locator('.gl-container canvas').count() > 0
        camera_before = await page.evaluate("""() =>
            JSON.stringify(document.querySelector('.js-plotly-plot')._fullLayout.scene.camera)
        """)
        box = await page.locator('.gl-container canvas').first.bounding_box()
        x, y = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.mouse.move(x + 120, y + 60, steps=15)
        await page.mouse.up()
        await page.wait_for_timeout(800)
        camera_after_drag = await page.evaluate("""() =>
            JSON.stringify(document.querySelector('.js-plotly-plot')._fullLayout.scene.camera)
        """)
        results['3d_rotated'] = camera_before != camera_after_drag
        await page.mouse.wheel(0, -450)
        await page.wait_for_timeout(700)
        camera_after_zoom = await page.evaluate("""() =>
            JSON.stringify(document.querySelector('.js-plotly-plot')._fullLayout.scene.camera)
        """)
        results['3d_zoomed'] = camera_after_zoom != camera_after_drag
        await page.evaluate("""() => {
            const g = document.querySelector('.js-plotly-plot');
            window.hoverEvents = 0;
            g.on('plotly_hover', () => window.hoverEvents++);
        }""")
        for fraction_x in [0.35, 0.45, 0.55, 0.65]:
            if await page.evaluate('window.hoverEvents > 0'):
                break
            for fraction_y in [0.35, 0.45, 0.55, 0.65]:
                await page.mouse.move(box['x'] + box['width'] * fraction_x,
                                      box['y'] + box['height'] * fraction_y)
                await page.wait_for_timeout(120)
        results['3d_hover'] = await page.evaluate('window.hoverEvents > 0')
        await page.screenshot(path='outputs/phase2_validation/interactive_3d.png')
        await browser.close()
    Path('outputs/phase2_validation/interactive_check.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8')
    print(results)


asyncio.run(check())

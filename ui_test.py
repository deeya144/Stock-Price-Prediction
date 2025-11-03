import time
import os
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    StaleElementReferenceException,
)
from webdriver_manager.chrome import ChromeDriverManager


@pytest.mark.ui
def test_streamlit_ui():
    """UI Automation test for Streamlit stock predictor dashboard"""

    # -------------------------------
    # 🚀 Setup Chrome WebDriver
    # -------------------------------
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_experimental_option("detach", True)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # -------------------------------
        # 🌐 Open Streamlit App
        # -------------------------------
        driver.get("http://localhost:8501")
        time.sleep(5)

        print("✅ Page loaded successfully!")

        buttons = driver.find_elements(By.TAG_NAME, "button")
        print(f"🧭 Found {len(buttons)} buttons on the page.")
        assert len(buttons) > 0, "❌ No buttons found on the dashboard!"

      
        clicked = False
        for _ in range(3): 
            try:
                buttons = driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    if "Predict" in btn.text or "Simulate" in btn.text:
                        driver.execute_script("arguments[0].scrollIntoView(true);", btn)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", btn)
                        print(f"✅ Clicked button using JS: '{btn.text}'")
                        clicked = True
                        break
                if clicked:
                    break
            except (ElementClickInterceptedException, StaleElementReferenceException):
                print("⚠️ Element became stale or blocked — retrying...")
                time.sleep(2)

        assert clicked, "❌ Could not find or click 'Predict & Simulate' button!"

   
        print("⏳ Waiting for charts to render...")
        time.sleep(10)

  
        if not os.path.exists("screenshots"):
            os.mkdir("screenshots")
        screenshot_path = "screenshots/ui_dashboard.png"
        driver.save_screenshot(screenshot_path)
        print(f"📸 Screenshot saved successfully: {screenshot_path}")

        print("🏁 UI Test Completed Successfully!")

    finally:
        driver.quit()
